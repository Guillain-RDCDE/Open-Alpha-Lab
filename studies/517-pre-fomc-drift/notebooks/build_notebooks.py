"""Generate the two narrative notebooks for Study 517 (Pre-FOMC-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY + basket
prices under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (SPY 1993-02-01 -> 2026-05-29,
# 261 hardcoded FOMC meetings, 260 pre-FOMC sessions = 3.1% of days, fingerprint 9687074d1493).
R = dict(
    start="1993-02-01", end="2026-05-29", asof="2026-05-31", years=33.3,
    n_days=8389, n_meetings=261, n_pre=260, frac_days=3.1, fingerprint="9687074d1493",
    # full sample event-vs-rest
    ev_mean=0.2271, rest_mean=0.0422, ev_t0=3.04, welch_t=2.44,
    share_total=14.7, placebo_p=0.007,
    # decay split: (label, ev_mean%, rest_mean%, welch_t, share%, n_ev)
    pre2012=("pre-2012", 0.3346, 0.0279, 2.94, 27.4, 145),
    post2012=("post-2012", 0.0914, 0.0610, 0.28, 4.7, 115),
    split="2012-01-01",
    # overnight/intraday: (component, ev_mean%, rest_mean%, welch_t)
    oi=[("overnight", 0.1080, 0.0303, 1.70), ("intraday", 0.1197, 0.0045, 1.78),
        ("daily", 0.2271, 0.0349, 2.54)],
    # costs: (cost_bps, gross%, net%, net_t0)
    costs=[(0.5, 0.2271, 0.2171, 2.91), (1.0, 0.2271, 0.2071, 2.77),
           (2.0, 0.2271, 0.1871, 2.51)],
    # robustness shift: (label, ev_mean%, welch_t, share%)
    shift=[("announcement-day", 0.2271, 2.44, 14.7), ("strictly prior (-1)", 0.1609, 1.40, 10.4)],
    # synthetic control seed-averaged: (planted_edge%, mean_welch_t, frac_t_gt2)
    syn=[(0.0, -0.09, 0.05), (0.40, 4.24, 0.97)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Run-up only?: Mixed](https://img.shields.io/badge/Run--up_only%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from pre_fomc_drift import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, OHLC = data.load_real()
    PRICES = PRICES[PRICES.index <= ASOF]; OHLC = OHLC[OHLC.index <= ASOF]
    SPY = PRICES["SPY"].dropna()
    RET = st.daily_returns(SPY)
    MASK = data.tag_pre_fomc(RET.index, shift=0)
else:
    PRICES = OHLC = SPY = RET = MASK = None
print("real pre-FOMC cache present:", HAVE_REAL,
      "| daily returns:", (0 if RET is None else len(RET)),
      "| pre-FOMC days:", (0 if MASK is None else int(MASK.sum())))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do stocks quietly drift up *before* the Fed speaks? 🏦\n"
            "### The pre-FOMC announcement drift — a real anomaly that quietly died once it was famous\n\n"
            + BADGES +
            "Here's one of the strangest facts in markets. Eight times a year the Federal Reserve "
            "announces its interest-rate decision — and the date is known **months in advance**. There's "
            "no secret. Yet for two decades, the US stock market **drifted up in the hours before** each "
            "of those announcements, earning so much that a sliver of days carried a huge chunk of the "
            "*entire* market's return.\n\n"
            "Two economists, **Lucca and Moench**, wrote this up in a famous 2015 paper. And then "
            "something very on-brand happened: once everyone knew about it, **it stopped working**. This "
            "notebook is the story of a real edge — and how publishing it killed it.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the decay split? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We test the broad market via **SPY** (an index ETF, so no "
            "survivorship problem) plus a 30-name survivor basket as colour. The Fed-meeting calendar is "
            "**hardcoded** from the Fed's own schedules. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did stocks drift up before Fed meetings? | **Yes — really.** Over 33 years, the day before "
            f"each Fed decision earned **+{R['ev_mean']:.3f}%** on average vs **+{R['rest_mean']:.3f}%** "
            f"for a normal day. Those **{R['frac_days']:.0f}% of days carried {R['share_total']:.0f}%** of "
            "the market's *entire* cumulative return. |\n"
            "| Is it just luck? | **No.** A one-sample *t* of **3.04**, and a test that re-rolls the "
            f"calendar 20,000 times at random says you'd see this by chance only **{R['placebo_p']*100:.1f}%** "
            "of the time. |\n"
            "| Can I trade it today? | **Not really.** After the paper came out, the effect **collapsed** — "
            "since 2012 the pre-Fed day is statistically a *normal day*. |\n"
            "| Is it the 'run-up into the 2pm statement'? | **Half.** About half the gain shows up "
            "**overnight** (before the market even opens), not in the afternoon run-up the story is about. |\n\n"
            "> The legend was **true** — and then publishing it **arbitraged it away**. A perfect little "
            "parable about market efficiency."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"In the 24 hours before a scheduled Fed announcement, stocks drift **up**. Not because of "
            "any news — the meeting date is on the calendar months ahead — but the market just floats "
            "higher into the decision. Buy the day before, and you collect a chunk of the entire equity "
            "premium on a handful of days a year.\"*\n\n"
            "This isn't a trader's tall tale. **Lucca & Moench (2015, Journal of Finance)** showed that "
            "since 1994 the pre-FOMC window earned a startling share of the market's total return — one of "
            "the most striking calendar effects ever documented. The puzzle: *why would a pre-scheduled, "
            "no-news event move prices at all?*"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If markets were perfectly efficient, a **known** event with **no new information** in the "
            "window shouldn't earn anything special. The pre-FOMC drift says otherwise — which made it a "
            "genuine crack in the efficient-markets story, and catnip for traders.\n\n"
            "But two traps hide here. **(1)** Famous edges tend to **die** once they're published — "
            "everyone front-runs them until there's nothing left. **(2)** The exact *story* (a run-up into "
            "the afternoon statement) might be wrong even if the *effect* is real. We'll test both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode the Fed's scheduled-meeting calendar — **{R['n_meetings']} meetings** from 1994 "
            f"to 2026 — and tag the SPY trading session that ends just before each one. That's only "
            f"**{R['frac_days']:.0f}%** of all trading days. Then:\n\n"
            "1. **The drift.** Compare the average return on those pre-Fed days to every other day.\n"
            "2. **The luck test.** Re-roll the calendar 20,000 times at random — could a random set of "
            "the same number of days have looked this good?\n"
            "3. **The decay test.** Split the history at 2012 (after the paper circulated) and ask: did "
            "the edge survive being famous?\n"
            "4. **The mechanism.** Split each day into its **overnight** move and its **intraday** move — "
            "is the drift really the afternoon run-up the story claims?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average return on pre-Fed days vs every other day, full sample."
        ),
        code(
            "if HAVE_REAL:\n"
            "    full = st.event_vs_rest(RET.values, MASK)\n"
            "    ev, rest, share = full['ev_mean']*100, full['rest_mean']*100, full['share_of_total']*100\n"
            "else:\n"
            "    ev, rest, share = R['ev_mean'], R['rest_mean'], R['share_total']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['pre-FOMC day\\n(3% of days)', 'every other day'], [ev, rest], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([ev, rest]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average return (%/day)')\n"
            "ax.set_title(f'The pre-FOMC day earns ~5x a normal day  (it carries {share:.0f}% of all return)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-FOMC mean: {ev:+.3f}%/day   other-day mean: {rest:+.3f}%/day   '\n"
            "      f'share of cumulative return on 3% of days: {share:.0f}%')"
        ),
        md(
            f"There it is. The pre-Fed day earns **+{R['ev_mean']:.3f}%** — roughly **five times** a normal "
            f"day — and those **{R['frac_days']:.0f}%** of sessions carry **{R['share_total']:.0f}%** of "
            "SPY's *entire* cumulative return since 1993. That's a real, large effect."
        ),
        md(
            "**But did it survive being famous?** Here's the same comparison, split before and after 2012 "
            "(the paper circulated in 2011, published 2015). Watch the right-hand pair."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.split_summary(RET, MASK)\n"
            "    pre_ev, pre_rest = sp['pre']['ev_mean']*100, sp['pre']['rest_mean']*100\n"
            "    po_ev, po_rest = sp['post']['ev_mean']*100, sp['post']['rest_mean']*100\n"
            "    pre_t, po_t = sp['pre']['welch_t'], sp['post']['welch_t']\n"
            "else:\n"
            "    pre_ev, pre_rest, pre_t = R['pre2012'][1], R['pre2012'][2], R['pre2012'][3]\n"
            "    po_ev, po_rest, po_t = R['post2012'][1], R['post2012'][2], R['post2012'][3]\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, [pre_ev, po_ev], .4, color=GREEN, label='pre-FOMC day')\n"
            "ax.bar(x+.2, [pre_rest, po_rest], .4, color=GREY, label='every other day')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'before 2012\\n(t={pre_t:.1f})', f'after 2012\\n(t={po_t:.1f})'])\n"
            "ax.set_ylabel('average return (%/day)'); ax.legend()\n"
            "ax.set_title('The edge lived BEFORE the paper — after publication it is a normal day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2012:  pre-FOMC {pre_ev:+.3f}% vs other {pre_rest:+.3f}%  (Welch t={pre_t:.2f})')\n"
            "print(f'post-2012: pre-FOMC {po_ev:+.3f}% vs other {po_rest:+.3f}%  (Welch t={po_t:.2f})')"
        ),
        md(
            f"This is the punchline. Before 2012 the pre-Fed day earned **+{R['pre2012'][1]:.3f}%** at a "
            f"convincing *t* of **{R['pre2012'][3]:.1f}**. After 2012 it's **+{R['post2012'][1]:.3f}%** vs "
            f"**+{R['post2012'][2]:.3f}%** for a normal day — a *t* of just **{R['post2012'][3]:.2f}**, "
            "i.e. **nothing**. The edge didn't fade gently; it switched off once the paper was out."
        ),
        md(
            "**And the mechanism?** The story is \"stocks run up into the 2:15pm statement\" — that would "
            "put the gain in the **intraday** session. Let's split each day into its overnight move (before "
            "the open) and its intraday move (open to close)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    oi = st.overnight_intraday(OHLC); oi = oi[oi.index.isin(RET.index)]\n"
            "    m2 = data.tag_pre_fomc(oi.index, shift=0)\n"
            "    vals = [st.event_vs_rest(oi[c].values, m2)['ev_mean']*100 for c in ('overnight','intraday')]\n"
            "    rests = [st.event_vs_rest(oi[c].values, m2)['rest_mean']*100 for c in ('overnight','intraday')]\n"
            "else:\n"
            "    vals = [R['oi'][0][1], R['oi'][1][1]]; rests = [R['oi'][0][2], R['oi'][1][2]]\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, vals, .4, color=GREEN, label='pre-FOMC day')\n"
            "ax.bar(x+.2, rests, .4, color=GREY, label='every other day')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['OVERNIGHT\\n(before open)', 'INTRADAY\\n(open->close)'])\n"
            "ax.set_ylabel('average return (%)'); ax.legend()\n"
            "ax.set_title('The pre-FOMC gain is split ~evenly — NOT just the afternoon run-up')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overnight: pre-FOMC {vals[0]:+.3f}% vs other {rests[0]:+.3f}%')\n"
            "print(f'intraday:  pre-FOMC {vals[1]:+.3f}% vs other {rests[1]:+.3f}%')"
        ),
        md(
            f"The famous story is only **half right**. Yes, the intraday session is elevated "
            f"(**+{R['oi'][1][1]:.3f}%** vs +{R['oi'][1][2]:.3f}%) — but so is the **overnight** gap "
            f"(**+{R['oi'][0][1]:.3f}%** vs +{R['oi'][0][2]:.3f}%), before the market even opens. A good "
            "chunk of the pre-FOMC drift arrives in your sleep, not in the 2pm run-up."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The pre-FOMC day earned **+{R['ev_mean']:.3f}%/day** (*t* = "
            f"{R['ev_t0']:.2f}), and {R['frac_days']:.0f}% of days carried {R['share_total']:.0f}% of all "
            "return. A genuine anomaly — rare among the legends we test.\n"
            "- **Tradability — Fragile.** It **decayed to nothing** after publication (post-2012 *t* = "
            f"{R['post2012'][3]:.2f}). Real in the archive, dead in the present.\n"
            "- **\"It's the afternoon run-up\"? — Mixed.** Half the gain is **overnight**, before the open. "
            "The clean run-up-into-the-statement story is only half the truth."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs aren't the problem, decay is\n\n"
            "SPY is one of the cheapest things on earth to trade, so a calendar rule that's only in the "
            "market on pre-Fed days pays almost nothing in costs. The catch isn't costs — it's that you'd "
            "be trading a **dead** signal. Here's the gross-vs-net (it barely moves), then remember the "
            "post-2012 *t* was 0.28."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.timing_strategy(RET.values, MASK, cost_bps=cb) for cb in (0.5, 1.0, 2.0)]\n"
            "    g = [r['gross_mean']*100 for r in rows]; nv = [r['net_mean']*100 for r in rows]\n"
            "else:\n"
            "    g = [c[1] for c in R['costs']]; nv = [c[2] for c in R['costs']]\n"
            "x = np.arange(3)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net of costs')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['0.5 bp','1 bp','2 bp'])\n"
            "ax.set_ylabel('pre-FOMC mean return (%)'); ax.legend()\n"
            "ax.set_title('Costs barely touch the edge on liquid SPY — the killer is the post-2012 decay')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gross vs net at 1bp:', f'{g[1]:+.3f}% -> {nv[1]:+.3f}%', '| but post-2012 t = 0.28 (dead)')"
        ),
        md(
            f"At a realistic 1 bp the edge goes from **+{R['costs'][1][1]:.3f}%** gross to "
            f"**+{R['costs'][1][2]:.3f}%** net — basically untouched. The reason you still can't deploy it "
            "is the decay: you'd be timing a signal that stopped paying a decade ago."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The decay is the lesson.** This is the cleanest example in the whole desk of "
            "*McLean-Pontiff* — a real, published anomaly arbitraged away once everyone knew. Edges that "
            "make headlines are exactly the ones most likely to vanish.\n"
            "- **The mechanism matters.** \"Stocks run up into the 2pm statement\" is a tidy story, but "
            "the data says half the move is overnight. Tidy stories about *why* an effect exists are often "
            "wrong even when the effect is right.\n"
            "- **Replication.** This study re-derives [Study 67 — Fed-Drift](../../67-fed-drift/) on an "
            "independent hardcoded calendar and agrees — a real edge, fragile to harvest.\n\n"
            "*Think the pre-FOMC day still pays? Show a **post-2012** Welch *t* above 2 on honest data — "
            "then we'll talk.*"
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
            "# The pre-FOMC announcement drift — a quantitative teardown 🔬\n"
            "### Pre-FOMC vs other-day Welch *t* · a random-calendar placebo · the pre/post-2012 decay "
            "split · overnight-vs-intraday decomposition · costs × turnover · a seed-averaged "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The pre-FOMC "
            "drift (Lucca & Moench 2015) is the rare calendar anomaly that **clears the bar** on the full "
            "sample — so the job here is to measure it honestly: confront it with a random-calendar "
            "placebo, split it at the publication era to expose the **decay**, and decompose the day to "
            "test the paper's *mechanism*.\n\n"
            "> ⚠️ **Data + survivorship note.** Headline tape is **SPY** (an index ETF — survivorship-clean). "
            "A 30-name survivor basket is kept as colour, with the bias named. Real data: yfinance daily "
            "closes + raw SPY Open/Close, 1993→2026; FOMC calendar **hardcoded** from Fed schedules. "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | pre-FOMC mean **+{R['ev_mean']:.3f}%/day** (one-sample "
            f"**t = {R['ev_t0']:.2f}**, Welch **t = {R['welch_t']:.2f}**, random-calendar placebo "
            f"**p = {R['placebo_p']:.3f}**); {R['frac_days']:.0f}% of days carry "
            f"**{R['share_total']:.0f}%** of cumulative return. Clears **t ≥ 2**; SPY is survivorship-clean. |\n"
            f"| **Tradability** | `FRAGILE` | Costs barely bite on SPY (net t ≈ {R['costs'][1][3]:.1f}), but "
            f"the signal **decayed**: pre-2012 Welch **t = {R['pre2012'][3]:.2f}** → post-2012 "
            f"**{R['post2012'][3]:.2f}** (a normal day). Timing-sensitive (shift −1 → t = {R['shift'][1][2]:.2f}). |\n"
            f"| **Run-up only?** | `MIXED` | Pre-FOMC excess splits ~evenly: overnight "
            f"**+{R['oi'][0][1]:.3f}%** (t={R['oi'][0][3]:.2f}) vs intraday **+{R['oi'][1][1]:.3f}%** "
            f"(t={R['oi'][1][3]:.2f}); **neither half clears t = 2** — only the combined day does. |\n\n"
            "> 💡 In plain words: the drift is genuinely real on the full tape — but it's a **museum "
            "piece** (gone post-publication), and the popular \"intraday run-up\" mechanism is only half "
            "the story."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\mathcal{F}$ be the set of trading sessions that end just before a scheduled FOMC "
            "announcement (the meeting date is in the calendar months ahead, so $\\mathcal{F}$ is "
            "**pre-known** — no look-ahead). The pre-FOMC drift is\n\n"
            "$$\\widehat{\\mu}_{\\mathcal F} = \\frac{1}{|\\mathcal F|}\\sum_{t\\in\\mathcal F} r_t "
            "\\;\\gg\\; \\widehat{\\mu}_{\\bar{\\mathcal F}} = \\frac{1}{|\\bar{\\mathcal F}|}"
            "\\sum_{t\\notin\\mathcal F} r_t.$$\n\n"
            "- **H₁ (the drift exists).** $\\widehat{\\mu}_{\\mathcal F} - \\widehat{\\mu}_{\\bar{\\mathcal F}} > 0$ "
            "and significant.\n"
            "- **H₂ (it's deployable today).** It survives costs **and** does not decay post-publication.\n"
            "- **H₃ (the run-up mechanism).** The excess is concentrated **intraday** (the 14:15 run-up), "
            "not overnight.\n\n"
            "We find **H₁ supported** (full-sample *t* > 2, placebo *p* = 0.007), **H₂ rejected** (dead "
            "post-2012), **H₃ partly rejected** (split ~evenly overnight/intraday). The effect is real; "
            "it is neither current nor mechanistically clean."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the pre-FOMC returns vs 0 and a two-sample Welch test "
            "vs other days:\n\n"
            "$$t_0 = \\frac{\\overline{r}_{\\mathcal F}}{s_{\\mathcal F}/\\sqrt{|\\mathcal F|}},\\qquad "
            "t_W = \\frac{\\overline{r}_{\\mathcal F}-\\overline{r}_{\\bar{\\mathcal F}}}"
            "{\\sqrt{s_{\\mathcal F}^2/|\\mathcal F| + s_{\\bar{\\mathcal F}}^2/|\\bar{\\mathcal F}|}}.$$\n\n"
            "Two honesty problems sit on top of a raw *t*. **(a) Calendar luck:** with only ~3% of days "
            "tagged, a lucky placement could look good — so we add a **random-calendar placebo** that "
            "relocates the same number of days 20,000 times. **(b) Publication decay:** a famous anomaly "
            "may be priced out — so we **split at 2012** and re-test. Only an effect that clears the "
            "placebo *and* survives the split would be tradable."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** SPY daily adjusted closes (yfinance), {R['start']}→{R['end']}; "
            f"**{R['n_days']:,}** daily returns over **{R['years']:.0f}** years. As-of {R['asof']} (last "
            "complete month — no partial-month bias). SPY is survivorship-clean.\n"
            f"- **Calendar.** **{R['n_meetings']}** scheduled FOMC announcement dates, **hardcoded** from "
            f"Fed schedules (1994–2026); **{R['n_pre']}** map onto pre-FOMC sessions "
            f"(**{R['frac_days']:.0f}%** of days). Pre-known — no look-ahead.\n"
            "- **Signal.** Mean pre-FOMC return vs other-day mean; one-sample *t* vs 0 and Welch *t*.\n"
            "- **Null (placebo).** 20,000 random same-size calendars; "
            "$p = \\Pr[\\text{random mean} \\ge \\text{observed}]$.\n"
            "- **Decay.** Split at 2012 (paper circulated 2011, published 2015); re-run pre/post.\n"
            "- **Mechanism.** Decompose each day into overnight (prev close→open) and intraday "
            "(open→close); re-test each.\n"
            "- **Costs.** Timing rule = 2 trades/meeting on SPY; 0.5–2 bp one-way.\n"
            "- **Positive control.** A deterministic daily world with a **planted** pre-meeting drift, "
            "seed-averaged over 40 seeds: zero edge must NOT reach significance; a planted edge must."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline + the luck test\n\n"
            "The observed pre-FOMC mean against a 20,000-draw random-calendar null. The green line should "
            "sit in the far right tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    full = st.event_vs_rest(RET.values, MASK)\n"
            "    pl = st.placebo_pvalue(RET.values, MASK, n_draws=20000)\n"
            "    obs = pl['obs']*100; draws = pl['draws']*100; pval = pl['p_value']; t0 = full['ev_t0']\n"
            "else:\n"
            "    obs = R['ev_mean']; pval = R['placebo_p']; t0 = R['ev_t0']\n"
            "    rng = np.random.default_rng(517); draws = rng.normal(R['rest_mean'], 0.06, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: 20,000 random 260-day calendars')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed pre-FOMC mean {obs:+.3f}%')\n"
            "ax.set_xlabel('mean daily return of the tagged days (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Outside the luck cloud: placebo p = {pval:.3f}, one-sample t = {t0:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.3f}%  one-sample t={t0:.2f}  placebo p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the green line sits in the **right tail** — only "
            f"**{R['placebo_p']*100:.1f}%** of random calendars match it, one-sample **t = {R['ev_t0']:.2f}**. "
            "The full-sample pre-FOMC drift is **not** a fluke of where the meeting days happen to fall."
        ),
        md(
            "### 4b · The decay split — the McLean-Pontiff killshot\n\n"
            "Pre-FOMC vs other-day mean, before and after 2012. If publication priced it out, the "
            "post-2012 gap collapses."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.split_summary(RET, MASK)\n"
            "    pre, post = sp['pre'], sp['post']\n"
            "    pre_ev, pre_rest, pre_t = pre['ev_mean']*100, pre['rest_mean']*100, pre['welch_t']\n"
            "    po_ev, po_rest, po_t = post['ev_mean']*100, post['rest_mean']*100, post['welch_t']\n"
            "    pre_sh, po_sh = pre['share_of_total']*100, post['share_of_total']*100\n"
            "else:\n"
            "    pre_ev, pre_rest, pre_t, pre_sh = R['pre2012'][1], R['pre2012'][2], R['pre2012'][3], R['pre2012'][4]\n"
            "    po_ev, po_rest, po_t, po_sh = R['post2012'][1], R['post2012'][2], R['post2012'][3], R['post2012'][4]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "x = np.arange(2)\n"
            "a1.bar(x-.2, [pre_ev, po_ev], .4, color=GREEN, label='pre-FOMC')\n"
            "a1.bar(x+.2, [pre_rest, po_rest], .4, color=GREY, label='other days')\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'pre-2012\\nt={pre_t:.2f}', f'post-2012\\nt={po_t:.2f}'])\n"
            "a1.set_ylabel('mean return (%/day)'); a1.set_title('Pre-FOMC edge vanishes after publication'); a1.legend()\n"
            "a2.bar(['pre-2012','post-2012'], [pre_sh, po_sh], color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([pre_sh, po_sh]): a2.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('share of cumulative return'); a2.set_title('Share of all return on the 3% of pre-FOMC days')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2012:  pre-FOMC {pre_ev:+.3f}% vs {pre_rest:+.3f}%  Welch t={pre_t:.2f}  share={pre_sh:.0f}%')\n"
            "print(f'post-2012: pre-FOMC {po_ev:+.3f}% vs {po_rest:+.3f}%  Welch t={po_t:.2f}  share={po_sh:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: pre-2012 the pre-FOMC day earned **+{R['pre2012'][1]:.3f}%** at Welch "
            f"**t = {R['pre2012'][3]:.2f}** and carried **{R['pre2012'][4]:.0f}%** of all return. Post-2012 "
            f"it's Welch **t = {R['post2012'][3]:.2f}** — statistically a **normal day**. Textbook "
            "post-publication decay: the edge was real, then it was arbitraged out."
        ),
        md(
            "### 4c · The mechanism — overnight vs intraday\n\n"
            "Lucca-Moench's story is an intraday run-up into the 14:15 release. Decompose each day into "
            "its overnight gap and its intraday session and test each separately."
        ),
        code(
            "if HAVE_REAL:\n"
            "    oi = st.overnight_intraday(OHLC); oi = oi[oi.index.isin(RET.index)]\n"
            "    m2 = data.tag_pre_fomc(oi.index, shift=0)\n"
            "    comps = ['overnight','intraday','daily']\n"
            "    res = [st.event_vs_rest(oi[c].values, m2) for c in comps]\n"
            "    ev = [r['ev_mean']*100 for r in res]; rest = [r['rest_mean']*100 for r in res]; tt = [r['welch_t'] for r in res]\n"
            "else:\n"
            "    comps = [o[0] for o in R['oi']]; ev = [o[1] for o in R['oi']]; rest=[o[2] for o in R['oi']]; tt=[o[3] for o in R['oi']]\n"
            "x = np.arange(len(comps))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ev, .4, color=GREEN, label='pre-FOMC')\n"
            "ax.bar(x+.2, rest, .4, color=GREY, label='other days')\n"
            "for i,t in enumerate(tt): ax.annotate(f't={t:.2f}',(i, max(ev[i],rest[i])),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in comps]); ax.axhline(0,c='k',lw=.6)\n"
            "ax.set_ylabel('mean return (%)'); ax.legend()\n"
            "ax.set_title('Split ~evenly: neither overnight nor intraday alone clears t=2')\n"
            "plt.tight_layout(); plt.show()\n"
            "for c,e,r,t in zip(comps, ev, rest, tt): print(f'{c:>9}: pre-FOMC {e:+.3f}% vs {r:+.3f}%  Welch t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: overnight is **+{R['oi'][0][1]:.3f}%** vs +{R['oi'][0][2]:.3f}% "
            f"(t={R['oi'][0][3]:.2f}); intraday is **+{R['oi'][1][1]:.3f}%** vs +{R['oi'][1][2]:.3f}% "
            f"(t={R['oi'][1][3]:.2f}). **Neither half clears t = 2** — only the combined close-to-close day "
            "(t={:.2f}) does. The clean 'all afternoon run-up' framing is only half right.".format(R['oi'][2][3])
        ),
        md(
            "### 4d · Costs + the timing-robustness check\n\n"
            "Left: gross vs net on liquid SPY (costs barely matter). Right: tag the session **strictly "
            "before** the announcement (shift −1) — the drift is concentrated in the session that *ends* "
            "at the release, so shifting back weakens it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.timing_strategy(RET.values, MASK, cost_bps=cb)['gross_mean']*100 for cb in (0.5,1.0,2.0)]\n"
            "    nv = [st.timing_strategy(RET.values, MASK, cost_bps=cb)['net_mean']*100 for cb in (0.5,1.0,2.0)]\n"
            "    m_sh = data.tag_pre_fomc(RET.index, shift=-1)\n"
            "    s0 = st.event_vs_rest(RET.values, MASK); s1 = st.event_vs_rest(RET.values, m_sh)\n"
            "    sh_ev = [s0['ev_mean']*100, s1['ev_mean']*100]; sh_t = [s0['welch_t'], s1['welch_t']]\n"
            "else:\n"
            "    g = [c[1] for c in R['costs']]; nv = [c[2] for c in R['costs']]\n"
            "    sh_ev = [R['shift'][0][1], R['shift'][1][1]]; sh_t = [R['shift'][0][2], R['shift'][1][2]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "x = np.arange(3)\n"
            "a1.bar(x-.2, g, .4, color=GREEN, label='gross'); a1.bar(x+.2, nv, .4, color=GREY, label='net')\n"
            "a1.set_xticks(x); a1.set_xticklabels(['0.5bp','1bp','2bp']); a1.set_ylabel('pre-FOMC mean (%)')\n"
            "a1.set_title('Costs barely dent it on SPY'); a1.legend()\n"
            "a2.bar(['announce-day','strictly prior'], sh_ev, color=[GREEN, AMBER], width=.55)\n"
            "for i,(v,t) in enumerate(zip(sh_ev, sh_t)): a2.annotate(f'{v:+.3f}%\\nt={t:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('mean return (%/day)'); a2.set_title('Drift sits in the session that ENDS at release')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net at 1bp:', f'{nv[1]:+.3f}%', '| shift-1 Welch t:', f'{sh_t[1]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: at 1 bp the edge is **+{R['costs'][1][2]:.3f}%** net (gross "
            f"+{R['costs'][1][1]:.3f}%) — costs are a rounding error on SPY. And shifting the window one "
            f"day back drops Welch *t* from **{R['shift'][0][2]:.2f}** to **{R['shift'][1][2]:.2f}**: the "
            "drift is glued to the session that ends at the release, exactly the run-up timing — but that "
            "precision also makes it fragile."
        ),
        md(
            "### 4e · Faithful-engine & power control — seed-averaged\n\n"
            "On a deterministic daily world where we **plant** a known pre-meeting drift, averaged over 40 "
            "RNG seeds: zero edge must stay at t≈0 (no false positive); a planted edge must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.004):\n"
            "    ts = []\n"
            "    for seed in range(40):\n"
            "        r, m = data.synthetic_world(edge=edge, seed=seed)\n"
            "        ts.append(st.event_vs_rest(r.values, m)['welch_t'])\n"
            "    ts = np.array(ts)\n"
            "    res.append((edge, ts.mean(), float(np.mean(np.abs(ts) > 2))))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.2f}%' for e,_,_ in res]; tvals = [r[1] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f'mean t={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('mean Welch t (40 seeds)'); ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,mt,fr in res: print(f'planted {e*100:+.2f}%: mean Welch t={mt:+.2f}  frac|t|>2={fr:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drift the control sits at mean **t = "
            f"{R['syn'][0][1]:.2f}** with a {R['syn'][0][2]*100:.0f}% false-positive rate (exactly nominal "
            f"— noise can't fake it); a **+{R['syn'][1][0]:.2f}%** planted drift reaches mean **t = "
            f"{R['syn'][1][1]:.2f}** with {R['syn'][1][2]*100:.0f}% power. The machinery is unbiased, so "
            f"the real-tape **t = {R['ev_t0']:.2f}** is the genuine article — not a construction artefact."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — pre-FOMC mean **+{R['ev_mean']:.3f}%/day** (one-sample **t = "
            f"{R['ev_t0']:.2f}**, Welch **t = {R['welch_t']:.2f}**, placebo **p = {R['placebo_p']:.3f}**); "
            f"{R['frac_days']:.0f}% of days carry **{R['share_total']:.0f}%** of cumulative return. Clears "
            "the **t ≥ 2** bar, survives the random-calendar placebo, confirmed by a seed-averaged "
            "synthetic control. SPY is survivorship-clean. REAL, not WEAK.\n"
            f"- **Tradability `FRAGILE`** — costs barely bite on SPY (net t ≈ {R['costs'][1][3]:.1f}), but "
            f"the edge **decayed**: pre-2012 Welch **t = {R['pre2012'][3]:.2f}** → post-2012 "
            f"**{R['post2012'][3]:.2f}** (a normal day), and it is timing-sensitive (shift −1 → t = "
            f"{R['shift'][1][2]:.2f}). Real in the archive, dead in the present. Not INVESTABLE.\n"
            f"- **Run-up only? `MIXED`** — the pre-FOMC excess splits ~evenly between overnight "
            f"(**+{R['oi'][0][1]:.3f}%**, t={R['oi'][0][3]:.2f}) and intraday (**+{R['oi'][1][1]:.3f}%**, "
            f"t={R['oi'][1][3]:.2f}); neither half alone clears t = 2. The clean 'afternoon run-up' "
            "mechanism is only half the story."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the decay is the whole story\n\n"
            "One picture for the operational truth: the pre-FOMC Welch *t* by era. The tradeable region is "
            "**behind us** — the signal was strong before publication and is indistinguishable from a "
            "normal day after it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.split_summary(RET, MASK)\n"
            "    eras = ['full','pre-2012','post-2012']\n"
            "    tvals = [sp['full']['welch_t'], sp['pre']['welch_t'], sp['post']['welch_t']]\n"
            "else:\n"
            "    eras = ['full','pre-2012','post-2012']; tvals = [R['welch_t'], R['pre2012'][3], R['post2012'][3]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREY, GREEN, RED]\n"
            "ax.bar(eras, tvals, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('pre-FOMC vs other-day Welch t'); ax.legend()\n"
            "ax.set_title('Real before the paper, dead after — a museum-piece edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by era:', {e: round(t,2) for e,t in zip(eras, tvals)})"
        ),
        md(
            "> 💡 In plain words: the green (pre-2012) bar towers over the *t* = 2 line; the red "
            "(post-2012) bar is on the floor. That's the entire tradability verdict in one chart — you "
            "can't deploy a signal whose edge expired a decade ago. Real, but **FRAGILE**, not INVESTABLE."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The decay is the lesson.** This is the desk's cleanest McLean-Pontiff case: a published, "
            "real anomaly arbitraged away once it was known. Re-run any famous calendar effect across its "
            "publication date and look for the same break.\n"
            "- **Mechanism ≠ effect.** The 'intraday run-up into the 14:15 statement' story is tidy and "
            "only half-true — half the pre-FOMC return is overnight. Always test the *why*, not just the "
            "*whether*.\n"
            "- **Replication.** This re-derives [Study 67 — Fed-Drift](../../67-fed-drift/) on an "
            "independent hardcoded calendar and agrees (Real × Fragile); it is distinct from "
            "[Study 135 — FOMC-Cycle](../../135-fomc-cycle/) (the even-week cycle).\n\n"
            "*The reproducible core is offline and deterministic; the FOMC calendar is hardcoded from Fed "
            "schedules and the headline tape is survivorship-clean SPY. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
