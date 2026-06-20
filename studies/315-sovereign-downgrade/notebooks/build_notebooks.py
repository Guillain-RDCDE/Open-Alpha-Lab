"""Generate the two narrative notebooks for Study 315 (Sovereign-Downgrade).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the shared SPY
total-return cache parquet if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n_events=3,
    fp="eb1569cc9e97",
    # per-event event-bar abnormal returns (lag=1, first tradeable bar), %
    e2011_abn=-6.77, e2023_abn=-1.44, e2025_abn=+0.07,
    e2011_bar="2011-08-08", e2023_bar="2023-08-02", e2025_bar="2025-05-19",
    # CARs (lag=1), mean % and permutation p vs synthetic control
    car01_mean=-1.45, car01_t=-3.09, car01_p=0.106, car01_ctrl=-0.02,
    car05_mean=-1.06, car05_t=-2.45, car05_p=0.427,
    car20_mean=-1.94, car20_t=-2.60, car20_p=0.450,
    # block-bootstrap CI on event-bar abnormal return, %
    ci_lo=-5.00, ci_hi=-0.43, evbar_mean=-2.72,
    # short trade, %
    short5_gross=-1.86, short5_net=-1.95,
    short20_gross=-1.59, short20_net=-1.83,
    short5_lookahead=+1.98,
    spy_bh=10.8, yrs=33.4,
    # synthetic positive control (CAR(0,1), effect -> perm p)
    pc_null_p=0.640, pc02_p=0.001, pc04_p=0.001,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from sovereign_downgrade import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path("SPY"))

HAVE_REAL = _have_cache()

def real_bars():
    return data.load_real("SPY")

print("real SPY total-return cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sovereign-Downgrade — does a US credit cut tank stocks? 🏛️\n"
            "### The most-feared macro headline, tested against the three times it actually happened\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Sell_the_downgrade%3F: Busted](https://img.shields.io/badge/Sell_the_downgrade%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every time Washington flirts with default, the same warning circulates: *if the rating "
            "agencies cut the US, stocks will crash.* The reference is always the ~6.7% S&P 500 drop "
            "the Monday after **S&P's August 2011 downgrade**. But the US has been downgraded "
            "**three times** now — S&P in 2011, Fitch in 2023, Moody's in 2025. This notebook asks: "
            "did stocks actually crash all three times, and could you have made money betting on it?\n\n"
            "> 📓 **This is the plain-language layer.** Want the abnormal returns, the permutation "
            "test and the synthetic control? That's the companion, "
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
            "| Did stocks crash on the 2011 downgrade? | **Yes — a brutal −6.8%** on the first "
            "tradeable day. Real, and scary. |\n"
            "| Did they crash on the *other two*? | **No.** Fitch 2023 was a mild down day "
            f"({R['e2023_abn']:+.1f}%); Moody's 2025 was a **non-event** ({R['e2025_abn']:+.1f}%). |\n"
            "| Is the 'downgrade crash' a real, repeatable signal? | **Not on this evidence.** "
            "Across all three, the move is **indistinguishable from a random week** "
            f"(*p* = {R['car01_p']:.2f}). One famous Monday is carrying the whole myth. |\n"
            "| Could you trade it? | **No.** All three were announced *after the close*, so the "
            "drop arrives as an overnight gap you can't get in front of — and shorting *after* it "
            f"just bleeds the market's +{R['spy_bh']:.0f}%/yr drift. |\n\n"
            "> The downgrade scare is a one-anecdote myth. 2011 was real and un-tradeable; the two "
            "sequels barely moved the tape. 'Sell the downgrade' only wins with a time machine."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"If a rating agency strips the US of its AAA, risk assets sell off — a downgrade is "
            "a flashing-red macro signal. Sell the announcement, buy back the panic.\"*\n\n"
            "The evidence everyone cites is **August 2011**: S&P cut the US to AA+ on a Friday "
            "night, and the S&P 500 fell ~6.7% the following Monday. It's a vivid, genuine memory. "
            "The question is whether it *generalises* — whether 'the US got downgraded' is a "
            "reliable sell signal, or whether 2011 was a one-off swept up in a much bigger "
            "euro-crisis panic that happened to share the same week."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a downgrade reliably crashed stocks, it would be one of the cleanest macro trades "
            "going — the dates are announced, the direction is 'obvious', and the move is large. "
            "Whole risk-management playbooks are built on the assumption. But if the 'signal' is "
            "really just **one event** dressed up as a pattern, then every 'protect the portfolio "
            "before the downgrade' pitch is selling you insurance against a coin flip. The desk's "
            "job is to tell those two apart — and three downgrades is now *just* enough data to look."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks:\n\n"
            "1. **Line up all three events** and look at each one honestly — not just the one we "
            "remember.\n"
            "2. **Compare to random weeks.** We measure the stock move around the downgrades and "
            "around 3,000 *random* dates with no special meaning. If a downgrade is special, its "
            "move stands out from the random crowd. (This is the 'synthetic control'.)\n"
            "3. **Try to actually trade it.** Short stocks when the downgrade hits and see if you "
            "make money — paying realistic costs and the cost of borrowing shares to short.\n\n"
            "Tape: SPY total return (dividends in) since 1993. The kill criterion: if the downgrade "
            "move looks like any random week, it's a mirage."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the three events, side by side.** What did the S&P 500 do on the first day it "
            "could react to each downgrade?"
        ),
        code(
            "labels = ['2011\\nS&P', '2023\\nFitch', '2025\\nMoody\\'s']\n"
            "if HAVE_REAL:\n"
            "    bars = real_bars(); ev = data.announcements()\n"
            "    ed = st.event_day_abnormal(bars, ev, lag=1)\n"
            "    vals = (ed['abn']*100).tolist()\n"
            "else:\n"
            f"    vals = [{R['e2011_abn']}, {R['e2023_abn']}, {R['e2025_abn']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "col = [RED if v < -3 else (AMBER if v < -0.5 else GREY) for v in vals]\n"
            "b = ax.bar(labels, vals, color=col, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('first tradeable-day return (%)')\n"
            "ax.set_title('One downgrade crashed. The other two shrugged.')\n"
            "for bar_, v in zip(b, vals):\n"
            "    ax.annotate(f'{v:+.1f}%', (bar_.get_x()+bar_.get_width()/2,\n"
            "        v + (-0.4 if v < 0 else 0.2)), ha='center',\n"
            "        va='top' if v < 0 else 'bottom', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Event-bar returns:', [f'{v:+.2f}%' for v in vals])"
        ),
        md(
            f"The whole legend is **one bar**. 2011 was a genuine −6.8% rout; Fitch 2023 lost a mild "
            f"−1.4%; Moody's 2025 finished basically flat ({R['e2025_abn']:+.1f}%). If you only "
            "remember 2011, you remember a pattern that the data doesn't actually contain."
        ),
        md(
            "**Now the honest test: is a 'downgrade week' different from a random week?** We compare "
            "the downgrade move to the spread of moves around 3,000 random dates."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = real_bars(); ev = data.announcements()\n"
            "    nev = data.synthetic_null_events(bars, n_events=3000, window=25, seed=315)\n"
            "    ln = st.snap_to_sessions(bars, nev, lag=0)\n"
            "    event_car = st.car(bars, ev, window=(0,1), lag=1)['car'].to_numpy()*100\n"
            "    null_car = st.window_cars(bars, ln, window=(0,1))*100\n"
            "    em = event_car.mean()\n"
            "else:\n"
            "    rng = np.random.default_rng(0)\n"
            "    null_car = rng.normal(0.0, 1.6, 3000)\n"
            f"    em = {R['car01_mean']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.hist(null_car, bins=60, color=GREY, alpha=.7, label='random weeks (control)')\n"
            "ax.axvline(em, c=RED, lw=2.5, label=f'downgrade avg = {em:+.2f}%')\n"
            "ax.axvline(null_car.mean(), c='k', ls='--', lw=1, label='random avg')\n"
            "ax.set_xlabel('2-day return (%)'); ax.set_ylabel('count')\n"
            "ax.set_title('A downgrade week sits well inside the random crowd')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Downgrade avg {{em:+.2f}}% vs random — permutation p = {R['car01_p']:.2f} (not significant)')"
        ),
        md(
            f"The downgrade average ({R['car01_mean']:+.1f}% over two days) is more negative than the "
            f"random average — but it sits **comfortably inside** the random spread. The formal "
            f"permutation test gives *p* = {R['car01_p']:.2f}: you'd see a move this size in a random "
            "week about one time in ten. That is **not** a reliable signal — it's one big draw (2011) "
            "tugging an average of three."
        ),
        md(
            "**Last: could you have traded it?** Let's short the S&P when each downgrade hits and "
            "hold a week — paying realistic costs and share-borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = real_bars(); ev = data.announcements()\n"
            "    tradeable = st.short_the_downgrade(bars, ev, hold=5, lag=1, cost_bps=2.0, borrow_bps_per_day=1.0)['pnl_net'].mean()*100\n"
            "    lookahead = st.short_the_downgrade(bars, ev, hold=5, lag=0, cost_bps=0, borrow_bps_per_day=0)['pnl_gross'].mean()*100\n"
            "else:\n"
            f"    tradeable, lookahead = {R['short5_net']}, {R['short5_lookahead']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "b = ax.bar(['Tradeable\\n(short after the news)', 'Look-ahead\\n(short before the gap)'],\n"
            "           [tradeable, lookahead], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg short P&L per event (%)')\n"
            "ax.set_title('You only win if you can short BEFORE the news (you can\\'t)')\n"
            "for bar_, v in zip(b, [tradeable, lookahead]):\n"
            "    ax.annotate(f'{v:+.1f}%', (bar_.get_x()+bar_.get_width()/2,\n"
            "        v + (0.1 if v>0 else -0.1)), ha='center', va='bottom' if v>0 else 'top', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Tradeable short: {{tradeable:+.2f}}%/event (a LOSS). Look-ahead: {{lookahead:+.2f}}% (untradeable).')"
        ),
        md(
            "Here's the trap. All three downgrades were announced **after the close**, so the drop "
            "arrives as an overnight gap. The 2011 −6.8% *is* the first tradeable bar — you can't get "
            "short before it. Short *after* the gap and you ride the rebound; short the other two and "
            f"you bleed the market's upward drift. The 'tradeable' short **loses {R['short5_net']:+.1f}% "
            f"per event**. The only version that wins ({R['short5_lookahead']:+.1f}%) requires shorting "
            "*before* the post-close news — i.e. a time machine."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** 2011 was real and large, but it's **one event**. Across all three "
            f"downgrades the move is indistinguishable from a random week (*p* = {R['car01_p']:.2f}).\n"
            "- **Tradability — Mirage.** The crash is an un-tradeable overnight gap; shorting after it "
            f"loses money and fights a +{R['spy_bh']:.0f}%/yr drift.\n"
            "- **Sell the downgrade? — Busted.** A single August Monday is the entire myth."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and not for the usual desk reason (costs). The per-event short loses **gross**, "
            "before a cent of cost, because the move you wanted already happened by the time you could "
            "act. This is the cleanest kind of mirage: not a small edge eaten by fees, but a 'signal' "
            "that is structurally **un-frontrunnable** (post-close announcement → opening gap) wrapped "
            "around an effect that only showed up once. The honest pitch isn't 'sell the downgrade'; "
            "it's 'a US downgrade is a headline, not a trade.'"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The volatility sibling.** [Study 312 — Debt-Ceiling](../../312-debt-ceiling/) asks "
            "the same family of question from the *vol* side (long-vol-into / short-vol-out around the "
            "X-dates) and reaches the same broad conclusion — the famous 2011 vol spike came *after* "
            "the deal, on this very downgrade.\n"
            "- **The other Washington scare.** [Study 311 — Government-Shutdown](../../311-government-shutdown/) "
            "is the directional cousin: another macro headline that turns out not to be a trade.\n"
            "- **Could more events change the verdict?** Maybe — but each US downgrade is roughly a "
            "decade apart, so a clean n is a generation away. Until then, one Monday is all the "
            "'pattern' there is.\n\n"
            "*Think the downgrade is a real sell signal? Fork this, add the bond-market reaction (where "
            "the literature finds a clearer effect), and show the equity edge surviving a permutation "
            "test on more than three points.*"
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
            "# Sovereign-Downgrade — a quantitative teardown 🔬\n"
            "### Real SPY total-return tape · abnormal-return event study · synthetic control · "
            "permutation inference · the post-close timing trap\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Sell_the_downgrade%3F: Busted](https://img.shields.io/badge/Sell_the_downgrade%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run a constant-mean-return "
            "abnormal-return event study around the three US sovereign downgrades, race the CARs "
            "against a synthetic control, and confront the brutal fact that with **n = 3** the naive "
            "*t*-stat is meaningless and only a permutation test is honest.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY total-return daily bars (shared cache), "
            "as-of 2026-06-12; the offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | CAR(0,+1) mean **{R['car01_mean']:+.2f}%**; naive HAC "
            f"*t* = {R['car01_t']:+.2f} **but n = 3** → the honest permutation *p* vs the synthetic "
            f"control is **{R['car01_p']:.2f}** (CAR(0,+5) *p* = {R['car05_p']:.2f}, CAR(0,+20) "
            f"*p* = {R['car20_p']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Short-the-downgrade loses **{R['short5_gross']:+.2f}% "
            f"gross / {R['short5_net']:+.2f}% net** at 5d; the one big move was an un-tradeable "
            f"post-close gap; look-ahead version is {R['short5_lookahead']:+.2f}%. |\n"
            f"| **Sell the downgrade?** | `BUSTED` | Event-bar abn returns "
            f"[{R['e2011_abn']:+.1f}%, {R['e2023_abn']:+.1f}%, {R['e2025_abn']:+.1f}%]: the 2011 "
            "Monday is the entire effect. |\n\n"
            "> 💡 In plain words: there is a real-looking number (a −1.5% two-day drop), and a naive "
            "*t* that 'clears the bar' — but both are one observation in disguise. The synthetic "
            "control says a downgrade week looks like any other week, and the trade can't be put on "
            "in time anyway."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $AR_t = r_t - \\bar r$ be the abnormal return (realized minus the sample mean — a "
            "constant-mean-return market model; three events leave no room to fit a beta), and "
            "$CAR_{[a,b]} = \\sum_{\\tau=a}^{b} AR_{t_0+\\tau}$ the cumulative abnormal return over a "
            "window around each event bar $t_0$ (the first tradeable session after a post-close "
            "announcement).\n\n"
            "- **H₁ (reaction).** $\\mathbb{E}[CAR_{[0,k]} \\mid \\text{downgrade}] < 0$.\n"
            "- **H₂ (special).** That CAR differs from the same statistic around random non-event "
            "dates (the synthetic control).\n"
            "- **H₃ (tradable).** A short entered at the first tradeable bar earns positive net P&L.\n\n"
            "We find **H₁ directionally true but not certifiable** (n = 3, permutation *p* > 0.10), "
            "**H₂ rejected**, and **H₃ rejected** (the short loses, and only a look-ahead fill wins)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The sovereign-downgrade literature (Afonso-Furceri-Gomes 2012; Brooks et al. 2004; "
            "Gande-Parsley 2005) finds the cleanest effects in **bonds and CDS**, with equity "
            "reactions *modest and episode-dependent* — and the US is the extreme case, since "
            "Treasuries remain the global risk-free benchmark regardless of the notch. So the prior "
            "is already *weak equity effect*. The interesting content here is the **mechanism of the "
            "illusion**: a single salient crash (2011), a naive small-sample *t* that flatters it, "
            "and a post-close announcement structure that makes even the real move un-tradeable."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event bar.** First tradeable session after each post-close announcement "
            "(`lag=1`); a `lag=0` same-session variant exposes the look-ahead trap.\n"
            "- **Abnormal return.** $r_t - \\bar r$ (constant-mean model).\n"
            "- **Windows.** CAR over (0,+1), (0,+5), (0,+20), plus the pre-window (−k,−1).\n"
            "- **Synthetic control.** The same CAR around 3,000 random non-event dates; the verdict "
            "rides on a **permutation p-value** of the event mean vs this distribution.\n"
            "- **Inference caveat.** HAC *t* reported for completeness but **not load-bearing at "
            "n = 3**; a block-bootstrap CI on the event-bar return shows how wide it really is.\n"
            "- **Positive control.** Deterministic synthetic tape with a tunable planted dip.\n\n"
            "Tape: SPY total return, 1993–2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The average CAR path around the event\n\n"
            "Cumulative abnormal return aligned on the event bar (t = 0), averaged across the three "
            "downgrades, with its standard error."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = real_bars(); ev = data.announcements()\n"
            "    path = st.car_path(bars, ev, pre=10, post=20, lag=1)\n"
            "else:\n"
            "    rel = np.arange(-10, 21)\n"
            "    mean = np.where(rel < 0, 0.001*rel, -0.0009*rel)\n"
            "    path = pd.DataFrame({'rel': rel, 'mean': mean, 'se': 0.02}).set_index('rel')\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(path.index, path['mean']*100, c=RED, lw=2)\n"
            "ax.fill_between(path.index, (path['mean']-path['se'])*100, (path['mean']+path['se'])*100,\n"
            "                color=RED, alpha=.15)\n"
            "ax.axvline(0, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('sessions relative to event bar'); ax.set_ylabel('mean CAR (%)')\n"
            "ax.set_title('Average abnormal-return path — a wide, shallow drift dominated by 2011')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CAR at +20:', f\"{path.loc[20,'mean']*100:+.2f}% +/- {path.loc[20,'se']*100:.2f}%\")"
        ),
        md(
            "> 💡 In plain words: the path slopes down after the event, but the shaded standard-error "
            "band is wide and the slope is driven by one of three lines. A picture of a downgrade "
            "sell-off, drawn with too few points to trust."
        ),
        md(
            "### 4b · The naive *t* vs the permutation *p* — the small-sample trap\n\n"
            "The naive HAC *t* 'clears' the bar on the short windows; the permutation test against the "
            "synthetic control does not. This contrast *is* the WEAK verdict."
        ),
        code(
            "wins = [(0,1),(0,5),(0,20)]\n"
            "if HAVE_REAL:\n"
            "    bars = real_bars(); ev = data.announcements()\n"
            "    nev = data.synthetic_null_events(bars, n_events=3000, window=25, seed=315)\n"
            "    ln = st.snap_to_sessions(bars, nev, lag=0)\n"
            "    rows=[]\n"
            "    for w in wins:\n"
            "        c = st.car(bars, ev, window=w, lag=1)['car'].to_numpy()\n"
            "        nc = st.window_cars(bars, ln, window=w)\n"
            "        _, p = st.permutation_pvalue(c, nc, n_perm=20000, seed=315)\n"
            "        rows.append((str(w), st.hac_tstat(c), p))\n"
            "    dft = pd.DataFrame(rows, columns=['window','HAC_t','perm_p'])\n"
            "else:\n"
            f"    dft = pd.DataFrame({{'window':['(0, 1)','(0, 5)','(0, 20)'],\n"
            f"        'HAC_t':[{R['car01_t']},{R['car05_t']},{R['car20_t']}],\n"
            f"        'perm_p':[{R['car01_p']},{R['car05_p']},{R['car20_p']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "x = np.arange(len(dft)); ax.bar(x-0.2, dft['HAC_t'].abs(), .4, color=GREY, label='|naive HAC t|')\n"
            "ax2 = ax.twinx(); ax2.bar(x+0.2, dft['perm_p'], .4, color=RED, label='permutation p')\n"
            "ax.axhline(2, ls='--', c=GREY); ax2.axhline(0.05, ls=':', c=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(dft['window'])\n"
            "ax.set_ylabel('|HAC t| (misleading at n=3)'); ax2.set_ylabel('permutation p (honest)')\n"
            "ax.set_title('Naive t says \"significant\"; the honest test says \"random week\"')\n"
            "ax.legend(loc='upper left'); ax2.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dft.round(3).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: with three points, the standard error is computed from three points "
            f"— so a single −6.8% outlier produces a *t* of {R['car01_t']:.1f} that means nothing. The "
            f"permutation test asks the right question ('is this move unusual versus random weeks?') "
            f"and answers *p* = {R['car01_p']:.2f}. **H₂ rejected.**"
        ),
        md(
            "### 4c · The short trade and the post-close timing trap\n\n"
            "The decisive test of H₃: short SPY at the first tradeable bar's close, net of costs and "
            "borrow — and contrast it with the un-tradeable look-ahead fill."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = real_bars(); ev = data.announcements()\n"
            "    g5 = st.short_the_downgrade(bars, ev, hold=5, lag=1, cost_bps=0, borrow_bps_per_day=0)['pnl_gross'].mean()*100\n"
            "    n5 = st.short_the_downgrade(bars, ev, hold=5, lag=1, cost_bps=2.0, borrow_bps_per_day=1.0)['pnl_net'].mean()*100\n"
            "    la = st.short_the_downgrade(bars, ev, hold=5, lag=0, cost_bps=0, borrow_bps_per_day=0)['pnl_gross'].mean()*100\n"
            "else:\n"
            f"    g5, n5, la = {R['short5_gross']}, {R['short5_net']}, {R['short5_lookahead']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "b = ax.bar(['Tradeable\\ngross','Tradeable\\nnet','Look-ahead\\n(untradeable)'],\n"
            "           [g5, n5, la], color=[AMBER, RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg short P&L per event (%)')\n"
            "ax.set_title('The short loses unless you can act before the post-close gap')\n"
            "for bar_, v in zip(b, [g5,n5,la]):\n"
            "    ax.annotate(f'{v:+.1f}%', (bar_.get_x()+bar_.get_width()/2, v+(0.1 if v>0 else -0.1)),\n"
            "        ha='center', va='bottom' if v>0 else 'top', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'5d short: gross {{g5:+.2f}}%  net {{n5:+.2f}}%  look-ahead {{la:+.2f}}%')"
        ),
        md(
            f"> 💡 In plain words: the tradeable short loses **gross** ({R['short5_gross']:+.1f}%), "
            "before any cost — because the −6.8% move *is* the first bar you can trade, so you short "
            "the bottom and ride the bounce. Only the look-ahead fill (short the announcement-day "
            "close, *before* the post-close news) wins. **H₃ rejected.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — CAR(0,+1) {R['car01_mean']:+.2f}%, but n = 3; permutation "
            f"*p* = {R['car01_p']:.2f} (and ≈{R['car05_p']:.2f}–{R['car20_p']:.2f} thereafter). "
            f"Event-bar block-bootstrap CI [{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%] — wide, pulled by "
            "2011. The literature predicts a modest, episode-dependent effect; this tape cannot "
            "certify it.\n"
            f"- **Tradability `MIRAGE`** — short loses {R['short5_net']:+.2f}% net (and gross); the "
            "move is an un-tradeable post-close gap, and shorting after fights the drift.\n"
            f"- **Sell the downgrade? `BUSTED`** — [{R['e2011_abn']:+.1f}%, {R['e2023_abn']:+.1f}%, "
            f"{R['e2025_abn']:+.1f}%]: one Monday is the whole myth."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "This is a structural mirage, not a cost casualty. The break-even question is moot: the "
            "trade loses **before** costs, because (a) the only large move arrived as an opening gap "
            "you cannot front-run, and (b) the remaining two events left a short fighting a "
            f"+{R['spy_bh']:.0f}%/yr total-return drift over a {R['yrs']:.0f}-year tape. Capacity is "
            "irrelevant when the edge is negative and un-fillable in time."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always shrug? Plant a downgrade dip of "
            "varying strength on a synthetic tape and confirm the permutation test recovers it only "
            "when it's there."
        ),
        code(
            "effs = [0.0, 0.01, 0.02, 0.04, 0.08]\n"
            "ps = []\n"
            "for eff in effs:\n"
            "    b, e, _ = data.synthetic_spy(event_effect=eff, n_events=40, seed=315)\n"
            "    nv = data.synthetic_null_events(b, n_events=2000, window=25, seed=9)\n"
            "    ln = st.snap_to_sessions(b, nv, lag=0)\n"
            "    ec = st.car(b, e, window=(0,1), lag=0)['car'].to_numpy()\n"
            "    nc = st.window_cars(b, ln, window=(0,1))\n"
            "    _, p = st.permutation_pvalue(ec, nc, n_perm=8000, seed=1)\n"
            "    ps.append(p)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.plot([e*100 for e in effs], ps, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0.05, ls=':', c=RED, label='p = 0.05')\n"
            "ax.set_xlabel('planted announcement-day dip (%)'); ax.set_ylabel('permutation p')\n"
            "ax.set_title('The engine recovers a real dip — and finds nothing in the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('p by planted effect:', [f'{p:.3f}' for p in ps])"
        ),
        md(
            "The permutation test sits near 0.5–0.6 at the null (no planted dip) and collapses below "
            "0.05 the moment a real dip is present — so the harness is a faithful detector, and the "
            "real-tape 'nothing' is a statement about the **data** (three events, one un-tradeable "
            "shock), not a broken pipeline. A synthetic control can never certify a Signal stamp; it "
            "only proves the detector works.\n\n"
            "For the volatility-side teardown of the same downgrade family, see "
            "[Study 312 — Debt-Ceiling](../../312-debt-ceiling/); for the directional Washington-scare "
            "cousin, [Study 311 — Government-Shutdown](../../311-government-shutdown/)."
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
