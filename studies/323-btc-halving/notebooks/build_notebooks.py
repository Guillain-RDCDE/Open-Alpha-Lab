"""Generate the two narrative notebooks for Study 323 (BTC-Halving).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
BTC-USD daily parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    asof="2026-06-14", fingerprint="3c26b7c8e544", n_days=4289,
    # cycle extrema: offsets in days after each halving
    hi_offsets=[1296, 525, 1402, 534], lo_offsets=[777, 24, 0, 139],
    # phase buckets (daily bps, HAC t)
    pb0_mean=31.9, pb0_t=3.45, pb1_mean=0.3, pb2_mean=1.9, pb3_mean=13.7, pb3_t=1.23,
    # per-cycle post-halving quarter means (the three-cluster reveal)
    q0_2016=36.5, q0_2020=52.1, q0_2024=7.7, n_cycles=3,
    # timing rule vs buy-and-hold
    cagr_strat=35.1, cagr_bh=52.3, sharpe_strat=0.60, sharpe_bh=0.63,
    time_in_market=52.6, n_flips=8, excess_bps=-3.3, excess_t=-0.92,
    boot_lo=-10.6, boot_hi=4.2,
    # synthetic control
    syn_null_t=1.09, syn_cycle_t=4.65, syn_null_mean=13.3, syn_cycle_mean=66.6,
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

from btc_halving import data, strategy as st

def _load_real():
    try:
        return data.load_real()
    except Exception as e:
        print("real cache unavailable -> using frozen numbers / synthetic:", e)
        return None

BARS = _load_real()
HAVE_REAL = BARS is not None
print("real BTC-USD cache present:", HAVE_REAL)
if HAVE_REAL:
    print("tape:", BARS.index[0].date(), "->", BARS.index[-1].date(),
          "fingerprint", data.fingerprint(BARS))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# BTC-Halving — does the four-year cycle really print the top and bottom?\n"
            "### Crypto's grandest chart, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Prints_the_top_%26_bottom%3F: Busted](https://img.shields.io/badge/Prints_the_top_%26_bottom%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every four years Bitcoin's mining reward halves, and every four years the same "
            "chart goes viral: a clean cycle that **bottoms just before the halving and tops "
            "about a year after**, then crashes into the next bottom. Because the halving "
            "dates are *known years in advance*, the story implies a free lunch — the calendar "
            "alone would have timed every top and bottom. This notebook asks the only two "
            "questions that matter: *is the cycle real, and could the calendar actually have "
            "timed it?*\n\n"
            "> **This is the plain-language layer.** Want the *t*-stats, the three-cluster "
            "problem and the timing race? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there a post-halving glow? | **Faintly.** The year after each halving has "
            f"carried the strongest returns (**+{R['pb0_mean']:.0f} bps/day** on average) — but "
            f"on only **{R['n_cycles']} cycles**, and the latest one (2024) was nearly flat. |\n"
            "| Does it print the top and bottom? | **No.** The tops landed "
            f"**{R['hi_offsets'][1]}**, **{R['hi_offsets'][2]}**, **{R['hi_offsets'][3]}** days "
            "after their halvings — all over the map. The 'clean cycle' is hindsight. |\n"
            "| Could the calendar time it? | **No — it loses.** Buying the 'bull half' of each "
            f"cycle earned **+{R['cagr_strat']:.0f}%/yr** while just holding BTC earned "
            f"**+{R['cagr_bh']:.0f}%/yr**. Timing the cycle *underperformed doing nothing.* |\n"
            "| So what is it? | A real-feeling pattern fit to a tiny number of cycles, fading "
            "as the asset matures — and a timing recipe that costs you return. |\n\n"
            "> The halving cycle isn't a scam — the post-halving glow is faintly there. But it "
            "rests on three data points, it doesn't mark the turns, and timing it lost to "
            "buy-and-hold. Grand chart, thin evidence."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Bitcoin runs on a four-year clock set by the halving. Accumulate in the "
            "bear, a year or so before the halving; the halving cuts new supply in half; "
            "demand meets a supply shock and price tops roughly 12-18 months later; then it "
            "crashes ~80% into the next bottom — and repeat. The dates are fixed, so the cycle "
            "is a roadmap.\"*\n\n"
            "— the popular synthesis behind PlanB's *stock-to-flow* essays and a thousand "
            "'cycle' charts on crypto Twitter.\n\n"
            "The four halvings to date: **2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20.** "
            "The idea has real intuition behind it: a *scheduled, transparent supply cut* "
            "sounds like it should matter."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the cycle were a real roadmap it would be the single most valuable calendar in "
            "finance — a buy-and-sell schedule fixed years ahead, no forecasting required. "
            "That's exactly why it deserves a hard look. There's also a clean lesson here: a "
            "**scheduled** event can't be a tradable surprise (the market has years to price "
            "it in), and a cycle fit to a handful of four-year periods is the textbook setup "
            "for fooling yourself. If we can show how the grandest chart in crypto is built on "
            "three data points, that lesson transfers everywhere."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks, announced before we run them:\n\n"
            "1. **Where did the tops and bottoms *actually* fall?** For each halving, find the "
            "real high and low of the next four years and measure how many days after the "
            "halving they landed. If the cycle is real, they cluster. If not, they scatter.\n"
            "2. **Is the post-halving glow more than three lucky cycles?** Slice the cycle "
            "into quarters and look at returns — but count *cycles*, not days. Three cycles is "
            "three data points, however many days they contain.\n"
            "3. **Could the calendar have timed it?** Buy BTC only during the 'bull half' of "
            "each cycle and race it against simply holding. If timing helps, it beats "
            "buy-and-hold.\n\n"
            "Tape: Yahoo `BTC-USD` daily — which, crucially, **only starts in 2014**, so it "
            "covers the 2016 (partial), 2020 and 2024 halvings, not 2012."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: did the halving mark the tops and bottoms?** Here's where each cycle's "
            "high and low actually fell, in days after its halving:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ext = st.cycle_extrema(BARS)\n"
            "    hi = ext['high_offset_d'].tolist(); lo = ext['low_offset_d'].tolist()\n"
            "    labs = [str(h.date()) for h in ext['halving']]\n"
            "else:\n"
            f"    hi = {R['hi_offsets']}; lo = {R['lo_offsets']}\n"
            "    labs = ['2012-11','2016-07','2020-05','2024-04']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "x = np.arange(len(labs))\n"
            "ax.scatter(hi, x, s=120, color=GREEN, label='cycle HIGH', zorder=3)\n"
            "ax.scatter(lo, x, s=120, color=RED, label='cycle LOW', zorder=3)\n"
            "ax.axvspan(365, 550, color=AMBER, alpha=.15, label='lore: top ~12-18mo after')\n"
            "ax.set_yticks(x); ax.set_yticklabels(labs)\n"
            "ax.set_xlabel('days after the halving'); ax.set_ylabel('halving')\n"
            "ax.set_title('Where the tops and bottoms ACTUALLY fell — no cluster')\n"
            "ax.legend(loc='lower right', fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('high offsets (days):', hi)\nprint('low  offsets (days):', lo)"
        ),
        md(
            f"The tops landed at **{R['hi_offsets'][0]}, {R['hi_offsets'][1]}, "
            f"{R['hi_offsets'][2]}, {R['hi_offsets'][3]}** days after their halvings; the "
            f"bottoms at **{R['lo_offsets'][0]}, {R['lo_offsets'][1]}, {R['lo_offsets'][2]}, "
            f"{R['lo_offsets'][3]}**. They're scattered across the whole cycle — the neat "
            "'top a year after, bottom just before' is a line drawn through points we already "
            "knew. **The halving does not print the turns.**"
        ),
        md(
            "**Now the glow itself.** The year right after a halving *has* carried the best "
            "returns — but look how few cycles that average rests on:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lr = np.log(BARS['close']).diff()\n"
            "    rows = []\n"
            "    for h in data.HALVINGS:\n"
            "        if not (BARS.index.min() <= h <= BARS.index.max()):\n"
            "            continue\n"
            "        seg = lr[(lr.index >= h) & (lr.index < h + pd.Timedelta(days=364))].dropna()\n"
            "        rows.append((str(h.date()), seg.mean()*1e4))\n"
            "    labs = [r[0] for r in rows]; vals = [r[1] for r in rows]\n"
            "else:\n"
            f"    labs = ['2016-07','2020-05','2024-04']\n"
            f"    vals = [{R['q0_2016']}, {R['q0_2020']}, {R['q0_2024']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "col = [GREEN if v > 20 else AMBER for v in vals]\n"
            "ax.bar(labs, vals, color=col, width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('avg daily return, year after halving (bps)')\n"
            "ax.set_title('The post-halving glow rests on THREE cycles — and is fading')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('post-halving year, daily bps per cycle:', [round(v,1) for v in vals])"
        ),
        md(
            f"Three bars. The post-halving year paid **+{R['q0_2016']:.0f}**, then "
            f"**+{R['q0_2020']:.0f}**, then just **+{R['q0_2024']:.0f}** bps/day — the glow is "
            "**decaying** as Bitcoin matures and the trade gets crowded. Whatever statistics "
            "you compute, the honest sample is *three cycles*, and one of them is nearly flat."
        ),
        md(
            "**Last: could the calendar have timed it?** Buy BTC only in the 'bull half' of "
            "each cycle (the first two years after a halving), sit in cash otherwise, and "
            "compare to just holding:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.timing_rule(BARS, 0.0, 0.5, cost_bps=10.0)\n"
            "    s = st.summarize_timing(book)\n"
            "    cs, cb = s['cagr_strat']*100, s['cagr_bh']*100\n"
            "    eqs, eqb = book['eq_strat'], book['eq_bh']\n"
            "else:\n"
            f"    cs, cb = {R['cagr_strat']}, {R['cagr_bh']}; eqs = eqb = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "if eqs is not None:\n"
            "    a1.plot(eqb.index, eqb, c=GREEN, lw=1.6, label='buy & hold BTC')\n"
            "    a1.plot(eqs.index, eqs, c=AMBER, lw=1.6, label='halving-timing rule')\n"
            "    a1.set_yscale('log'); a1.legend(fontsize=8); a1.set_title('Timing trails holding')\n"
            "    a1.set_ylabel('growth of $1 (log)')\n"
            "else:\n"
            "    a1.text(.5,.5,'(run with cache for the equity curve)', ha='center')\n"
            "a2.bar(['Halving timing','Buy & hold BTC'], [cs, cb], color=[AMBER, GREEN], width=.5)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('annualised %/yr')\n"
            "a2.set_title('...so timing the cycle LOSES to holding')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timing {cs:.1f}%/yr   vs   buy-and-hold {cb:.1f}%/yr')"
        ),
        md(
            f"**The calendar lost.** Timing the cycle earned **+{R['cagr_strat']:.0f}%/yr** "
            f"while just holding BTC earned **+{R['cagr_bh']:.0f}%/yr**. By sitting in cash "
            "through the 'bear half' you missed enough of the asset's secular rise to "
            "underperform — even though you 'knew' the cycle in advance. The cycle edge was "
            "Bitcoin's beta, minus the days you sat out."
        ),

        # ---- BEAT 5 — THE VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The post-halving year carries the best returns "
            f"(+{R['pb0_mean']:.0f} bps/day), but on only {R['n_cycles']} cycles "
            f"(+{R['q0_2016']:.0f}/+{R['q0_2020']:.0f}/+{R['q0_2024']:.0f} bps/day — fading). "
            "Suggestive, not provable on this tape.\n"
            "- **Tradability — Mirage.** Timing the cycle earned "
            f"+{R['cagr_strat']:.0f}%/yr vs buy-and-hold's +{R['cagr_bh']:.0f}%/yr. It loses to "
            "doing nothing.\n"
            "- **Prints the top & bottom? — Busted.** The real tops and bottoms scatter across "
            "the whole cycle; the halving marks neither."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The cost sweep is almost beside the point — the rule only flips a handful of "
            "times, so costs barely register. The problem isn't costs; it's that the rule "
            "**sits in cash through part of the greatest bull asset of the era** and so "
            "underperforms it. A timing overlay that loses to buy-and-hold isn't fragile — "
            "it's a mirage: you'd have done strictly better holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for c in [0.0, 10.0, 25.0, 50.0]:\n"
            "        s = st.summarize_timing(st.timing_rule(BARS, 0.0, 0.5, cost_bps=c))\n"
            "        rows.append((c, s['cagr_strat']*100))\n"
            "    cc = [r[0] for r in rows]; cg = [r[1] for r in rows]\n"
            f"    bh = st.summarize_timing(st.timing_rule(BARS, 0.0, 0.5))['cagr_bh']*100\n"
            "else:\n"
            f"    cc = [0,10,25,50]; cg = [36.0,35.1,33.8,31.7]; bh = {R['cagr_bh']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(cc, cg, 'o-', c=AMBER, lw=2, label='halving-timing')\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=2, label='buy & hold')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('annualised %/yr')\n"
            "ax.set_title('Timing stays below buy-and-hold at every cost level')\n"
            "ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even at zero cost the timing rule trails buy-and-hold.')"
        ),
        md(
            "Even at **zero** cost the timing rule sits below the buy-and-hold line. The "
            "binding problem is the cash drag, not the trading fee — the definition of a "
            "tradability **mirage**."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further\n\n"
            "- **Two cycles is not a law.** The whole edifice rests on three (really two clean) "
            "four-year windows. The honest next step is *more cycles* — which only time can "
            "provide — or a cross-section of other halving coins (LTC, BCH) to borrow "
            "statistical power, with all the caveats that brings.\n"
            "- **The cousins on this desk.** [Study 117 — Pi-Cycle-Top](../../117-pi-cycle-top/) "
            "and [Study 174 — Bitcoin-Rainbow](../../174-bitcoin-rainbow/) are the same family: "
            "a compelling overlay fit to one price path that calls the top in hindsight.\n"
            "- **The supply-shock story.** [Study 292 — Bitcoin-Hashrate](../../292-bitcoin-hashrate/) "
            "and [Study 293 — MVRV](../../293-mvrv-ratio/) test other 'this predicts BTC' "
            "on-chain theses — each, like this one, loses to simply holding.\n\n"
            "*Think the cycle is real? Fork this, add LTC and BCH halvings, and show the "
            "post-halving glow surviving once the calendar can't sit in cash through the "
            "biggest up-leg.*"
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
            "# BTC-Halving — a quantitative teardown\n"
            "### Real BTC-USD tape · cycle-extrema scatter · phase-bucket HAC *t* · the "
            "three-cluster problem · timing vs buy-and-hold · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Prints_the_top_%26_bottom%3F: Busted](https://img.shields.io/badge/Prints_the_top_%26_bottom%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error (and its honest "
            "sample size).* We test whether the halving-locked cycle is statistically real, "
            "whether the halving marks the cycle extrema, and whether a calendar-timed rule "
            "beats buy-and-hold — and we name the central limit loudly: **Yahoo covers three "
            "halvings, so the effective *n* is three cycles.**\n\n"
            "> **Not investment advice.** Real data: Yahoo `BTC-USD` daily, 2014-09-17 → "
            f"{R['asof']} (fingerprint `{R['fingerprint']}`); the offline core and tests run "
            "on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Post-halving quarter +{R['pb0_mean']:.1f} bps/day, naive "
            f"HAC *t* +{R['pb0_t']:.2f} — but fed by **{R['n_cycles']}** cycles "
            f"(+{R['q0_2016']:.0f}/+{R['q0_2020']:.0f}/+{R['q0_2024']:.0f} bps/day, decaying). "
            "A daily *t* on three clusters overstates it. |\n"
            f"| **Tradability** | `MIRAGE` | Timing rule +{R['cagr_strat']:.0f}%/yr vs B&H "
            f"+{R['cagr_bh']:.0f}%/yr; excess HAC *t* {R['excess_t']:+.2f}, bootstrap CI "
            f"[{R['boot_lo']:+.1f}, {R['boot_hi']:+.1f}] bps/day. Underperforms holding. |\n"
            f"| **Prints the top & bottom?** | `BUSTED` | High offsets "
            f"{R['hi_offsets']} d; low offsets {R['lo_offsets']} d — no cluster. |\n\n"
            "> In plain words: a faint, fading post-halving drift measured on three cycles, an "
            "extrema pattern that doesn't exist, and a calendar timer that loses to buy-and-hold."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\phi_t \\in [0,1)$ be the fractional phase within the 4-year halving cycle "
            "(0 at a halving). The lore makes four testable claims:\n\n"
            "- **H₁ (cycle exists).** $\\mathbb{E}[r_t \\mid \\phi_t]$ varies with phase — the "
            "post-halving phase pays more.\n"
            "- **H₂ (the halving prints the top).** The cycle high clusters at "
            "$\\phi \\approx 0.25\\text{–}0.4$ (~12-18 months after).\n"
            "- **H₃ (the halving prints the bottom).** The cycle low clusters near $\\phi "
            "\\approx 0$ or $\\phi \\approx 1$.\n"
            "- **H₄ (tradable).** A long-the-bull-half rule beats buy-and-hold net of costs.\n\n"
            "We find **H₁ weakly supported but uncertifiable** (n = 3 cycles), **H₂ and H₃ "
            "rejected** (extrema scatter), and **H₄ rejected** (it loses to holding)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A *scheduled* supply cut cannot be a tradable surprise — efficient markets price "
            "a known event in advance (Fama 1970) — so any post-halving drift must come from "
            "something other than the cut being news. Stock-to-flow tried to supply that "
            "mechanism and broke down out of sample, the classic spurious-regression of two "
            "trending series (Granger-Newbold 1974). The interesting content isn't the binary "
            "verdict; it's **how a three-cluster sample masquerades as a thousand independent "
            "days**, and how a calendar 'edge' on a single trending asset is just beta with a "
            "cash drag."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Extrema.** For each halving, the day-offset of the price high/low in the "
            "$[h, h+1458\\text{d})$ window (`cycle_extrema`). Cluster → H₂/H₃; scatter → reject.\n"
            "- **Phase buckets.** Mean daily log-return in each quarter of $\\phi$, with a "
            "Newey-West HAC *t* (`phase_bucket_returns`) — **read against the cycle count.**\n"
            "- **Per-cycle reveal.** The post-halving-quarter mean *per halving* — the honest "
            "unit of observation.\n"
            "- **Timing race.** Long $\\phi \\in [0, 0.5)$ else flat; one-bar fill lag, "
            "one-way × NAV costs; excess-vs-buy-and-hold HAC *t* and block-bootstrap CI.\n"
            "- **Positive control.** A deterministic GBM tape with a tunable halving-locked "
            "sinusoid — the harness must detect the planted cycle and find nothing in the null.\n\n"
            "Tape: Yahoo `BTC-USD` daily, **2014-09-17 →** — covering 2016 (partial), 2020, 2024."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Do the extrema cluster? (H₂, H₃)\n\n"
            "Day-offset of each cycle's high and low, relative to the halving."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ext = st.cycle_extrema(BARS)\n"
            "    print(ext.to_string(index=False))\n"
            "    hi = ext['high_offset_d'].tolist(); lo = ext['low_offset_d'].tolist()\n"
            "    labs = [str(h.date()) for h in ext['halving']]\n"
            "else:\n"
            f"    hi = {R['hi_offsets']}; lo = {R['lo_offsets']}\n"
            "    labs = ['2012-11','2016-07','2020-05','2024-04']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "x = np.arange(len(labs))\n"
            "ax.scatter(hi, x, s=120, color=GREEN, label='HIGH', zorder=3)\n"
            "ax.scatter(lo, x, s=120, color=RED, label='LOW', zorder=3)\n"
            "ax.axvspan(365, 550, color=AMBER, alpha=.15, label='lore: top window')\n"
            "ax.set_yticks(x); ax.set_yticklabels(labs); ax.set_xlabel('days after halving')\n"
            "ax.set_title('Cycle extrema do not cluster — H2/H3 rejected')\n"
            "ax.legend(loc='lower right', fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: highs at {R['hi_offsets']} days, lows at {R['lo_offsets']} "
            "days. No cluster — the halving marks neither the top nor the bottom. **H₂, H₃ "
            "rejected.**"
        ),
        md(
            "### 4b · The phase-bucket drift and the three-cluster trap (H₁)\n\n"
            "Mean daily return by cycle quarter, with the *naive* daily HAC *t* — then the "
            "honest per-cycle view that deflates it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.phase_bucket_returns(BARS, 4)\n"
            "    print(pb.round(3).to_string(index=False))\n"
            "    means = pb['mean_bps'].tolist(); ts = pb['tstat'].tolist()\n"
            "    lr = np.log(BARS['close']).diff()\n"
            "    per = []\n"
            "    for h in data.HALVINGS:\n"
            "        if not (BARS.index.min() <= h <= BARS.index.max()):\n"
            "            continue\n"
            "        seg = lr[(lr.index >= h) & (lr.index < h + pd.Timedelta(days=364))].dropna()\n"
            "        per.append((str(h.date()), seg.mean()*1e4))\n"
            "    per_labs = [p[0] for p in per]; per_vals = [p[1] for p in per]\n"
            "else:\n"
            f"    means = [{R['pb0_mean']}, {R['pb1_mean']}, {R['pb2_mean']}, {R['pb3_mean']}]\n"
            f"    ts = [{R['pb0_t']}, 0.03, 0.18, {R['pb3_t']}]\n"
            f"    per_labs = ['2016-07','2020-05','2024-04']\n"
            f"    per_vals = [{R['q0_2016']}, {R['q0_2020']}, {R['q0_2024']}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "qcol = [GREEN if t >= 2 else AMBER for t in ts]\n"
            "a1.bar([f'Q{i}' for i in range(len(means))], means, color=qcol, width=.6)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_ylabel('mean daily ret (bps)')\n"
            "a1.set_title(f'Q0 (post-halving) leads: t={ts[0]:+.2f} (NAIVE)')\n"
            "a2.bar(per_labs, per_vals, color=[GREEN if v>20 else AMBER for v in per_vals], width=.55)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('post-halving year, daily bps')\n"
            "a2.set_title(f'...but on {len(per_vals)} cycles, and fading')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-cycle post-halving daily bps:', [round(v,1) for v in per_vals])"
        ),
        md(
            f"> 💡 In plain words: the post-halving quarter's naive HAC *t* of "
            f"+{R['pb0_t']:.2f} *looks* like it clears the bar — but it pools ~1000 "
            f"autocorrelated days from **{R['n_cycles']} halvings** as if they were "
            "independent. The honest unit is the cycle, and the three cycles paid "
            f"+{R['q0_2016']:.0f}/+{R['q0_2020']:.0f}/+{R['q0_2024']:.0f} bps/day — *decaying*, "
            "with the latest near zero. With three clusters no *t* is trustworthy "
            "(Cameron-Gelbach-Miller 2011). **H₁ weak, not real.**"
        ),
        md(
            "### 4c · The timing race (H₄)\n\n"
            "Long the bull half of each cycle vs buy-and-hold — the only honest benchmark for "
            "a single trending asset. Excess-vs-buy-and-hold with HAC *t* and bootstrap CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.timing_rule(BARS, 0.0, 0.5, cost_bps=10.0)\n"
            "    s = st.summarize_timing(book)\n"
            "    lo, hi = st.block_bootstrap_ci(book['ret_strat'].to_numpy() - book['ret_bh'].to_numpy())\n"
            "    cs, cb, et = s['cagr_strat']*100, s['cagr_bh']*100, s['excess_t']\n"
            "    tim, fl = s['time_in_market']*100, s['n_flips']\n"
            "    eqs, eqb = book['eq_strat'], book['eq_bh']\n"
            "else:\n"
            f"    cs, cb, et = {R['cagr_strat']}, {R['cagr_bh']}, {R['excess_t']}\n"
            f"    lo, hi = {R['boot_lo']}, {R['boot_hi']}; tim, fl = {R['time_in_market']}, {R['n_flips']}\n"
            "    eqs = eqb = None\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "if eqs is not None:\n"
            "    ax.plot(eqb.index, eqb, c=GREEN, lw=1.6, label=f'buy & hold ({cb:.0f}%/yr)')\n"
            "    ax.plot(eqs.index, eqs, c=AMBER, lw=1.6, label=f'halving timing ({cs:.0f}%/yr)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "else:\n"
            "    ax.bar(['timing','buy & hold'], [cs, cb], color=[AMBER, GREEN], width=.5)\n"
            "    ax.set_ylabel('annualised %/yr')\n"
            "ax.set_title('Timing the cycle loses to holding it')\n"
            "ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timing {cs:.1f}%/yr  vs  B&H {cb:.1f}%/yr   excess t={et:+.2f}   CI[{lo:+.1f},{hi:+.1f}] bps/day')\n"
            "print(f'time in market {tim:.0f}%, {fl} flips')"
        ),
        md(
            f"> 💡 In plain words: +{R['cagr_strat']:.0f}%/yr vs +{R['cagr_bh']:.0f}%/yr, excess "
            f"HAC *t* {R['excess_t']:+.2f}, bootstrap CI "
            f"[{R['boot_lo']:+.1f}, {R['boot_hi']:+.1f}] bps/day straddling zero (tilted "
            "negative). The timer sits in cash through part of the biggest up-leg and loses. "
            "**H₄ rejected.**"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — post-halving quarter +{R['pb0_mean']:.1f} bps/day (naive "
            f"HAC *t* +{R['pb0_t']:.2f}), but on {R['n_cycles']} cycles "
            f"(+{R['q0_2016']:.0f}/+{R['q0_2020']:.0f}/+{R['q0_2024']:.0f}, fading). Literature-"
            "flavoured, uncertifiable on this tape.\n"
            f"- **Tradability `MIRAGE`** — timing +{R['cagr_strat']:.0f}%/yr vs B&H "
            f"+{R['cagr_bh']:.0f}%/yr; excess *t* {R['excess_t']:+.2f}. Loses to holding.\n"
            f"- **Prints the top & bottom? `BUSTED`** — extrema at {R['hi_offsets']} / "
            f"{R['lo_offsets']} days, no cluster."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the cost sweep and the real wall\n\n"
            "Costs are a sideshow (the rule flips a handful of times); the wall is the cash "
            "drag against a trending asset."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cc = [0.0, 10.0, 25.0, 50.0]\n"
            "    cg = [st.summarize_timing(st.timing_rule(BARS, 0.0, 0.5, cost_bps=c))['cagr_strat']*100 for c in cc]\n"
            "    bh = st.summarize_timing(st.timing_rule(BARS, 0.0, 0.5))['cagr_bh']*100\n"
            "else:\n"
            f"    cc = [0,10,25,50]; cg = [36.0,35.1,33.8,31.7]; bh = {R['cagr_bh']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(cc, cg, 'o-', c=AMBER, lw=2, label='halving-timing')\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=2, label='buy & hold')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('annualised %/yr')\n"
            "ax.set_title('Below buy-and-hold at every cost — incl. zero')\n"
            "ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even gross (0 bps) the timer trails buy-and-hold: the constraint is cash drag, not cost.')"
        ),
        md(
            "> 💡 In plain words: the timer is below buy-and-hold even at **zero cost**. The "
            "binding constraint is the days out of the market, not the trading fee — a "
            "tradability **mirage**, not a fragile-but-real edge."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful cycle detector, or does it always print the same "
            "number? Plant a halving-locked sinusoid of varying amplitude and read the "
            "post-halving bucket's HAC *t*."
        ),
        code(
            "amps = [0.0, 0.5, 1.0, 1.5, 2.0]\n"
            "tstats = []\n"
            "for a in amps:\n"
            "    b, _ = data.synthetic_daily(n_days=4200, cycle_amp=a, seed=323)\n"
            "    tstats.append(st.phase_bucket_returns(b, 4).loc[0, 'tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if t >= 2 else GREY for t in tstats]\n"
            "ax.bar([str(a) for a in amps], tstats, color=col, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted halving-cycle amplitude'); ax.set_ylabel('post-halving bucket HAC t')\n"
            "ax.set_title('The engine detects a planted cycle and finds nothing in the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('null (amp=0) t=%.2f ; planted (amp=1.5) t=%.2f' % (tstats[0], tstats[3]))"
        ),
        md(
            f"The post-halving bucket's *t* is ~{R['syn_null_t']:.1f} at the null (amp = 0) and "
            f"rises to ~{R['syn_cycle_t']:.1f} with a planted cycle — so the engine is faithful, "
            "and the real-tape bucket-0 number is a statement about *three BTC cycles*, not a "
            "machinery artefact. The synthetic Sharpe never backs the Signal stamp.\n\n"
            "For the same 'overlay calls the top in hindsight' family, see "
            "[Study 117 — Pi-Cycle-Top](../../117-pi-cycle-top/) and "
            "[Study 174 — Bitcoin-Rainbow](../../174-bitcoin-rainbow/); for other on-chain "
            "'predicts BTC' theses that also lose to holding, "
            "[Study 292 — Bitcoin-Hashrate](../../292-bitcoin-hashrate/)."
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
