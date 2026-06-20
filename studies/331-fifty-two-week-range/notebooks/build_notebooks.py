"""Generate the two narrative notebooks for Study 331 (Fifty-Two-Week-Range).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
control figures run anywhere, offline and deterministic; the real-tape cells use the
cached daily parquets (this study's _cache or the shared study-202 cache) if present, and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    # Range-position (POS) Q5-Q1 spread, gross, weekly, by horizon
    pos1_bps=11.1, pos1_t=2.00, pos1_win=54.5, pos1_ci_lo=1.2, pos1_ci_hi=22.5,
    pos5_bps=7.4, pos5_t=0.67, pos21_bps=12.8, pos21_t=0.32, pos65_bps=37.3, pos65_t=0.45,
    # High-ratio (HR) spread t by horizon
    hr1_t=1.58, hr5_t=-0.85, hr21_t=-1.29, hr65_t=-1.39,
    # Paired difference POS - HR
    diff1_t=0.17, diff5_bps=18.0, diff5_t=2.72, diff5_ci_lo=4.5, diff5_ci_hi=30.3,
    diff21_bps=70.6, diff21_t=3.42, diff65_bps=176.4, diff65_t=4.04,
    # Spanning regression (incremental, z-scored)
    inc_pos5_t=1.87, inc_pos21_t=2.16, inc_hr21_t=-2.26,
    # Long-only top quintile vs equal-weight baseline (5d)
    lo_pos_ann=12.0, lo_base_ann=11.3, lo_pos_t=0.26,
    # Cost sweep on POS 21d spread
    c0_bps=12.8, c0_t=0.32, c5_bps=7.8, c10_bps=2.8, c20_bps=-7.2, c20_t=-0.18,
    # Data stamp
    n_days=3383, n_names=20, fp_close="280c2c864e21",
    start="2013-01-02", end="2026-06-15",
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

from fifty_two_week_range import data, strategy as st

HAVE_REAL = data.have_real_cache()

def real_panels():
    return data.load_ohlc_panels(allow_survivorship_bias=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Fifty-Two-Week-Range — is *where in the range* smarter than *off the high*?\n"
            "### A favourite momentum read, raced head-to-head against the simpler signal, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Better than the high?: Confirmed, but hollow](https://img.shields.io/badge/Better_than_the_high%3F-Confirmed,_but_hollow-8b949e?style=flat-square)\n\n"
            "Traders love to say a stock is \"trading at 90% of its 52-week range\" — it sounds "
            "richer than \"5% off its high\", because it looks at *both* ends of the year at once. "
            "George & Hwang (2004) made the high-only version famous; this notebook tests the "
            "fuller idea: does adding the 52-week **low** as a second anchor actually help? We race "
            "the two signals against each other and ask the only thing that matters — *does the low "
            "tell you anything the high doesn't already?*\n\n"
            "> **This is the plain-language layer.** Want the *t*-stats, the paired horse-race test "
            "and the regression? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does range position predict returns? | **No.** Its top-minus-bottom spread is "
            f"noise at every sensible horizon (5d *t* = +{R['pos5_t']:.2f}, 21d +{R['pos21_t']:.2f}); "
            f"only the 1-day spread touches the bar (*t* = +{R['pos1_t']:.2f}) — and that's mostly "
            "bid-ask bounce. |\n"
            "| Does the *low* add anything over the *high*? | **Statistically yes** — range position "
            f"beats the high-only signal in a head-to-head (*t* up to +{R['diff65_t']:.2f}). |\n"
            "| ...so range position is the better signal? | **It's a hollow win.** It only wins "
            "because the high-only signal is *losing* on this sample. Being less bad than a loser "
            "isn't an edge. |\n"
            "| Could you trade it? | **No.** The biggest gross spread isn't even significant, and it "
            f"turns negative by ~15 bps of cost. Nothing to charge costs against. |\n\n"
            "> The richer-sounding signal really is richer than the high alone — but \"richer than a "
            "signal that's quietly losing money\" buys you nothing you can bank."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Don't just look at how far a stock is from its 52-week high — look at **where it "
            "sits in its whole 52-week range**. A stock at 95% of a wide range, well clear of its "
            "low, is a more *confirmed* winner than one merely brushing a high it keeps revisiting.\"*\n\n"
            "It's the natural generalisation of the famous George & Hwang (2004) result. Their signal "
            "is `close / 52-week-high`. The range version, "
            "`(close − 52w-low) / (52w-high − 52w-low)`, anchors on **both** ends. The bet: the "
            "second anchor — the low — carries extra information."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The desk has already tested the two halves on their own: the high in isolation "
            "([Study 236](../../236-fifty-two-week-high/)) and the low in isolation "
            "([Study 202](../../202-fifty-two-week-low/)). Both came up short on this large-cap "
            "sample. So this study isn't \"does the high work\" again — it's the sharper, *marginal* "
            "question: **when you combine both anchors, does the extra one earn its keep?** That's a "
            "question about whether more-elaborate framings of the same data actually add information "
            "— a lesson that transfers far beyond this one signal."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We **race the two signals against each other** on the same days:\n\n"
            "1. **Each signal's own spread.** Rank the 20 stocks by each signal, buy the top fifth, "
            "sell the bottom fifth, measure the gap. If range position has a standalone edge, its "
            "spread is reliably positive.\n"
            "2. **The head-to-head difference.** Subtract one spread from the other, *day by day*, and "
            "test that paired difference. This is the honest way to claim one beats the other — not "
            "two separate *t*-stats.\n"
            "3. **What would make us say 'mirage'.** A standalone spread that's noise, or one that "
            "evaporates the moment we charge realistic costs.\n\n"
            "Tape: 20 S&P 500 large-caps, 2013–2026 (survivorship-biased — all still trade)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does range position predict returns on its own?** Here's its top-minus-bottom "
            "spread across holding horizons, with the bar for significance marked."
        ),
        code(
            "horizons = [1, 5, 21, 65]\n"
            "if HAVE_REAL:\n"
            "    close = real_panels()['close']; pos = st.range_position(close)\n"
            "    ts = []\n"
            "    for h in horizons:\n"
            "        fwd = st.forward_returns(close, h)\n"
            "        ts.append(st.summarize(st.quintile_spread(pos, fwd)['spread_gross'])['tstat'])\n"
            "else:\n"
            f"    ts = [{R['pos1_t']}, {R['pos5_t']}, {R['pos21_t']}, {R['pos65_t']}]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "col = [GREEN if abs(t) >= 2 else GREY for t in ts]\n"
            "ax.bar([f'{h}d' for h in horizons], ts, color=col, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='|t| = 2 bar')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat of the Q5-Q1 spread'); ax.legend()\n"
            "ax.set_title('Range position alone: only the 1-day spread touches the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spread t by horizon:', [round(t,2) for t in ts])"
        ),
        md(
            f"Only the **1-day** spread reaches the bar (*t* = +{R['pos1_t']:.2f}) — and a 1-day "
            "cross-sectional spread is exactly where mechanical bid-ask bounce hides, not a tradable "
            f"edge. At 5, 21 and 65 days the spread is **noise** (*t* = +{R['pos5_t']:.2f}, "
            f"+{R['pos21_t']:.2f}, +{R['pos65_t']:.2f}). On its own, range position predicts nothing."
        ),
        md(
            "**Now the study's real question: does it at least beat the high-only signal?** Here are "
            "both signals' spreads side by side — and the head-to-head difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = real_panels()['close']\n"
            "    pos, hr = st.range_position(close), st.high_ratio(close)\n"
            "    pos_t, hr_t, diff_t = [], [], []\n"
            "    for h in horizons:\n"
            "        fwd = st.forward_returns(close, h)\n"
            "        qp = st.quintile_spread(pos, fwd); qh = st.quintile_spread(hr, fwd)\n"
            "        pos_t.append(st.summarize(qp['spread_gross'])['tstat'])\n"
            "        hr_t.append(st.summarize(qh['spread_gross'])['tstat'])\n"
            "        diff_t.append(st.diff_tstat(qp['spread_gross'], qh['spread_gross'])['tstat'])\n"
            "else:\n"
            f"    pos_t = [{R['pos1_t']}, {R['pos5_t']}, {R['pos21_t']}, {R['pos65_t']}]\n"
            f"    hr_t  = [{R['hr1_t']}, {R['hr5_t']}, {R['hr21_t']}, {R['hr65_t']}]\n"
            f"    diff_t= [{R['diff1_t']}, {R['diff5_t']}, {R['diff21_t']}, {R['diff65_t']}]\n"
            "x = np.arange(len(horizons)); w = 0.27\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.bar(x-w, pos_t, w, label='Range position', color=GREEN)\n"
            "ax.bar(x,   hr_t, w, label='High only (236)', color=RED)\n"
            "ax.bar(x+w, diff_t, w, label='Difference (POS - HR)', color=GREY)\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('HAC t-stat'); ax.legend()\n"
            "ax.set_title('Range position beats the high — but only because the high is LOSING')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('POS-HR difference t:', [round(t,2) for t in diff_t])"
        ),
        md(
            f"There's the twist. The **difference** is significant (5d *t* = +{R['diff5_t']:.2f}, "
            f"21d +{R['diff21_t']:.2f}, 65d +{R['diff65_t']:.2f}) — range position genuinely beats the "
            "high-only signal. **But look at the red bars:** the high-only signal's own spread is "
            f"*negative* (*t* = {R['hr21_t']:.2f} at 21d). Range position wins the race because its "
            "rival is running backwards. Being less-bad than a loser is not an edge."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Range position's own spread is noise except a fragile 1-day "
            f"*t* = +{R['pos1_t']:.2f} (bid-ask bounce); long-only it ties the basket "
            f"(+{R['lo_pos_ann']:.1f}% vs +{R['lo_base_ann']:.1f}%/yr, *t* = +{R['lo_pos_t']:.2f}).\n"
            "- **Tradability — Mirage.** The biggest gross spread isn't significant and turns "
            "negative by ~15 bps round-trip.\n"
            "- **Better than the high alone? — Confirmed, but hollow.** It wins the head-to-head "
            "(*t* up to +" f"{R['diff65_t']:.2f}) only by avoiding the high signal's inverted drag."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's barely anything to trade in the first place — but watch what little there is do "
            "under realistic costs (the 21-day spread, the largest gross number)."
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    close = real_panels()['close']; pos = st.range_position(close)\n"
            "    fwd = st.forward_returns(close, 21)\n"
            "    net = [st.summarize(st.quintile_spread(pos, fwd, cost_bps=c)['spread_net'])['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['c0_bps']}, {R['c5_bps']}, {R['c10_bps']}, {R['c20_bps']}]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n>=0 for n in net], color=GREEN, alpha=.10)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.10)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net spread (bps/period)')\n"
            "ax.set_title('It was never significant gross — and goes negative by ~15 bps')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net spread by cost:', [round(n,1) for n in net])"
        ),
        md(
            f"The gross spread (+{R['c0_bps']:.1f} bps) was never significant (*t* = +{R['c0_t']:.2f}), "
            "and costs only make it worse — negative by ~15 bps. Unlike a real-but-fragile edge, "
            "there's no there there to protect. **Mirage.**"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The two halves.** [Study 236 — Fifty-Two-Week-High](../../236-fifty-two-week-high/) "
            "and [Study 202 — Fifty-Two-Week-Low](../../202-fifty-two-week-low/) test each anchor "
            "alone — both inverted/weak on this large-cap survivor sample.\n"
            "- **Survivorship.** Our basket is 20 survivors. The original George-Hwang result lived on "
            "a broad, unbiased CRSP universe in 1963–2001. A delisting-aware small-cap panel is the "
            "fork most likely to revive *any* of these signals.\n"
            "- **A real short-horizon signal for contrast.** "
            "[Study 75 — Knee-Jerk](../../75-knee-jerk/) is what a genuine mean-reversion edge looks "
            "like — useful to hold next to a non-signal like this one.\n\n"
            "*Think the low anchor earns its keep somewhere? Fork this, swap in a delisting-aware "
            "small-cap panel, and show the standalone spread — not just the head-to-head — clearing "
            "the bar.*"
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
            "# Fifty-Two-Week-Range — a quantitative teardown\n"
            "### Real daily panel · cross-sectional quintiles · paired spread-vs-spread HAC test · "
            "Fama-MacBeth spanning regression · block-bootstrap CIs · two-knob synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Better than the high?: Confirmed, but hollow](https://img.shields.io/badge/Better_than_the_high%3F-Confirmed,_but_hollow-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the 52-week "
            "**range position** has a standalone cross-sectional edge, and — the study's actual "
            "question — whether it carries *incremental* information over the George-Hwang **high "
            "ratio** once we control for it.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 20 S&P 500 large-caps "
            f"({R['start']}→{R['end']}, **survivorship-biased**); as-of {R['end']}; the offline core "
            "and tests run on a deterministic two-knob synthetic panel. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Range-position Q5−Q1 spread: 1d *t* = +{R['pos1_t']:.2f} "
            f"(bootstrap CI [+{R['pos1_ci_lo']:.1f},+{R['pos1_ci_hi']:.1f}] bps, bid-ask-tainted), "
            f"5/21/65d *t* = +{R['pos5_t']:.2f}/+{R['pos21_t']:.2f}/+{R['pos65_t']:.2f}; long-only "
            f"*t* = +{R['lo_pos_t']:.2f}. No robust standalone edge. |\n"
            f"| **Tradability** | `MIRAGE` | Largest gross spread +{R['c0_bps']:.1f} bps "
            f"(*t* = +{R['c0_t']:.2f}); net turns negative ({R['c20_bps']:.1f} bps) by 20 bps. |\n"
            f"| **Better than the high alone?** | `CONFIRMED` (hollow) | Paired POS−HR difference "
            f"5/21/65d *t* = +{R['diff5_t']:.2f}/+{R['diff21_t']:.2f}/+{R['diff65_t']:.2f}; spanning "
            f"incremental POS\\|HR *t* = +{R['inc_pos21_t']:.2f} while HR\\|POS = {R['inc_hr21_t']:.2f}. "
            "Range-position wins only because the high ratio is inverted here. |\n\n"
            "> In plain words: the second anchor (the low) really does add information over the high "
            "alone — but on this sample the high alone is a *losing* signal, so \"adds information\" "
            "buys a relative win, not a tradable one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the price, $H_t,L_t$ the trailing 252-day high/low. Define\n\n"
            "$$\\mathrm{pos}_t=\\frac{P_t-L_t}{H_t-L_t}\\in[0,1],\\qquad "
            "\\mathrm{hr}_t=\\frac{P_t}{H_t}\\in(0,1].$$\n\n"
            "- **H₁ (standalone).** The cross-sectional Q5−Q1 spread on $\\mathrm{pos}$ has "
            "$\\mathbb{E}[\\text{spread}]>0$ at HAC *t* ≥ 2.\n"
            "- **H₂ (incremental, the study's spine).** $\\mathrm{pos}$ beats $\\mathrm{hr}$: the "
            "*paired difference* of their spreads, and the Fama-MacBeth slope on $\\mathrm{pos}$ "
            "controlling for $\\mathrm{hr}$, are positive at *t* ≥ 2.\n"
            "- **H₃ (tradable).** Any spread survives realistic one-way × NAV costs.\n\n"
            "We find **H₁ rejected** (no standalone edge), **H₂ supported but hollow** (pos beats hr "
            "only because hr is inverted on this sample), **H₃ rejected** (cost wall)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₂ held *and* H₁ held, the low would be a genuine free upgrade to a working momentum "
            "signal. If H₂ holds but H₁ doesn't — which is what we find — the lesson is subtler and "
            "more useful: **a more-elaborate framing can be *measurably better* than a simpler one "
            "and still be worthless**, because \"better\" is relative and the baseline was already "
            "underwater. Naming that distinction precisely is the product here."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signals.** $\\mathrm{pos}$ and $\\mathrm{hr}$ on trailing 252-day high/low, formed "
            "on closes up to *t*.\n"
            "- **Forward return.** $\\log(P_{t+h}/P_t)$, $h\\in\\{1,5,21,65\\}$ — the **one** "
            "execution lag, no second shift.\n"
            "- **Quintile spread.** Equal-weight Q5−Q1 per weekly rebalance; HAC *t* on the series.\n"
            "- **The horse race.** (a) paired difference of the two spreads on common dates "
            "(`diff_tstat`); (b) Fama-MacBeth per-period cross-sectional OLS of forward returns on "
            "z-scored $\\mathrm{pos}$ and $\\mathrm{hr}$ together (`spanning_regression`).\n"
            "- **Inference.** Newey-West HAC; circular block-bootstrap CIs.\n"
            "- **Costs.** One-way × NAV; shorts pay borrow via the round-trip charge.\n"
            "- **Positive control.** Deterministic synthetic panel with two separable knobs — a "
            "shared confound (`range_edge`) and the low-only discriminating edge (`low_edge`).\n\n"
            "Universe: 20 S&P 500 large-caps, survivorship-biased (named on the Signal axis)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Standalone spread & its bootstrap CI (H₁)\n\n"
            "The range-position spread by horizon, with a block-bootstrap CI on the only borderline "
            "horizon (1 day)."
        ),
        code(
            "horizons = [1, 5, 21, 65]\n"
            "if HAVE_REAL:\n"
            "    close = real_panels()['close']; pos = st.range_position(close)\n"
            "    recs = []\n"
            "    for h in horizons:\n"
            "        fwd = st.forward_returns(close, h)\n"
            "        s = st.summarize(st.quintile_spread(pos, fwd)['spread_gross'])\n"
            "        recs.append((h, s['mean_bps'], s['tstat']))\n"
            "    fwd1 = st.forward_returns(close, 1)\n"
            "    ci = st.block_bootstrap_ci(st.quintile_spread(pos, fwd1)['spread_gross'].dropna())\n"
            "else:\n"
            f"    recs = [(1,{R['pos1_bps']},{R['pos1_t']}),(5,{R['pos5_bps']},{R['pos5_t']}),"
            f"(21,{R['pos21_bps']},{R['pos21_t']}),(65,{R['pos65_bps']},{R['pos65_t']})]\n"
            f"    ci = ({R['pos1_ci_lo']}, {R['pos1_ci_hi']})\n"
            "dfp = pd.DataFrame(recs, columns=['horizon','mean_bps','t'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "col = [GREEN if abs(t)>=2 else GREY for t in dfp['t']]\n"
            "ax.bar(dfp['horizon'].astype(str)+'d', dfp['t'], color=col, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t (Q5-Q1 spread)')\n"
            "ax.set_title('H1: no standalone edge beyond a fragile, bounce-tainted 1-day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfp.round(2).to_string(index=False))\n"
            "print(f'1-day spread bootstrap 95% CI: [{ci[0]:+.1f}, {ci[1]:+.1f}] bps')"
        ),
        md(
            f"> 💡 In plain words: the 1-day *t* = +{R['pos1_t']:.2f} barely clears the bar and its CI "
            f"([+{R['pos1_ci_lo']:.1f},+{R['pos1_ci_hi']:.1f}] bps) hugs zero — and 1-day "
            "cross-sectional spreads are where bid-ask bounce (Roll 1984) masquerades as alpha. "
            "Everything longer is noise. **H₁ rejected.**"
        ),
        md(
            "### 4b · The horse race — paired difference & spanning regression (H₂)\n\n"
            "The decisive, *relative* test. Left: the paired POS−HR spread difference per horizon. "
            "Right: the Fama-MacBeth incremental slopes (each signal controlling for the other)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = real_panels()['close']\n"
            "    pos, hr = st.range_position(close), st.high_ratio(close)\n"
            "    diff_t = []\n"
            "    for h in horizons:\n"
            "        fwd = st.forward_returns(close, h)\n"
            "        qp = st.quintile_spread(pos, fwd); qh = st.quintile_spread(hr, fwd)\n"
            "        diff_t.append(st.diff_tstat(qp['spread_gross'], qh['spread_gross'])['tstat'])\n"
            "    fwd21 = st.forward_returns(close, 21)\n"
            "    inc_pos = st.hac_tstat(st.spanning_regression(pos, hr, fwd21)['b_target'])\n"
            "    inc_hr  = st.hac_tstat(st.spanning_regression(hr, pos, fwd21)['b_target'])\n"
            "else:\n"
            f"    diff_t = [{R['diff1_t']}, {R['diff5_t']}, {R['diff21_t']}, {R['diff65_t']}]\n"
            f"    inc_pos, inc_hr = {R['inc_pos21_t']}, {R['inc_hr21_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))\n"
            "c1 = [GREEN if abs(t)>=2 else GREY for t in diff_t]\n"
            "a1.bar([f'{h}d' for h in horizons], diff_t, color=c1, width=.55)\n"
            "a1.axhline(2, ls='--', c='k'); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('HAC t (POS - HR spread)'); a1.set_title('Range position beats the high...')\n"
            "a2.bar(['POS | HR', 'HR | POS'], [inc_pos, inc_hr], color=[GREEN, RED], width=.5)\n"
            "a2.axhline(2, ls='--', c='k'); a2.axhline(-2, ls='--', c='k'); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_ylabel('incremental HAC t (21d)'); a2.set_title('...and carries info the high lacks')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('POS-HR diff t:', [round(t,2) for t in diff_t])\n"
            "print(f'incremental: POS|HR t={inc_pos:+.2f}   HR|POS t={inc_hr:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: range position wins the head-to-head (5d *t* = +{R['diff5_t']:.2f}, "
            f"65d +{R['diff65_t']:.2f}) and the spanning regression agrees — its incremental slope is "
            f"+{R['inc_pos21_t']:.2f} while the high ratio's is **{R['inc_hr21_t']:.2f}** (negative!). "
            "So **H₂ is supported** — but the high ratio's negative incremental slope is the whole "
            "reason: pos wins by *not* loading on a signal that's inverted on this large-cap survivor "
            "sample. A relative win over a loser, not an absolute edge. (Recall study 236 already "
            "found the George-Hwang anomaly inverted here.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pos spread noise except a fragile 1d *t* = +{R['pos1_t']:.2f} "
            f"(CI [+{R['pos1_ci_lo']:.1f},+{R['pos1_ci_hi']:.1f}] bps); long-only *t* = "
            f"+{R['lo_pos_t']:.2f}.\n"
            f"- **Tradability `MIRAGE`** — gross spread insignificant (*t* = +{R['c0_t']:.2f}), "
            f"negative by 20 bps cost ({R['c20_bps']:.1f} bps, *t* = {R['c20_t']:.2f}).\n"
            f"- **Better than the high? `CONFIRMED` (hollow)** — paired diff *t* up to "
            f"+{R['diff65_t']:.2f}, incremental +{R['inc_pos21_t']:.2f}; but the win is entirely the "
            "high ratio's inverted drag."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost wall and the long-only reality\n\n"
            "Even granting the largest gross spread, costs (one-way × NAV, shorts paying borrow) "
            "finish it; and a long-only investor gains nothing over the basket."
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    close = real_panels()['close']; pos = st.range_position(close)\n"
            "    fwd21 = st.forward_returns(close, 21)\n"
            "    net = [st.summarize(st.quintile_spread(pos, fwd21, cost_bps=c)['spread_net']) for c in costs]\n"
            "    nb = [s['mean_bps'] for s in net]; nt = [s['tstat'] for s in net]\n"
            "    fwd5 = st.forward_returns(close, 5)\n"
            "    s5, b5 = st.long_top_decile(pos, fwd5, decile_frac=0.20)\n"
            "    lo_ann, base_ann, lo_t = st.summarize(s5)['ann_pct'], st.summarize(b5)['ann_pct'], st.diff_tstat(s5,b5)['tstat']\n"
            "else:\n"
            f"    nb = [{R['c0_bps']}, {R['c5_bps']}, {R['c10_bps']}, {R['c20_bps']}]\n"
            f"    nt = [{R['c0_t']}, 0.20, 0.07, {R['c20_t']}]\n"
            f"    lo_ann, base_ann, lo_t = {R['lo_pos_ann']}, {R['lo_base_ann']}, {R['lo_pos_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))\n"
            "a1.plot(costs, nb, 'o-', c=AMBER, lw=2); a1.axhline(0, c='k', lw=1)\n"
            "a1.fill_between(costs, nb, 0, where=[v>=0 for v in nb], color=GREEN, alpha=.10)\n"
            "a1.fill_between(costs, nb, 0, where=[v<0 for v in nb], color=RED, alpha=.10)\n"
            "a1.set_xlabel('round-trip cost (bps)'); a1.set_ylabel('net spread (bps)')\n"
            "a1.set_title('Cost wall (21d spread)')\n"
            "a2.bar(['Top quintile\\n(range pos)', 'Equal-weight\\nbasket'], [lo_ann, base_ann],\n"
            "       color=[AMBER, GREEN], width=.5); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_ylabel('annualised %/yr'); a2.set_title(f'Long-only ties the basket (t={lo_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net spread by cost:', [round(v,1) for v in nb], 't:', [round(v,2) for v in nt])\n"
            "print(f'long-only {lo_ann:+.1f}%/yr vs baseline {base_ann:+.1f}%/yr (diff t={lo_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the gross was never significant, so cost can only worsen it — "
            f"negative by ~15 bps. And a long-only top quintile earns +{R['lo_pos_ann']:.1f}%/yr vs "
            f"the basket's +{R['lo_base_ann']:.1f}%/yr (*t* = +{R['lo_pos_t']:.2f}): you'd have done "
            "as well holding everything. Nothing to protect — **Mirage.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the two-knob synthetic positive control\n\n"
            "Is the *engine* faithful, or does it always declare one signal the winner? Plant (a) a "
            "**confound** both signals share and (b) a **discriminating** edge only the low anchor "
            "sees, and check the horse race reads each correctly."
        ),
        code(
            "scen = [('null\\n(no edge)', 0.0, 0.0),\n"
            "        ('confound\\n(both capture)', 0.004, 0.0),\n"
            "        ('low-only edge\\n(pos should win)', 0.0, 0.01)]\n"
            "pos_ts, hr_ts, diff_ts = [], [], []\n"
            "for _, re_, le_ in scen:\n"
            "    p, _ = data.synthetic_panel(range_edge=re_, low_edge=le_, n_days=1500, seed=331)\n"
            "    pos, hr = st.range_position(p), st.high_ratio(p)\n"
            "    fwd = st.forward_returns(p, 5)\n"
            "    qp = st.quintile_spread(pos, fwd); qh = st.quintile_spread(hr, fwd)\n"
            "    pos_ts.append(st.summarize(qp['spread_gross'])['tstat'])\n"
            "    hr_ts.append(st.summarize(qh['spread_gross'])['tstat'])\n"
            "    diff_ts.append(st.diff_tstat(qp['spread_gross'], qh['spread_gross'])['tstat'])\n"
            "x = np.arange(len(scen)); w = 0.27\n"
            "fig, ax = plt.subplots(figsize=(10, 4.4))\n"
            "ax.bar(x-w, pos_ts, w, label='range pos t', color=GREEN)\n"
            "ax.bar(x,   hr_ts, w, label='high ratio t', color=RED)\n"
            "ax.bar(x+w, diff_ts, w, label='POS-HR diff t', color=GREY)\n"
            "ax.axhline(2, ls='--', c='k'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([s[0] for s in scen])\n"
            "ax.set_ylabel('HAC t'); ax.legend()\n"
            "ax.set_title('The engine reads each planted regime correctly')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pos t :', [round(t,2) for t in pos_ts])\n"
            "print('hr  t :', [round(t,2) for t in hr_ts])\n"
            "print('diff t:', [round(t,2) for t in diff_ts])"
        ),
        md(
            "The control passes on all three counts: **null** → both coins, no separation; "
            "**confound** → both spreads strong, difference *not* significant (the low adds nothing "
            "when the edge is shared); **low-only edge** → range position beats the high ratio. So the "
            "real-tape finding is a statement about the market, not a broken pipeline: on this "
            "large-cap survivor sample the range anchor has no standalone edge, and its head-to-head "
            "win is a hollow consequence of the high ratio being inverted.\n\n"
            "Forks worth running: a delisting-aware small-cap panel (where George-Hwang originally "
            "lived), and a longer pre-2013 window. See "
            "[Study 236](../../236-fifty-two-week-high/) and [Study 202](../../202-fifty-two-week-low/) "
            "for the anchors in isolation."
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
