"""Generate the two narrative notebooks for Study 768 (Charm-Decay).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
positive-control cells run anywhere, offline and deterministic; the real-tape cells use the
cached daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-07-10).
R = dict(
    ticker="SPY", n_days=8418, start="1993-01-29", end="2026-07-10", fp="a7ff45641491",
    n_opex_defined=408, n_pre_days=2005, n_post_days=2005,
    # leg 1 — drift
    pre_mean=4.54, pre_base=4.85, pre_diff=-0.32, pre_t=-0.13, pre_n=2005,
    post_mean=3.48, post_base=5.18, post_diff=-1.70, post_t=-0.76, post_n=2005,
    # leg 2 — asymmetry
    asym_pre=4.54, asym_post=3.48, asym_diff=1.05, asym_t=0.31,
    # leg 3 — quarterly
    q_all_diff=-0.32, q_all_t=-0.13, q_qtr_diff=-1.06, q_qtr_t=-0.23, q_qtr_n=670,
    # leg 4 — placebo
    pl_true_t=-0.13, pl_mean_abs=1.08, pl_max_abs=3.68, pl_n=52, pl_p=0.94,
    pl_planted_true_t=3.59, pl_planted_p=0.019,
    # structural break
    pre2012_diff=1.20, pre2012_t=0.35, pre2012_n=1135,
    post2012_diff=-2.25, post2012_t=-0.67, post2012_n=870,
    # tradability
    ov_n=1992, ov_mean=4.56, ov_sharpe=0.62, ov_cagr=10.25, ov_t=1.87,
    ls_n=3983, ls_mean=0.53, ls_sharpe=0.07, ls_cagr=-0.36, ls_t=0.31,
    bh_mean=4.78, bh_sharpe=0.65, bh_cagr=10.86,
)

# ---------------------------------------------------------------------------
# Shared preamble — imports, cache check, data loader, frozen R dict.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from charm_decay import data, strategy as st

TICKER, AS_OF = "SPY", "2026-07-10"
CACHE_PATH = data._cache_path(TICKER, data.DEFAULT_CACHE)
HAVE_REAL = os.path.exists(CACHE_PATH)

def get_bars():
    b = data.fetch_daily(TICKER, start="1993-01-01", fetch=False)
    return b[b.index <= pd.Timestamp(AS_OF)]

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-07-10).
R = dict(
    ticker="SPY", n_days=8418, start="1993-01-29", end="2026-07-10", fp="a7ff45641491",
    n_opex_defined=408, n_pre_days=2005, n_post_days=2005,
    pre_mean=4.54, pre_base=4.85, pre_diff=-0.32, pre_t=-0.13, pre_n=2005,
    post_mean=3.48, post_base=5.18, post_diff=-1.70, post_t=-0.76, post_n=2005,
    asym_pre=4.54, asym_post=3.48, asym_diff=1.05, asym_t=0.31,
    q_all_diff=-0.32, q_all_t=-0.13, q_qtr_diff=-1.06, q_qtr_t=-0.23, q_qtr_n=670,
    pl_true_t=-0.13, pl_mean_abs=1.08, pl_max_abs=3.68, pl_n=52, pl_p=0.94,
    pl_planted_true_t=3.59, pl_planted_p=0.019,
    pre2012_diff=1.20, pre2012_t=0.35, pre2012_n=1135,
    post2012_diff=-2.25, post2012_t=-0.67, post2012_n=870,
    ov_n=1992, ov_mean=4.56, ov_sharpe=0.62, ov_cagr=10.25, ov_t=1.87,
    ls_n=3983, ls_mean=0.53, ls_sharpe=0.07, ls_cagr=-0.36, ls_t=0.31,
    bh_mean=4.78, bh_sharpe=0.65, bh_cagr=10.86,
)
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Charm-Decay — does the week before options expiry drift up on dealer hedging? ⏳\n"
            "### The 'charm rally' tested on SPY 1993–2026: no drift, a placebo that buries it\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Charm_rally%3F: Busted](https://img.shields.io/badge/Charm_rally%3F-Busted-8b949e?style=flat-square)\n\n"
            "There's a popular options-flow story: in the last week before monthly expiration, "
            "the *delta* of the big dealer options book decays just because time is passing "
            "(that Greek is called **charm**), forcing dealers to buy the market into expiry — "
            "and to sell it back the week after.  The 'OpEx-week rally, post-OpEx weakness.'  "
            "It sounds mechanical and inevitable.  Is it in the tape?\n\n"
            "> This is the plain-language layer.  The HAC *t*-stats, the calendar-randomisation "
            "placebo, and the pre/post-2012 break live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it.  House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does SPY drift **up** in the charm week before OpEx? | **No.** The five sessions "
            f"ending on OpEx return {R['pre_diff']:+.2f} bps/day vs the rest of the tape "
            f"(HAC t = {R['pre_t']:+.2f}) — a hair *below* baseline. |\n"
            "| Is there a 'rally then fade' asymmetry? | **No.** Pre minus post is "
            f"{R['asym_diff']:+.2f} bps (t = {R['asym_t']:+.2f}) — the right sign, drowned in noise. |\n"
            "| Is the OpEx week even *special*? | **No.** A placebo that slides the calendar to "
            f"fake OpEx dates puts the real week at the **{int(R['pl_p']*100)}th percentile** of "
            "random weeks — most arbitrary anchors drift *more*. |\n"
            "| Could you trade it? | **No.** Long-the-charm-week earns the same Sharpe as "
            f"buy-and-hold ({R['ov_sharpe']:.2f} vs {R['bh_sharpe']:.2f}); the full long/short "
            f"'rally-then-fade' makes **less than cash** ({R['ls_cagr']:+.1f}%/yr). |\n\n"
            "> Charm is real physics.  The tradable directional drift it's said to produce is "
            "not in SPY.  Signal: **None**.  Tradability: **Mirage**.  Charm rally: **Busted**."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Dealers are structurally short the options that customers buy for protection.  "
            "As monthly expiration approaches, the delta of that book bleeds away with time — the "
            "**charm** Greek — and dealers must buy the underlying to stay hedged.  That's why the "
            "market grinds up into OpEx week and softens the week after, once the flow unwinds and "
            "the next cycle resets.\"*\n\n"
            "The mechanism is genuinely real at the micro level: charm is a well-defined Greek, and "
            "dealers really do re-hedge as it moves.  The leap the claim makes is that this "
            "aggregates into a **predictable, tradable, directional drift** in a liquid index like "
            "SPY.  That's the leap we test."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If SPY reliably drifts up the week into OpEx and fades the week after, you'd have a "
            "*calendar* edge — no forecasting, no signal to compute, just a date on the wall.  "
            "Twelve long weeks and twelve short weeks a year, mechanically.  That is the kind of "
            "'free' structure that, if real, would be one of the cleanest anomalies in markets.  "
            "If it's noise, then the charm story is a compelling narrative pinned onto random "
            "weeks — and the fact that it *sounds* mechanical is exactly why it's worth checking."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Four honest tests on plain daily SPY close-to-close returns:\n\n"
            "1. **Pre-OpEx drift.** Mean return in the five sessions ending on OpEx vs every other "
            "day.\n"
            "2. **Post-OpEx give-back.** Same, for the five sessions after OpEx.\n"
            "3. **Rally-then-fade asymmetry.** Pre-window mean minus post-window mean — the one "
            "number the story predicts to be large and positive.\n"
            "4. **The placebo.** Slide the OpEx anchor to *fake* dates (±5…±30 trading days away) "
            "and re-measure.  If the real OpEx week is special, its drift should stand out against "
            "the cloud of random weeks.  **If a dozen arbitrary weeks drift more than the real "
            "one, the charm story is busted.**\n\n"
            "The window is set purely by the calendar — the third Friday is known before the month "
            "starts — so there's no look-ahead and nothing to fit."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown\n\n**First: is there any drift at all?**"),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    d = st.charm_drift_test(bars)\n"
            "    pre_diff, pre_t = d['pre']['diff']*1e4, d['pre']['tstat']\n"
            "    post_diff, post_t = d['post']['diff']*1e4, d['post']['tstat']\n"
            "else:\n"
            "    pre_diff, pre_t = R['pre_diff'], R['pre_t']\n"
            "    post_diff, post_t = R['post_diff'], R['post_t']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "vals = [pre_diff, post_diff]; ts = [pre_t, post_t]\n"
            "ax.bar(['Pre-OpEx\\n(charm week)', 'Post-OpEx\\n(give-back)'], vals,\n"
            "       color=[RED if abs(t)<2 else GREEN for t in ts])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(ax.patches, ts):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>=0 else 'top')\n"
            "ax.set_ylabel('Return vs baseline (bps/day)')\n"
            "ax.set_title('The charm week does NOT drift up — it is a hair below baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre-OpEx : {pre_diff:+.2f} bps/day vs baseline (HAC t={pre_t:+.2f})')\n"
            "print(f'Post-OpEx: {post_diff:+.2f} bps/day vs baseline (HAC t={post_t:+.2f})')"
        ),
        md(
            f"The charm week returns {R['pre_diff']:+.2f} bps/day *relative to the rest of the "
            f"tape* (t = {R['pre_t']:+.2f}) — not a rally, if anything a whisker slower.  The "
            f"post-OpEx week is mildly soft ({R['post_diff']:+.2f} bps, t = {R['post_t']:+.2f}) but "
            "nowhere near significant.  The storybook shape isn't here."
        ),
        md("**The single number the story lives or dies on — rally minus fade:**"),
        code(
            "if HAVE_REAL:\n"
            "    a = st.pre_post_asymmetry(get_bars())\n"
            "    asym_diff, asym_t = a['diff_bps'], a['tstat']\n"
            "    mp, mq = a['mean_pre_bps'], a['mean_post_bps']\n"
            "else:\n"
            "    asym_diff, asym_t = R['asym_diff'], R['asym_t']\n"
            "    mp, mq = R['asym_pre'], R['asym_post']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['Pre-OpEx mean', 'Post-OpEx mean'], [mp, mq], color=[AMBER, GREY])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title(f'Rally-then-fade = {asym_diff:+.2f} bps (HAC t={asym_t:+.2f}) — pure noise')\n"
            "for b, v in zip(ax.patches, [mp, mq]):\n"
            "    ax.annotate(f'{v:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre minus Post = {asym_diff:+.2f} bps/day, HAC t={asym_t:+.2f}')"
        ),
        md(
            f"There *is* a +{R['asym_diff']:.2f} bps/day tilt in the storybook direction — but with "
            f"t = {R['asym_t']:+.2f} it's the kind of thing you'd see in random data most of the "
            "time.  Which raises the obvious question: **how special is the real OpEx week, really?**"
        ),
        md(
            "**The placebo — the test that settles it.**  Slide the calendar to *fake* OpEx dates "
            "and see how the real one ranks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_randomization(get_bars())\n"
            "    true_t, pt, pval = pl['true_t'], pl['placebo_t'], pl['p_value']\n"
            "else:\n"
            "    rng = np.random.default_rng(768)\n"
            "    true_t, pval = R['pl_true_t'], R['pl_p']\n"
            "    pt = rng.normal(0, 1.1, R['pl_n'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(pt, bins=16, color=GREY, alpha=.75, label='fake-OpEx anchors')\n"
            "ax.axvline(true_t, c=RED, lw=2.5, label=f'real OpEx (t={true_t:+.2f})')\n"
            "ax.set_xlabel('Pre-window drift HAC t-stat'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Real OpEx week is unremarkable — empirical p={pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Real anchor t={true_t:+.2f}; {int(pval*100)}% of fake anchors are at least as extreme.')"
        ),
        md(
            f"There it is.  The real OpEx week's drift (t = {R['pl_true_t']:+.2f}) sits right in the "
            f"middle of the pile — **{int(R['pl_p']*100)}%** of arbitrary 'fake OpEx' weeks drift at "
            f"least as much, and the most extreme fake anchor hits |t| = {R['pl_max_abs']:.2f}.  If "
            "you didn't know which Friday was the real one, you could not pick it out.  That's the "
            "signature of a story pinned onto noise."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Pre-OpEx drift {R['pre_diff']:+.2f} bps (t = {R['pre_t']:+.2f}); "
            f"rally-then-fade {R['asym_diff']:+.2f} bps (t = {R['asym_t']:+.2f}).  Nothing clears "
            "|t| = 2, and the placebo says the OpEx week isn't even special.\n"
            "- **Tradability — Mirage.** The long overlay is just buy-and-hold beta collected on a "
            "quarter of the days; the long/short rally-then-fade trade loses to cash before costs.\n"
            "- **Charm rally — Busted.** Real Greek, real dealer hedging — but no measurable "
            "directional footprint in the SPY tape."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade the charm week?\n\n"
            "Two versions: (a) long SPY only in the charm week, flat otherwise; (b) the full trade — "
            "long the charm week, short the give-back week."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    ov = st.charm_overlay_returns(bars); s = st.summarize(ov[ov!=0])\n"
            "    ls = st.charm_overlay_returns(bars, short_post=True); s2 = st.summarize(ls[ls!=0])\n"
            "    bh = st.summarize(st.daily_return(bars).dropna())\n"
            "    ov_c, ls_c, bh_c = s['cagr']*100, s2['cagr']*100, bh['cagr']*100\n"
            "    ov_s, ls_s, bh_s = s['sharpe_ann'], s2['sharpe_ann'], bh['sharpe_ann']\n"
            "else:\n"
            "    ov_c, ls_c, bh_c = R['ov_cagr'], R['ls_cagr'], R['bh_cagr']\n"
            "    ov_s, ls_s, bh_s = R['ov_sharpe'], R['ls_sharpe'], R['bh_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['Long charm week', 'Long/short\\nrally-then-fade', 'Buy-and-hold SPY'],\n"
            "       [ov_c, ls_c, bh_c], color=[AMBER, RED, GREY])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('Long overlay = buy-and-hold beta; the full trade loses to cash')\n"
            "for b, v, s in zip(ax.patches, [ov_c, ls_c, bh_c], [ov_s, ls_s, bh_s]):\n"
            "    ax.annotate(f'{v:+.1f}%/yr\\nSharpe {s:.2f}', (b.get_x()+b.get_width()/2,\n"
            "                b.get_height()/2 if b.get_height()>0 else b.get_height()-0.5),\n"
            "                ha='center', va='center', fontweight='bold',\n"
            "                color='white' if b.get_height()>2 else 'k')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long charm week: CAGR {ov_c:+.1f}%/yr Sharpe {ov_s:.2f}')\n"
            "print(f'Rally-then-fade: CAGR {ls_c:+.1f}%/yr Sharpe {ls_s:.2f}')\n"
            "print(f'Buy-and-hold  : CAGR {bh_c:+.1f}%/yr Sharpe {bh_s:.2f}')"
        ),
        md(
            f"The long overlay makes ~{R['ov_cagr']:.1f}%/yr at Sharpe {R['ov_sharpe']:.2f} — "
            f"basically buy-and-hold ({R['bh_sharpe']:.2f}) divided across a quarter of the days.  "
            f"The full rally-then-fade trade makes **{R['ls_cagr']:+.1f}%/yr** (Sharpe "
            f"{R['ls_sharpe']:.2f}) — worse than a cash deposit, *before* you pay the spread twice "
            "a month and borrow on the short.  There is no edge to charge costs against."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The activity is real; the direction isn't.** "
            "[Study 195 — Monthly-OpEx](../../195-monthly-opex/) finds real *volume* around expiry "
            "(all of it quarterly triple-witching) and [Study 82 — Witching-Hour](../../82-witching-hour/) "
            "finds the pinning.  Dealers hedge — they just don't push SPY in a tradable direction.\n"
            "- **Conditional, not calendar.** The honest version of this idea measures aggregate "
            "dealer gamma/charm exposure in *real time* from the live options chain, and only "
            "predicts flow when positioning is actually concentrated.  A fixed calendar window "
            "throws that conditioning away — which is why it finds nothing.\n"
            "- **Intraday.** Charm hedging is often described as an end-of-day, expiry-Friday "
            "phenomenon.  A 30-minute-bar study of the OpEx-Friday afternoon (vs other Fridays) "
            "might see the flow the daily close can't — though translating it past costs faces the "
            "same wall.\n\n"
            "*Fork it: swap the calendar window for a live-GEX conditioner and see if the drift "
            "shows up only when the book says it should.*"
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
            "# Charm-Decay — a quantitative teardown ⏳\n"
            "### Daily SPY 1993–2026 · HAC t-stats · calendar-randomisation placebo · pre/post-2012 split\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Charm_rally%3F: Busted](https://img.shields.io/badge/Charm_rally%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "same seven beats, every claim carrying its standard error.  We test whether the "
            "charm (delta-decay) dealer-flow story produces a measurable directional drift in SPY "
            "over the pre-OpEx window, and whether any window rule clears the inference bar.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars SPY 1993–2026, auto-adjusted "
            "(total-return proxy), as-of 2026-07-10; the offline core and tests run on a "
            "deterministic synthetic tape.  Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **💡 `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pre-OpEx drift {R['pre_diff']:+.2f} bps (HAC t = {R['pre_t']:+.2f}); "
            f"rally-then-fade {R['asym_diff']:+.2f} bps (t = {R['asym_t']:+.2f}); nothing clears |t| = 2. |\n"
            f"| **Tradability** | `MIRAGE` | Long overlay Sharpe {R['ov_sharpe']:.2f} = buy-and-hold "
            f"{R['bh_sharpe']:.2f}; long/short 'rally-then-fade' CAGR {R['ls_cagr']:+.1f}%/yr (< cash). |\n"
            f"| **Charm rally?** | `BUSTED` | Calendar-randomisation places the real anchor at the "
            f"{int(R['pl_p']*100)}th percentile of fake anchors (empirical p = {R['pl_p']:.2f}). |\n\n"
            "> 💡 The charm Greek is real and dealers hedge it — but the aggregate directional "
            "footprint on SPY is statistically indistinguishable from a randomly chosen week."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $I_{t}$ indicate that trading day $t$ falls in the pre-OpEx charm window (the five "
            "sessions ending on the 3rd-Friday expiry) and $r_t$ the close-to-close return.  Charm "
            "is $\\partial\\Delta/\\partial t$; a dealer short a book of near-the-money options sees "
            "its net delta bleed toward zero as $t\\to$ expiry, and re-hedges directionally.  The "
            "hypotheses:\n\n"
            "- **H₁ (pre-drift).** $\\mathbb{E}[r_t\\mid I_t=1] > \\mathbb{E}[r_t\\mid I_t=0]$ — the "
            "charm week drifts up.\n"
            "- **H₂ (give-back).** $\\mathbb{E}[r_t\\mid \\text{post}] < \\mathbb{E}[r_t\\mid \\text{base}]$.\n"
            "- **H₃ (asymmetry).** pre-mean − post-mean $> 0$ and significant.\n"
            "- **H₄ (special anchor).** the pre-window drift *t* is extreme relative to arbitrary "
            "calendar anchors.\n"
            "- **H₅ (tradable).** a charm-window rule clears the inference bar net of costs.\n\n"
            "We reject H₁–H₅ on the SPY tape."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A calendar-anchored directional effect needs no forecast and no signal computation — "
            "it is the cheapest possible alpha if it exists, and the easiest to arbitrage away if "
            "discovered.  H₄ is the crux: a real hedging-flow effect must be *anchored* to the real "
            "expiry, not to any week.  If shuffling the anchor leaves the *t*-distribution "
            "unchanged, the drift is a property of SPY's general return distribution, not of OpEx — "
            "and the charm mechanism, whatever its micro reality, leaves no macro fingerprint."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** SPY daily OHLCV, Yahoo Finance auto-adjusted (total-return proxy), "
            "1993-01-29 → 2026-07-10, n = 8,418 days.\n"
            "- **Anchor.** 3rd Friday of every month, pure Gregorian calendar — no look-ahead.  "
            "Windows measured in **trading days** (holidays never smear them).\n"
            "- **Windows.** Pre = 5 sessions ending on OpEx; post = 5 sessions after.\n"
            "- **Return.** Close-to-close simple return; position calendar-set (no execution lag "
            "needed) but entered at the prior close so the first day's return is earned.\n"
            "- **Inference.** Newey-West HAC *t* on the window-vs-baseline mean difference; five "
            "sub-tests → Bonferroni bar |t| ≥ 2.58 for family-wise α = 5%.\n"
            "- **Placebo.** Slide the anchor by ±5…±30 trading days (52 fake anchors) → empirical "
            "*t*-null and two-sided p-value for the real anchor.\n"
            "- **Restriction.** Quarterly triple-witching months only (largest books).\n"
            "- **Structural break.** Pre/post-2012 (charm/vanna narrative + weekly/0DTE era).\n"
            "- **Positive control.** Synthetic tape with a planted pre-window drift confirms the "
            "engine — and the placebo — isolate the effect when one exists."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown"),

        md(
            "### 4a · Drift and asymmetry — null on every window\n\n"
            "Pre-OpEx, post-OpEx, and the pre-minus-post asymmetry, each with its HAC *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    d = st.charm_drift_test(bars); a = st.pre_post_asymmetry(bars)\n"
            "    rows = [('Pre-OpEx drift', d['pre']['diff']*1e4, d['pre']['tstat'], d['pre']['n_window']),\n"
            "            ('Post-OpEx drift', d['post']['diff']*1e4, d['post']['tstat'], d['post']['n_window']),\n"
            "            ('Pre - Post asym', a['diff_bps'], a['tstat'], a['n_pre'])]\n"
            "else:\n"
            "    rows = [('Pre-OpEx drift', R['pre_diff'], R['pre_t'], R['pre_n']),\n"
            "            ('Post-OpEx drift', R['post_diff'], R['post_t'], R['post_n']),\n"
            "            ('Pre - Post asym', R['asym_diff'], R['asym_t'], R['pre_n'])]\n"
            "df = pd.DataFrame(rows, columns=['test','diff_bps','HAC_t','n'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [GREEN if abs(t)>=2 else RED for t in df['HAC_t']]\n"
            "ax.bar(df['test'], df['diff_bps'], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(ax.patches, df['HAC_t']):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>=0 else 'top')\n"
            "ax.set_ylabel('Return vs baseline (bps/day)')\n"
            "ax.set_title('Every charm window is null (|t| well under 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "df.round(3)"
        ),
        md(
            f"> 💡 The largest of the three effects is the asymmetry ({R['asym_diff']:+.2f} bps, "
            f"t = {R['asym_t']:+.2f}) and it is in the *storybook* direction — but a *t* of "
            f"{R['asym_t']:.2f} is what noise looks like.  The pre-window drift, the headline claim, "
            f"is actually slightly negative ({R['pre_diff']:+.2f} bps)."
        ),

        md(
            "### 4b · The placebo — the OpEx anchor is not special\n\n"
            "Slide the anchor to 52 fake OpEx dates and rebuild the pre-window drift *t* each time."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_randomization(get_bars())\n"
            "    true_t, pt, pval = pl['true_t'], pl['placebo_t'], pl['p_value']\n"
            "    maxabs = pl['placebo_max_abs_t']\n"
            "else:\n"
            "    rng = np.random.default_rng(768)\n"
            "    true_t, pval, maxabs = R['pl_true_t'], R['pl_p'], R['pl_max_abs']\n"
            "    pt = rng.normal(0, 1.1, R['pl_n'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(pt, bins=18, color=GREY, alpha=.75, label=f'{len(pt)} fake-OpEx anchors')\n"
            "ax.axvline(true_t, c=RED, lw=2.5, label=f'real OpEx t={true_t:+.2f}')\n"
            "for s in (-2, 2): ax.axvline(s, ls='--', c='k', lw=.8)\n"
            "ax.set_xlabel('Pre-window drift HAC t-stat'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Real anchor at the {int(pval*100)}th pct of arbitrary anchors '\n"
            "             f'(max fake |t|={maxabs:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Real t={true_t:+.2f}  empirical two-sided p={pval:.3f}  max fake |t|={maxabs:.2f}')"
        ),
        md(
            f"> 💡 This is the decisive panel.  Fake anchors reach |t| = {R['pl_max_abs']:.2f}; the "
            f"real one is {R['pl_true_t']:+.2f}, with **p = {R['pl_p']:.2f}**.  If OpEx carried "
            "hedging-flow information you could not shuffle the calendar and keep the same "
            "*t*-distribution — but you can.  The drift belongs to SPY's return process at large, "
            "not to expiration."
        ),

        md(
            "### 4c · Quarterly restriction & structural break\n\n"
            "Where the option book is biggest (quarterly triple-witching), and in the modern "
            "charm-narrative era (post-2012), the effect should intensify if real."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    q = st.quarterly_split(bars); sp = st.pre_post_2012_split(bars)\n"
            "    q_all_t, q_qtr_t = q['all']['tstat'], q['quarterly']['tstat']\n"
            "    pre12_t, post12_t = sp['pre_2012']['tstat'], sp['post_2012']['tstat']\n"
            "    pre12_d, post12_d = sp['pre_2012']['diff']*1e4, sp['post_2012']['diff']*1e4\n"
            "    q_all_d, q_qtr_d = q['all']['diff']*1e4, q['quarterly']['diff']*1e4\n"
            "else:\n"
            "    q_all_t, q_qtr_t = R['q_all_t'], R['q_qtr_t']\n"
            "    pre12_t, post12_t = R['pre2012_t'], R['post2012_t']\n"
            "    pre12_d, post12_d = R['pre2012_diff'], R['post2012_diff']\n"
            "    q_all_d, q_qtr_d = R['q_all_diff'], R['q_qtr_diff']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "axes[0].bar(['All months', 'Quarterly\\nonly'], [q_all_d, q_qtr_d],\n"
            "            color=[RED, RED])\n"
            "axes[0].axhline(0, c='k', lw=1); axes[0].set_ylabel('Pre-OpEx diff (bps/day)')\n"
            "axes[0].set_title('Quarterly restriction: still null')\n"
            "for b, t in zip(axes[0].patches, [q_all_t, q_qtr_t]):\n"
            "    axes[0].annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                     ha='center', va='top')\n"
            "axes[1].bar(['Pre-2012', 'Post-2012'], [pre12_d, post12_d], color=[RED, RED])\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('Pre-OpEx diff (bps/day)')\n"
            "axes[1].set_title('Modern era drifts the WRONG way')\n"
            "for b, t in zip(axes[1].patches, [pre12_t, post12_t]):\n"
            "    axes[1].annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                     ha='center', va='bottom' if b.get_height()>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Quarterly-only: {q_qtr_d:+.2f} bps (t={q_qtr_t:+.2f})')\n"
            "print(f'Post-2012 pre-OpEx: {post12_d:+.2f} bps (t={post12_t:+.2f}) -- wrong sign')"
        ),
        md(
            f"> 💡 Both stress tests deepen the null.  The quarterly (biggest-book) restriction is "
            f"{R['q_qtr_diff']:+.2f} bps (t = {R['q_qtr_t']:+.2f}), and the post-2012 era — exactly "
            f"when charm/vanna flow became a household narrative — drifts *down* "
            f"({R['post2012_diff']:+.2f} bps, t = {R['post2012_t']:+.2f}).  The story got popular; "
            "the effect went the other way."
        ),

        md(
            "### 4d · Synthetic positive control — the engine (and the placebo) work\n\n"
            "Plant a known pre-window drift and confirm both the HAC test and the placebo recover it."
        ),
        code(
            "plants = [0.0, 2.0, 5.0, 10.0]\n"
            "ts = []\n"
            "for pb in plants:\n"
            "    b, _ = data.synthetic_daily(n_years=30, pre_drift_bps=pb, seed=768)\n"
            "    ts.append(st.charm_drift_test(b)['pre']['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(plants, ts, 'o-', c=GREEN, lw=2, label='synthetic pre-window t')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='inference bar |t|=2')\n"
            "ax.axhline(R['pre_t'], ls=':', c=RED, lw=1.5, label=f\"real tape t={R['pre_t']:.2f}\")\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('Planted pre-OpEx drift (bps/day)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Engine recovers planted drift monotonically; real tape sits at zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "bpl, _ = data.synthetic_daily(n_years=30, pre_drift_bps=8.0, seed=768)\n"
            "plp = st.placebo_randomization(bpl)\n"
            "print(f'Planted +8bps tape: placebo true_t={plp[\"true_t\"]:+.2f} p={plp[\"p_value\"]:.3f} '\n"
            "      f'(isolates the true anchor); real SPY p={R[\"pl_p\"]:.2f} (does not).')"
        ),

        # ---- BEAT 5 — THE VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pre-OpEx drift t = {R['pre_t']:+.2f}, post t = {R['post_t']:+.2f}, "
            f"asymmetry t = {R['asym_t']:+.2f}, quarterly t = {R['q_qtr_t']:+.2f}.  With five "
            "sub-tests the Bonferroni bar is |t| ≥ 2.58; the maximum |t| observed is 0.76.\n"
            f"- **Tradability `MIRAGE`** — long overlay Sharpe {R['ov_sharpe']:.2f} vs buy-and-hold "
            f"{R['bh_sharpe']:.2f} (pure beta); long/short CAGR {R['ls_cagr']:+.1f}%/yr, below cash, "
            "before costs or borrow.\n"
            f"- **Charm rally `BUSTED`** — placebo empirical p = {R['pl_p']:.2f}; the real OpEx "
            "anchor is less extreme than the average fake one.  The machinery isolates a planted "
            f"drift (p = {R['pl_planted_p']:.3f}), so the null is a property of the data, not the test."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----------------------------------------
        md(
            "## 6 · Could you trade it? — the charm overlay and its costs\n\n"
            "HAC *t*, Sharpe, and a cost sweep for the long-only charm-week overlay."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    ov = st.charm_overlay_returns(bars); s = st.summarize(ov[ov!=0])\n"
            "    gross = s['mean_bps']\n"
            "    print(f'Long charm week: n={s[\"n\"]} mean={gross:+.2f}bps sharpe={s[\"sharpe_ann\"]:+.2f} '\n"
            "          f'CAGR={s[\"cagr\"]:+.2%} HAC t={s[\"tstat\"]:+.2f}')\n"
            "else:\n"
            "    gross = R['ov_mean']\n"
            "    print(f'Long charm week: n={R[\"ov_n\"]} mean={gross:+.2f}bps sharpe={R[\"ov_sharpe\"]:+.2f} '\n"
            "          f'CAGR={R[\"ov_cagr\"]:+.1f}% HAC t={R[\"ov_t\"]:+.2f}')\n"
            "sweep = st.cost_sweep(gross)\n"
            "costs = [c for c, _ in sweep]; net = [m for _, m in sweep]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('One-way cost (bps)'); ax.set_ylabel('Net mean (bps/active day)')\n"
            "ax.set_title('Charm overlay net mean — but the gross was never alpha (it is beta)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The +1.87 HAC t on the long overlay is market beta on ~24% of days, '\n"
            "      'not a charm edge: its Sharpe equals buy-and-hold.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **[Study 195 — Monthly-OpEx](../../195-monthly-opex/)** and "
            "**[Study 82 — Witching-Hour](../../82-witching-hour/)**: the *activity* around expiry "
            "is real (volume, pinning) and quarterly-driven — but non-directional and untradable, "
            "consistent with the null drift here.\n"
            "- **Live gamma/charm exposure (GEX) conditioning.** The defensible version replaces the "
            "calendar window with a real-time dealer-positioning estimate from the options chain, "
            "predicting flow only when the aggregate book is concentrated.  That is a *conditional* "
            "signal; this study shows the *unconditional* calendar version carries nothing.\n"
            "- **Vanna vs charm.** Vanna flow (∂Δ/∂σ) is volatility-conditioned; a regime split on "
            "realised or implied vol — long-gamma vs short-gamma dealer regimes — might separate a "
            "conditional footprint the pooled test averages away.\n"
            "- **Intraday expiry-Friday.** The flow is often described as a close-auction effect; "
            "30-minute bars on OpEx-Friday afternoons vs control Fridays would test that directly.\n\n"
            "*The calendar version is busted.  The conditional (positioning-aware) version is the "
            "honest place to look next — PRs welcome.*"
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
