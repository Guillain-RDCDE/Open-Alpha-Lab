"""Generate the two narrative notebooks for Study 322 (FOMC-Blackout).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached SPY
total-return parquet under ../../../_cache/ if present and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader, online or off.
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
    fp="c35148b53fca", asof="2026-06-12", n_days=8166, n_blk=1741, n_oth=6425,
    # Headline blackout vs other
    blk_mean=5.00, blk_t=2.02, blk_sharpe=0.66,
    oth_mean=4.73, oth_t=3.72, all_mean=4.79, all_t=4.26,
    diff_bps=0.27, welch_t=0.08,
    # Calm check
    blk_vol=1922, oth_vol=1866, vol_ratio=1.030,
    # Era split
    pre_diff=2.67, pre_t=0.57, post_diff=-2.35, post_t=-0.53,
    # Timing book
    book_gross=2.69, book_net5=1.87, bh_ann=12.07, trips=16.3,
    book_sharpe=0.30, bh_sharpe=0.64, pct_in_market=21,
    # Synthetic control
    syn_excess_diff=14.84, syn_excess_t=3.44, syn_calm_ratio=0.48,
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

from fomc_blackout import data, strategy as st

def load_real_safe():
    \"\"\"Cache-first real tape; None if offline with no cache.\"\"\"
    try:
        return data.load_real(fetch=False)
    except Exception as e:
        print("no real cache (", type(e).__name__, ") -> using frozen numbers / synthetic tape")
        return None

REAL = load_real_safe()
HAVE_REAL = REAL is not None
print("real SPY total-return cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# FOMC-Blackout — is the Fed's quiet period a 'calm before the storm' trade? 🤫\n"
            "### The ~10 silent days before each rate decision, tested in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Calm_before_the_storm%3F: Busted](https://img.shields.io/badge/Calm_before_the_storm%3F-Busted-8b949e?style=flat-square)\n\n"
            "Before every Fed rate decision there's a **blackout** — roughly ten days when Fed "
            "officials go silent and may not talk about policy in public. A piece of trading "
            "folklore says this quiet stretch is a *calm before the storm*: the market drifts up "
            "steadily while it waits, saving the fireworks for the announcement. This notebook asks "
            "the only two things that matter — does the blackout window actually pay *more* than the "
            "rest of the year, and is it really any calmer?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the placebo and the "
            "Sharpe race? That's the companion, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** "
            "— same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the blackout window pay more? | **No.** Blackout days earn "
            f"**+{R['blk_mean']:.2f} bps/day** — but the rest of the year earns +{R['oth_mean']:.2f}. "
            f"The *difference* is **+{R['diff_bps']:.2f} bps** (a coin-flip *t* of +{R['welch_t']:.2f}). |\n"
            "| Is it a 'calm before the storm'? | **No — it's slightly *stormier*.** Volatility in "
            f"the window is marginally **higher** (ratio {R['vol_ratio']:.2f}), not lower. |\n"
            "| Could you trade it? | **No.** Holding the index only during blackouts earns "
            f"**~+{R['book_gross']:.1f}%/yr** vs **+{R['bh_ann']:.1f}%/yr** for just buying and "
            "holding — you give up most of the year's return for nothing extra. |\n"
            "| So what is it? | The blackout window earns **ordinary equity drift** and nothing more. "
            "The 'calm before the storm' is being long the market part-time. |\n\n"
            "> The one real cousin — a genuine jump in the **24 hours** right before the meeting "
            "(Lucca & Moench 2015) — is a *single day*, not ten. Spread across the whole quiet "
            "period it disappears."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Fed goes quiet for about ten days before each meeting — the blackout. With no "
            "new Fed-speak to spook anyone, the market drifts calmly upward into the decision. Buy "
            "the calm, sit out the storm.\"*\n\n"
            "The kernel of truth: the blackout is a real Fed rule. From the *second Saturday before* "
            "a meeting through the *Thursday after*, officials don't comment on policy in public. The "
            "leap of faith is that this *communications* silence shows up as a *market* pattern — a "
            "steady, tradable drift with unusually low volatility."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were true, it would be a gift: a calendar window known *months* in advance "
            "(the FOMC schedule is public), no forecasting required, low volatility, positive drift. "
            "You'd hold the index during blackouts and sit in cash otherwise. And there's a real, "
            "famous effect nearby — the **pre-FOMC announcement drift** (Lucca & Moench 2015), a big "
            "jump in the 24 hours before the decision. The question is whether that one-day spark "
            "spreads into a ten-day glow you can actually sit in — or whether stretching it across "
            "the whole window just averages it away to nothing."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks:\n\n"
            "1. **Compare like with like.** Don't ask \"is the blackout positive?\" (everything is, "
            "in a rising market). Ask \"does it beat the *rest of the year*?\" — the **difference** "
            "is the edge.\n"
            "2. **Check the calm.** Measure the window's volatility against everything else. If "
            "there's no lull, the whole story collapses.\n"
            "3. **Slide the window (placebo).** Move the 'blackout' off the meeting by a few weeks "
            "and re-measure. A real effect peaks on the true window; a fake one looks the same "
            "everywhere.\n\n"
            "Tape: SPY daily total return since 1994, every day flagged from the public FOMC schedule."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: blackout days vs the rest of the year.** Same chart, both means side by side:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.blackout_table(REAL)\n"
            "    blk, oth = tbl['blackout']['mean_bps'], tbl['other']['mean_bps']\n"
            "    diff, wt = tbl['diff_bps'], tbl['welch_t']\n"
            "else:\n"
            f"    blk, oth, diff, wt = {R['blk_mean']}, {R['oth_mean']}, {R['diff_bps']}, {R['welch_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Blackout window\\n(~10 days pre-meeting)', 'Rest of the year'],\n"
            "           [blk, oth], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean daily total return (bps)')\n"
            "ax.set_title('Blackout days earn the same drift as every other day')\n"
            "for bar_ in b:\n"
            "    ax.annotate(f'{bar_.get_height():+.2f}', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()+0.1), ha='center', va='bottom', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Difference (blackout - rest): {diff:+.2f} bps   Welch t = {wt:+.2f}')"
        ),
        md(
            f"There it is. Blackout days earn **+{R['blk_mean']:.2f} bps/day**, the rest of the year "
            f"**+{R['oth_mean']:.2f}**. The gap — the thing that would be an *edge* — is "
            f"**+{R['diff_bps']:.2f} bps**, with a *t* of +{R['welch_t']:.2f}. That is statistical "
            "zero. The blackout window is just *being long the market*."
        ),
        md(
            "**Second: is it actually calmer?** The whole 'calm before the storm' idea needs lower "
            "volatility in the window. Let's look:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vt = st.vol_table(REAL); vb, vo = vt['blackout_vol_bps'], vt['other_vol_bps']\n"
            "else:\n"
            f"    vb, vo = {R['blk_vol']}, {R['oth_vol']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Blackout window', 'Rest of the year'], [vb, vo],\n"
            "           color=[RED, GREY], width=.5)\n"
            "ax.set_ylabel('annualised volatility (bps)')\n"
            "ax.set_title(f'No lull: blackout vol is slightly HIGHER (ratio {vb/vo:.2f})')\n"
            "for bar_ in b:\n"
            "    ax.annotate(f'{bar_.get_height():.0f}', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()+15), ha='center', va='bottom', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'vol ratio (blackout / rest) = {vb/vo:.3f}  -> not calmer')"
        ),
        md(
            f"The window is **not** calmer — volatility is marginally *higher* (ratio "
            f"{R['vol_ratio']:.2f}). The Fed being quiet doesn't make the *market* quiet. The "
            "'storm' isn't saved for the announcement; it's spread out like any other fortnight."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Blackout minus rest-of-year = **+{R['diff_bps']:.2f} bps/day** "
            f"(*t* = +{R['welch_t']:.2f}). No excess drift.\n"
            f"- **Tradability — Mirage.** A blackout-only timing book earns ~+{R['book_gross']:.1f}%/yr "
            f"vs +{R['bh_ann']:.1f}%/yr buy-and-hold — a slice of beta, not an edge.\n"
            f"- **Calm before the storm? — Busted.** Window volatility is *higher* (ratio "
            f"{R['vol_ratio']:.2f}), not lower."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Suppose you tried anyway: hold SPY only during blackouts, cash the rest of the time. "
            "Here's what that part-time bet earns against just buying and holding:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.cost_sweep(REAL); book = cs['book_gross_ann_pct'].iloc[0]\n"
            "    bh = cs['buyhold_ann_pct'].iloc[0]\n"
            "else:\n"
            f"    book, bh = {R['book_gross']}, {R['bh_ann']}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "b = ax.bar(['Blackout-only\\ntiming book', 'Just buy & hold\\nSPY'], [book, bh],\n"
            "           color=[RED, GREEN], width=.5)\n"
            "ax.set_ylabel('annualised return (%/yr)'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_title('You give up most of the year for nothing extra')\n"
            "for bar_ in b:\n"
            "    ax.annotate(f'{bar_.get_height():+.1f}%', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()+0.2), ha='center', va='bottom', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'In the market only ~{R['pct_in_market']}% of the time -> ~+{{book:.1f}}%/yr vs +{{bh:.1f}}%/yr')"
        ),
        md(
            f"The blackout book is in the market only ~{R['pct_in_market']}% of the time, so it "
            f"captures only the drift that happens to fall there: **~+{R['book_gross']:.1f}%/yr** vs "
            f"**+{R['bh_ann']:.1f}%/yr** for doing nothing but holding. There's no edge to harvest — "
            "only a fraction of the beta you already had. Costs don't even get a chance to kill it; "
            "it was never alive."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The real cousin.** The **pre-FOMC announcement drift** (Lucca & Moench 2015) is a "
            "genuine, documented jump in the **24 hours** before the decision. It lives *inside* this "
            "window — but it's one day, not ten, and averaging it across the blackout washes it out. "
            "Fork this and isolate that single day.\n"
            "- **The other FOMC calendar study.** [Study 135 — FOMC-Cycle](../../135-fomc-cycle/) "
            "tests the Cieslak even/odd-week drift *after* meetings — a different window, a different "
            "(weak, decayed) story.\n"
            "- **Other quiet periods.** Corporate-earnings blackouts (the buyback-restriction window) "
            "are a separate folklore with a clearer mechanism — worth a study of its own.\n\n"
            "*Think the calm is hiding in a tighter window? Fork this, shrink it to the last day or "
            "two, and show the drift surviving once you compare it to the rest of the year.*"
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
            "# FOMC-Blackout — a quantitative teardown 🔬\n"
            "### Real total-return tape · HAC/Welch inference · calendar placebo · "
            "decay split · excess-Sharpe race · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Calm_before_the_storm%3F: Busted](https://img.shields.io/badge/Calm_before_the_storm%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the ~10-day "
            "pre-meeting FOMC communications blackout carries an excess return (and lower volatility) "
            "over the rest of the year, and whether any of it is tradable.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY daily auto-adjusted (total-return proxy), "
            "1994-2026; as-of 2026-06-12; the offline core and tests run on a deterministic synthetic "
            "tape. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition — so "
            "this notebook still reads even if you skim the maths."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | blackout − other = **+{R['diff_bps']:.2f} bps/day**, "
            f"Welch *t* = **+{R['welch_t']:.2f}**; placebo flat; pre/post-2011 sign flips "
            f"(+{R['pre_diff']:.2f} → {R['post_diff']:.2f} bps). |\n"
            f"| **Tradability** | `MIRAGE` | blackout-only book +{R['book_gross']:.1f}%/yr vs "
            f"+{R['bh_ann']:.1f}%/yr B&H; excess Sharpe {R['book_sharpe']:.2f} vs {R['bh_sharpe']:.2f} "
            "— a fraction of beta, before any cost. |\n"
            f"| **Calm before the storm?** | `BUSTED` | window vol ratio **{R['vol_ratio']:.2f}** "
            "(higher, not lower). |\n\n"
            "> 💡 In plain words: the blackout window earns plain equity drift, is no calmer than any "
            "other fortnight, and as a timing rule just throws away ~79% of the year's return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the daily total return and $B_t\\in\\{0,1\\}$ the pre-meeting blackout flag "
            "($B_t=1$ iff $t$ is within 10 calendar days *before* a scheduled FOMC decision):\n\n"
            "- **H₁ (excess drift).** $\\mathbb{E}[r_t\\mid B_t{=}1] > \\mathbb{E}[r_t\\mid B_t{=}0]$ "
            "— the *difference*, not the level.\n"
            "- **H₂ (calm).** $\\mathrm{Var}[r_t\\mid B_t{=}1] < \\mathrm{Var}[r_t\\mid B_t{=}0]$.\n"
            "- **H₃ (localised).** The diff peaks at the true window and fades as it slides off the "
            "meeting (placebo).\n"
            "- **H₄ (tradable).** A blackout-only timing book beats buy-and-hold on excess Sharpe.\n\n"
            "We find **H₁–H₄ all rejected.** The level of the blackout arm is positive only because "
            "it inherits the market's drift (H₀: it's beta)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The nearby *real* effect is Lucca & Moench's (2015) **pre-FOMC announcement drift**: a "
            "~49 bps excess in the **24 hours** before scheduled meetings, a meaningful chunk of the "
            "equity premium. The folklore quietly upgrades that one-day spark into a ten-day, "
            "low-vol, sit-in-it window. If that upgrade held, it would be an unusually clean "
            "calendar trade. The content of this study is *precisely where the upgrade fails*: "
            "spreading a one-day effect over ten days averages it to noise, and the 'calm' half of "
            "the story was never in the data at all."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Flag.** $B_t = 1$ for trading days within 10 calendar days before each scheduled "
            "FOMC date (the public schedule — **no execution lag needed**, the label is "
            "calendar-known months ahead).\n"
            "- **Inference.** Per-arm Newey-West HAC *t*; Welch *t* on the blackout-minus-other "
            "difference (the decisive test).\n"
            "- **Localisation.** Calendar-shift placebo: slide the window by ±5/10/20 days and "
            "re-measure.\n"
            "- **Decay.** Pre/post-2011 split (after Lucca-Moench; McLean-Pontiff decay).\n"
            "- **Calm.** Annualised vol, blackout vs rest.\n"
            "- **Tradability.** Long-only-in-blackout book, costs one-way × NAV, raced against B&H "
            "on **excess-of-cash** Sharpe.\n"
            "- **Positive control.** Deterministic synthetic tape with a tunable planted excess and "
            "vol multiplier.\n\n"
            "Tape: SPY daily total return, 1994-2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Blackout vs rest — the level is beta, the diff is zero\n\n"
            "Per-arm mean and HAC *t*, plus the Welch *t* on the difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.blackout_table(REAL)\n"
            "    arms = ['blackout','other','all']\n"
            "    means = [tbl[a]['mean_bps'] for a in arms]; ts = [tbl[a]['tstat'] for a in arms]\n"
            "    diff, wt = tbl['diff_bps'], tbl['welch_t']\n"
            "else:\n"
            f"    means = [{R['blk_mean']}, {R['oth_mean']}, {R['all_mean']}]\n"
            f"    ts = [{R['blk_t']}, {R['oth_t']}, {R['all_t']}]; diff, wt = {R['diff_bps']}, {R['welch_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "labels = ['Blackout','Other','All']\n"
            "a1.bar(labels, means, color=[AMBER, GREY, GREEN], width=.6)\n"
            "a1.set_ylabel('mean (bps/day)'); a1.set_title('Means: blackout ~ other ~ all')\n"
            "a1.axhline(0, c='k', lw=1)\n"
            "a2.bar(labels, ts, color=[AMBER, GREY, GREEN], width=.6)\n"
            "a2.axhline(2, ls='--', c=GREY); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_ylabel('per-arm HAC t'); a2.set_title('Each arm is just long the market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'DIFF blackout-other = {diff:+.2f} bps   Welch t = {wt:+.2f}  (statistical zero)')"
        ),
        md(
            f"> 💡 In plain words: all three arms sit around +{R['all_mean']:.1f} bps/day — they are "
            "the *same* drift. The blackout arm's lone HAC *t* of "
            f"+{R['blk_t']:.2f} is just \"the market went up,\" and the diff that would be an edge is "
            f"+{R['diff_bps']:.2f} bps at *t* = +{R['welch_t']:.2f}. **H₁ rejected.**"
        ),
        md(
            "### 4b · The calendar-shift placebo — no peak at the true window\n\n"
            "Slide the 10-day window by a calendar offset and re-measure the diff. A real effect "
            "peaks at offset 0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_shift(REAL, data.FOMC_DATES)\n"
            "    offs = list(pl.index); diffs = list(pl['diff_bps'])\n"
            "else:\n"
            "    offs = [-20,-10,-5,0,5,10,20]\n"
            "    diffs = [3.67, 0.01, -0.09, 0.27, -0.33, 0.34, 1.58]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [RED if o == 0 else GREY for o in offs]\n"
            "ax.bar([str(o) for o in offs], diffs, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('window offset (calendar days; 0 = true blackout)')\n"
            "ax.set_ylabel('blackout - other diff (bps)')\n"
            "ax.set_title('The true window (red) is not a peak — flat noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('A real localised effect would tower at offset 0. It does not.')"
        ),
        md(
            "> 💡 In plain words: the true window's diff is *smaller* than several arbitrary shifts. "
            "Whatever wiggles there are, they aren't tied to the meeting. **H₃ rejected.**"
        ),
        md(
            "### 4c · Calm? — and did anything decay?\n\n"
            "Volatility inside vs outside the window, and the pre/post-2011 split."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vt = st.vol_table(REAL); vb, vo = vt['blackout_vol_bps'], vt['other_vol_bps']\n"
            "    era = st.split_by_era(REAL); pre, post = era['pre'], era['post']\n"
            "    pre_d, pre_t = pre['diff_bps'], pre['welch_t']\n"
            "    post_d, post_t = post['diff_bps'], post['welch_t']\n"
            "else:\n"
            f"    vb, vo = {R['blk_vol']}, {R['oth_vol']}\n"
            f"    pre_d, pre_t, post_d, post_t = {R['pre_diff']}, {R['pre_t']}, {R['post_diff']}, {R['post_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['Blackout','Other'], [vb, vo], color=[RED, GREY], width=.5)\n"
            "a1.set_ylabel('annualised vol (bps)'); a1.set_title(f'Not calmer (ratio {vb/vo:.2f})')\n"
            "a2.bar(['Pre-2011','Post-2011'], [pre_d, post_d], color=[GREY, RED], width=.5)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('diff (bps)')\n"
            "a2.set_title(f'Sign flips: t {pre_t:+.2f} -> {post_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'vol ratio {vb/vo:.3f} | pre diff {pre_d:+.2f} (t{pre_t:+.2f})  post {post_d:+.2f} (t{post_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the window is marginally *stormier* (ratio {R['vol_ratio']:.2f}) — "
            f"**H₂ rejected.** And the diff flips from +{R['pre_diff']:.2f} bps pre-2011 to "
            f"{R['post_diff']:.2f} bps after, neither significant: there was never a stable effect to "
            "decay."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — blackout − other = +{R['diff_bps']:.2f} bps/day, Welch "
            f"*t* = +{R['welch_t']:.2f}; placebo flat; pre/post-2011 sign flip. The blackout level "
            f"(+{R['blk_mean']:.2f} bps, HAC *t* +{R['blk_t']:.2f}) is pure equity beta.\n"
            f"- **Tradability `MIRAGE`** — blackout-only book +{R['book_gross']:.1f}%/yr vs "
            f"+{R['bh_ann']:.1f}%/yr B&H; excess Sharpe {R['book_sharpe']:.2f} < {R['bh_sharpe']:.2f}. "
            "Worse than buy-and-hold before a cent of cost.\n"
            f"- **Calm before the storm? `BUSTED`** — vol ratio {R['vol_ratio']:.2f} (higher)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the excess-Sharpe race\n\n"
            "Hold the index on blackout days, cash otherwise (~16 one-way trips/yr). Race it against "
            "buy-and-hold on an **excess-of-cash** basis, and sweep costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.cost_sweep(REAL); es = st.excess_sharpe(REAL)\n"
            "    costs = list(cs.index); net = list(cs['book_net_ann_pct'])\n"
            "    bh = cs['buyhold_ann_pct'].iloc[0]\n"
            "    bsh, bhsh = es['book_excess_sharpe'], es['buyhold_excess_sharpe']\n"
            "else:\n"
            f"    costs = [0.0,1.0,2.0,5.0]; net = [{R['book_gross']}, 2.53, 2.36, {R['book_net5']}]\n"
            f"    bh = {R['bh_ann']}; bsh, bhsh = {R['book_sharpe']}, {R['bh_sharpe']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(costs, net, 'o-', c=RED, lw=2, label='blackout book (net)')\n"
            "a1.axhline(bh, ls='--', c=GREEN, lw=2, label=f'buy & hold ({bh:.1f}%/yr)')\n"
            "a1.set_xlabel('round-trip cost (bps)'); a1.set_ylabel('annualised %/yr')\n"
            "a1.set_ylim(0, bh*1.1); a1.legend(); a1.set_title('Book never gets near buy-and-hold')\n"
            "a2.bar(['Blackout\\nbook','Buy & hold'], [bsh, bhsh], color=[RED, GREEN], width=.5)\n"
            "a2.set_ylabel('excess-of-cash Sharpe'); a2.set_title('Worse risk-adjusted, even gross')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'book excess Sharpe {bsh:.3f}  vs  buy&hold {bhsh:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the book is dominated by buy-and-hold on *both* return and "
            f"risk-adjusted return (excess Sharpe {R['book_sharpe']:.2f} vs {R['bh_sharpe']:.2f}), "
            "and that's *before* costs. There is no alpha to charge against — only a part-time slice "
            "of the beta you already owned. Textbook **Mirage**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or does it always print zero? Plant a known "
            "blackout excess (and a known 'calm') and confirm it recovers them."
        ),
        code(
            "rows = []\n"
            "for lab, exc, vm in [('Null (no effect)', 0.0, 1.0),\n"
            "                     ('+15 bps excess', 0.0015, 1.0),\n"
            "                     ('Calm (vol x0.5)', 0.0, 0.5)]:\n"
            "    f, _ = data.synthetic_blackout(blackout_excess=exc, blackout_vol_mult=vm, seed=322)\n"
            "    t = st.blackout_table(f); v = st.vol_table(f)\n"
            "    rows.append((lab, t['diff_bps'], t['welch_t'], v['ratio']))\n"
            "dfc = pd.DataFrame(rows, columns=['planted','diff_bps','welch_t','vol_ratio'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(dfc['planted'], dfc['welch_t'], color=[GREY, GREEN, GREY], width=.55)\n"
            "a1.axhline(2, ls='--', c=GREY); a1.set_ylabel('Welch t on diff')\n"
            "a1.set_title('Engine banks a planted excess'); a1.tick_params(axis='x', labelsize=8)\n"
            "a2.bar(dfc['planted'], dfc['vol_ratio'], color=[GREY, GREY, GREEN], width=.55)\n"
            "a2.axhline(1, ls='--', c='k'); a2.set_ylabel('vol ratio')\n"
            "a2.set_title('...and a planted calm'); a2.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfc.round(3).to_string(index=False))"
        ),
        md(
            f"The harness recovers a planted +15 bps excess (Welch *t* ≈ +{R['syn_excess_t']:.1f}) and "
            f"a planted calm (vol ratio ≈ {R['syn_calm_ratio']:.2f}), and scores ~0 on the null. So "
            "its *silence* on the real tape is a statement about the **market**, not a broken "
            "pipeline: the FOMC blackout window carries no excess drift and no lull.\n\n"
            "The one real effect nearby — the **24-hour** pre-FOMC drift (Lucca & Moench 2015) — "
            "lives inside this window but is a single day; spread over ten it averages out. For the "
            "*post*-meeting calendar, see [Study 135 — FOMC-Cycle](../../135-fomc-cycle/)."
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
