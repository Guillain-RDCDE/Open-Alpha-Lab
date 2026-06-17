"""Generate (and execute) the two narrative notebooks for Study 286 (Valentines-Day).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded Valentine session table and the synthetic generator run anywhere, offline
and deterministically; the real ^GSPC cells use the repo-level cache
(``_cache/^GSPC_split_only.parquet``) if present, and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md) and fall back to synthetic
data, so the notebook re-runs for any reader.

``build_notebooks.py`` writes each notebook then executes it in place with nbconvert
(no ``--allow-errors``); every code cell ends up with a real execution_count and no
error output.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=76,
    val_mean_pct=0.0388,
    val_std_pct=0.9613,
    uncond_mean_pct=0.0364,
    uncond_std_pct=0.9931,
    excess_mean_bps=0.24,
    up_rate_pct=47.37,
    uncond_up_rate_pct=53.14,
    t_vs_zero=0.352,
    t_vs_uncond=0.022,
    p_vs_uncond=0.9825,
    hac_t_excess=0.021,
    perm_p=0.981,
    gross_mean_per_event_bps=3.88,
    net_mean_per_event_bps=1.88,
    early_mean_pct=0.0125,
    early_up_pct=36.8,
    late_mean_pct=0.0652,
    late_up_pct=57.9,
    fp_event="23a6d0b41333",
    fp_uncond="ca0a9a85212a",
    year_start=1950,
    year_end=2025,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from valentines_day import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc_daily(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""

# R-dict preamble for notebooks (so cells can reference R['key'])
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=76,
    val_mean_pct=0.0388, val_std_pct=0.9613,
    uncond_mean_pct=0.0364, uncond_std_pct=0.9931,
    excess_mean_bps=0.24,
    up_rate_pct=47.37, uncond_up_rate_pct=53.14,
    t_vs_zero=0.352, t_vs_uncond=0.022, p_vs_uncond=0.9825,
    hac_t_excess=0.021, perm_p=0.981,
    gross_mean_per_event_bps=3.88, net_mean_per_event_bps=1.88,
    early_mean_pct=0.0125, early_up_pct=36.8,
    late_mean_pct=0.0652, late_up_pct=57.9,
    year_start=1950, year_end=2025,
)

# Build the real event frame + baseline if the cache is present, else synthetic null.
if HAVE_REAL:
    EV = data.build_event_frame(fetch=False)
    UNC = data.unconditional_daily(fetch=False)
    val_ret = EV['ret']
else:
    _df, _ = data.synthetic_daily(start_year=R['year_start'], end_year=R['year_end'],
                                  premium_bps=0.0, seed=286)
    EV = _df.loc[_df['is_valentine'], ['ret']].copy()
    EV.index = list(range(R['year_start'], R['year_start'] + len(EV)))
    UNC = _df['ret']
    val_ret = EV['ret']
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Valentines-Day -- does the market love Valentine's Day?\n"
            "### A cherished bit of calendar folklore, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every February you'll hear it: the market is *in a good mood* on Valentine's "
            "Day, so stocks pop on Feb 14. It's a charming idea -- chocolates, roses, "
            "and a green tape. This notebook asks the only question that matters: does the "
            "S&P 500 actually do anything special on Valentine's Day, or is Feb 14 just "
            "another ordinary trading session wearing a heart-shaped hat?\n\n"
            "> **This is the plain-language layer.** Want the Newey-West t-stat, the "
            "permutation distribution, and the power calculation? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the S&P pop on Valentine's Day? | **No.** Mean Valentine return "
            "**+0.039%/day** -- a rounding error above the **+0.036%** the market drifts "
            "up on *any* day. |\n"
            "| Is the day even up more often than usual? | **No -- the opposite.** Valentine "
            "sessions are up just **47.4%** of the time, *below* the **53.1%** base rate. |\n"
            "| Is the tiny edge real? | **No.** Excess = +0.24 bps, HAC t = 0.02, "
            "permutation p = 0.98. Dead-centre noise. |\n"
            "| Could you trade it? | **No.** The ~4 bps gross 'edge' is noise, and one-way "
            "costs halve it. Buy-and-hold of that same day wins. |\n\n"
            "> The market does not love Valentine's Day. There *is* a faint positive drift "
            "on Feb 14 -- but it's the market's everyday drift, present on every session."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The market is in a good mood on Valentine's Day, so the S&P 500 tends to "
            "rise on Feb 14.\"*\n\n"
            "There's no academic paper behind this one -- it's pure calendar folklore, a "
            "cousin of the Super Bowl Indicator and 'sell in May'. The mechanism is, to put "
            "it gently, vibes: investor mood on a romantic holiday is supposed to leak into "
            "the closing print. Markets would be very strange indeed if a greeting-card "
            "holiday moved the index. Let's look."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a single calendar date reliably delivered a positive return, it would be the "
            "easiest trade in finance: buy the close on Feb 13, sell the close on Feb 14, "
            "once a year, forever. The fact that nobody does this is a hint. The fun question "
            "is *why* such a date might briefly look special -- and the answer is a clean "
            "lesson in the difference between 'positive' and 'positive beyond the baseline'."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong null.** The S&P drifts *up* about +0.036%/day on average. So a "
            "'predictor' that says 'Valentine's Day is positive' gets the market's everyday "
            "drift for free. The correct question is whether Feb 14 beats **+0.036%**, not 0.\n"
            "2. **Tiny effect, ~1% daily noise.** One trading day has a ~1% standard "
            "deviation. With 76 Valentine sessions, the standard error of the mean is "
            "~0.11%/day -- so a 'Valentine premium' would have to be enormous (~0.23%/day) "
            "to even register.\n"
            "3. **The weekday confound.** Feb 14 (and its roll-forward when it's a weekend) "
            "lands on different weekdays across the years, tangling any 'effect' with the "
            "old Monday-vs-Friday seasonality.\n\n"
            "We test on the ^GSPC daily tape, 1950-2025 -- 76 Valentine sessions, each one "
            "hardcoded in this study's `data.py`."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the first ^GSPC session on or after Feb 14 each year "
            "(Feb 14 often falls on a weekend), paired with that session's return."
        ),
        code(
            "vt = data.valentine_table()\n"
            "print('Valentine sessions:', len(vt), 'years (1950-2025)')\n"
            "if HAVE_REAL:\n"
            "    val_mean = float(val_ret.mean()*100)\n"
            "    unc_mean = float(UNC.mean()*100)\n"
            "    up_rate = float((val_ret>0).mean()*100)\n"
            "    unc_up = float((UNC>0).mean()*100)\n"
            "else:\n"
            "    val_mean, unc_mean = R['val_mean_pct'], R['uncond_mean_pct']\n"
            "    up_rate, unc_up = R['up_rate_pct'], R['uncond_up_rate_pct']\n"
            "print(f'Valentine mean return:      {val_mean:+.4f}%/day')\n"
            "print(f'Unconditional daily mean:   {unc_mean:+.4f}%/day  <- the honest baseline')\n"
            "print(f'Valentine up-rate:          {up_rate:.1f}%  (vs {unc_up:.1f}% on an average day)')\n"
            "print()\n"
            "print('The critical number is +0.036%/day. Any positive day gets this for free.')"
        ),
        md("**The 'edge' that isn't -- Valentine return vs the everyday drift.**"),
        code(
            "if HAVE_REAL:\n"
            "    vm = float(val_ret.mean()*100); um = float(UNC.mean()*100)\n"
            "else:\n"
            "    vm, um = R['val_mean_pct'], R['uncond_mean_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Valentine\\nsession', 'Average\\nday'], [vm, um],\n"
            "              color=[RED, GREY], width=0.5)\n"
            "ax.axhline(um, ls='--', c=AMBER, lw=2, label=f'Everyday drift ({um:+.4f}%)')\n"
            "ax.set_ylabel('Mean daily return (%)')\n"
            "ax.set_title('Valentine\\'s Day: +0.24 bps above the everyday drift -- pure noise')\n"
            "for b, v in zip(bars, [vm, um]):\n"
            "    ax.annotate(f'{v:+.4f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Excess over the everyday drift: {(vm-um)*100:+.2f} bps/day -- a rounding error.')"
        ),
        md(
            "The Valentine session averages **+0.039%/day**; an average day averages "
            "**+0.036%/day**. The 'Valentine edge' is **+0.24 bps** -- about 0.7% of a "
            "single day's standard deviation. The market's everyday drift does all the work.\n\n"
            "> Being *positive* is not the same as being *special*. Almost every day is "
            "faintly positive on average."
        ),
        md("**The up-rate has the sign backwards.**"),
        code(
            "if HAVE_REAL:\n"
            "    ur = float((val_ret>0).mean()*100); uur = float((UNC>0).mean()*100)\n"
            "else:\n"
            "    ur, uur = R['up_rate_pct'], R['uncond_up_rate_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Valentine sessions\\n(% up)', 'Average day\\n(% up)'], [ur, uur],\n"
            "              color=[RED, GREY], width=0.5)\n"
            "ax.axhline(50, ls=':', c='k', lw=1, label='50% (coin)')\n"
            "ax.set_ylabel('Up-rate (%)'); ax.set_ylim(0, 70)\n"
            "ax.set_title('Valentine sessions are up LESS often than an average day')\n"
            "for b, v in zip(bars, [ur, uur]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+0.5), ha='center', va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Valentine up-rate {ur:.1f}% vs {uur:.1f}% unconditional -- gap {ur-uur:+.1f}pp (wrong way).')"
        ),
        md(
            "If anything, the folklore has it backwards: Valentine sessions close **up** "
            "only **47.4%** of the time, *below* the **53.1%** you'd get on an average day. "
            "The tiny positive *mean* comes from a few big up-days, not from the day being "
            "reliably green."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Excess +0.24 bps/day, HAC t = **0.02**, permutation "
            "p = **0.98**. The Valentine session is statistically identical to an average "
            "day -- and is actually *down* more often than usual.\n"
            "- **Tradability -- Mirage.** The only 'trade' is to hold the S&P for one "
            "pre-chosen February day. The ~4 bps gross edge is noise and one-way costs "
            "halve it; buy-and-hold of that day is strictly better.\n"
            "- **Busted.** The faint positive drift on Feb 14 is the market's everyday "
            "drift. There is no Valentine's Day magic."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: buy the close on Feb 13, sell the close on Feb 14, "
            "once a year. Here's the gross 'edge' and what's left after a modest 1 bp "
            "one-way cost (2 bps round-trip) charged on NAV."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tradability(val_ret, cost_bps_oneway=1.0)\n"
            "    gross_bps = tr['gross_mean_per_event']*1e4\n"
            "    net_bps = tr['net_mean_per_event']*1e4\n"
            "else:\n"
            "    gross_bps, net_bps = R['gross_mean_per_event_bps'], R['net_mean_per_event_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "bars = ax.bar(['Gross\\nper event', 'Net\\n(after 2 bps cost)'], [gross_bps, net_bps],\n"
            "              color=[GREY, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean return per event (bps)')\n"
            "ax.set_title('Costs eat half of an already-vanishing \\'edge\\'')\n"
            "for b, v in zip(bars, [gross_bps, net_bps]):\n"
            "    ax.annotate(f'{v:+.2f} bps', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross: {gross_bps:+.2f} bps/event | Net: {net_bps:+.2f} bps/event')\n"
            "print('A few basis points of noise, halved by costs. There is no trade here.')"
        ),
        md(
            "Four basis points gross, ~two net -- both inside the noise band of a single "
            "trading day. There is nothing to harvest: holding the S&P for that same day "
            "(buy-and-hold) earns the gross return without the round-trip cost."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a Valentine premium "
            "into a synthetic tape and shows the machinery detects it cleanly above ~10 "
            "bps/day. The real tape has +0.24 bps -- nowhere close.\n"
            "- **The Super Bowl Indicator** -- the same base-rate trap, the same tiny-n "
            "reckoning: [Study 158 -- Super-Bowl](../../158-super-bowl/).\n"
            "- **The honest event-study template** this study mirrors: "
            "[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/).\n\n"
            "*Think Cupid really moves the tape? Fork this, pick your own date convention, "
            "and show a Valentine excess return that clears |t| >= 2 against the everyday "
            "drift. That's the bar -- and it hasn't been cleared.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Valentines-Day -- a quantitative teardown\n"
            "### 76 Feb-14 sessions * ^GSPC daily * one-sample t * Newey-West HAC t * "
            "permutation * n=76 power * synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test the "
            "Valentine session return against the correct null (the S&P's everyday daily "
            "drift), run a one-sample t-test, a Newey-West HAC t on the excess return, a "
            "10,000-draw permutation test, split the sample in half, and close with a power "
            "calculation and a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily close-to-close, price-only "
            "(1950-2025); Valentine session dates hardcoded in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Signal-axis caveat:** price-only index, and the ^GSPC level is vendor "
            "back-revised -- this can only flatter a non-result, not rescue it."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Excess mean **+0.24 bps/day**; one-sample t vs the "
            "everyday drift = **+0.02**; Newey-West HAC t = **+0.02**; permutation "
            "p = **0.98**. Up-rate 47% < 53% base rate. |\n"
            "| **Tradability** | `MIRAGE` | No vehicle; ~4 bps gross noise, halved by "
            "one-way costs; dominated by buy-and-hold of the same day. |\n"
            "| **Busted?** | `YES` | The faint Feb-14 positivity is the market's everyday "
            "+3.6 bps drift -- present on every session. |\n\n"
            "> The unconditional daily drift does all the work. The Valentine label "
            "contributes nothing statistically distinguishable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC close-to-close return on the Valentine session of year "
            "$t$, and $\\mu$ the unconditional daily mean over the same window. The "
            "hypotheses:\n\n"
            "- **H1 (excess return).** $E[r_t] > \\mu$ -- the Valentine session beats the "
            "*everyday drift* (not zero). Testing against 0 instead of $\\mu$ is the "
            "base-rate error that flatters every 'X-day effect'.\n"
            "- **H2 (hit-rate).** $P(r_t > 0) > P(\\text{any day} > 0)$ -- Feb 14 is up "
            "more often than the base rate.\n\n"
            "We reject H1 (excess +0.24 bps, t = 0.02) and find H2 points the *wrong way* "
            "(47% < 53%)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "A single-day calendar effect, if real, would be a textbook free lunch: one "
            "pre-announced date, a one-day hold, repeatable annually. It is also the "
            "archetypal data-snooping mirage -- there are 365 dates, and the prettiest one "
            "will always show a few basis points of 'edge' ex-post. The interesting finding "
            "is how *ordinary* Feb 14 is once you test it against the right baseline."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily close-to-close returns, price-only, 1950-2025; "
            "Valentine session = first trading day on or after Feb 14 (hardcoded in "
            "`data.py`, 76 sessions).\n"
            "- **Statistic.** Mean of the 76 Valentine returns vs the unconditional daily "
            "mean $\\mu$ (the excess return).\n"
            "- **Correct null.** One-sample t-test against $\\mu$, *not* 0; plus a "
            "permutation null of random 76-session draws.\n"
            "- **Inference.** `ttest_1samp` (vs 0 and vs $\\mu$); Newey-West HAC t "
            "(Bartlett, lag 4) on the excess; permutation test (10,000 draws).\n"
            "- **Positive control.** Synthetic daily tape with a planted Valentine premium "
            "to confirm the machinery fires when an effect exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The correct baseline: excess return and HAC t\n\n"
            "The critical methodological point: the S&P has a positive daily drift. Testing "
            "the Valentine return against 0 -- as folklore implicitly does -- is wrong."
        ),
        code(
            "es = st.event_stats(val_ret, UNC, n_permutations=10_000, seed=286)\n"
            "print(f\"n Valentine sessions: {es['n']}\")\n"
            "print(f\"Valentine mean:       {es['val_mean']*100:+.4f}%/day\")\n"
            "print(f\"Unconditional mean:   {es['uncond_mean']*100:+.4f}%/day  (the honest baseline)\")\n"
            "print(f\"Excess mean:          {es['excess_mean']*1e4:+.2f} bps/day\")\n"
            "print()\n"
            "print(f\"One-sample t vs 0:            {es['t_vs_zero']:+.3f}  (p={es['p_vs_zero']:.4f})\")\n"
            "print(f\"One-sample t vs everyday mu:  {es['t_vs_uncond']:+.3f}  (p={es['p_vs_uncond']:.4f})\")\n"
            "print(f\"Newey-West HAC t (excess):    {es['hac_t_excess']:+.3f}\")\n"
            "print()\n"
            "print('Even against the LENIENT null (0), t is only ~0.35.')\n"
            "print('Against the CORRECT null (the everyday drift), t collapses to ~0.02.')"
        ),
        md(
            "> Against zero, the Valentine return looks faintly positive (t ~ 0.35). Against "
            "the **correct** null -- the market's +3.6 bps everyday drift -- the excess is "
            "+0.24 bps with t = **0.02**. The whole 'effect' was the everyday drift in "
            "disguise."
        ),
        md(
            "### 4b - The permutation test: is the Valentine mean distinguishable from noise?\n\n"
            "Draw 76 random sessions from the same window 10,000 times and record the "
            "distribution of their means."
        ),
        code(
            "perm_means = es['perm_means'] * 100\n"
            "obs = float(val_ret.mean()*100)\n"
            "mu = float(UNC.mean()*100)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_means, bins=40, color=GREY, alpha=0.7, label='Random 76-session means')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Valentine mean: {obs:+.4f}%')\n"
            "ax.axvline(mu, c=AMBER, lw=2, ls='--', label=f'Everyday drift: {mu:+.4f}%')\n"
            "ax.set_xlabel('Mean daily return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The Valentine mean lands dead-centre in the null distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Permutation p (two-sided): {es['perm_p']:.4f}\")\n"
            "print(f\"Percentile of Valentine mean: {(perm_means < obs).mean():.1%} of draws below\")"
        ),
        md(
            "The Valentine mean sits essentially on top of the everyday drift, in the heart "
            "of the null. Permutation p = **0.98** -- there is no evidence the Feb-14 label "
            "carries any information about the day's return."
        ),
        md(
            "### 4c - Sub-periods: does the 'effect' even agree with itself?\n\n"
            "Split the 76 sessions into two halves."
        ),
        code(
            "rows = []\n"
            "for a, b in [(R['year_start'], 1987), (1988, R['year_end'])]:\n"
            "    sub = EV[(EV.index >= a) & (EV.index <= b)]['ret']\n"
            "    rows.append({'window': f'{a}-{b}', 'n': len(sub),\n"
            "                 'mean_%': round(float(sub.mean()*100), 4),\n"
            "                 'up_%': round(float((sub>0).mean()*100), 1)})\n"
            "sp = pd.DataFrame(rows)\n"
            "print(sp.to_string(index=False))\n"
            "print()\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar(sp['window'], sp['mean_%'], color=[GREY, RED], width=0.5)\n"
            "ax.axhline(R['uncond_mean_pct'], ls='--', c=AMBER, lw=2,\n"
            "           label=f\"Everyday drift ({R['uncond_mean_pct']:+.4f}%)\")\n"
            "ax.set_ylabel('Valentine mean return (%)')\n"
            "ax.set_title('The two halves disagree -- the signature of noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('One half below the baseline, the other above -- no stable premium.')"
        ),
        md(
            "The 1950-1987 half is **below** the everyday drift (and up only ~37%); the "
            "1988-2025 half is above it (up ~58%). Two 38-observation halves pointing in "
            "opposite directions is exactly what sampling noise looks like."
        ),
        md(
            "### 4d - Power calculation: what premium could n=76 detect?\n\n"
            "The minimum detectable mean at 80% power, two-sided, alpha=0.05."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vol = float(val_ret.std(ddof=1))\n"
            "else:\n"
            "    vol = R['val_std_pct']/100\n"
            "n = R['n']\n"
            "from scipy.stats import norm\n"
            "z_a = norm.ppf(1 - 0.05/2); z_b = norm.ppf(0.80)\n"
            "se = vol / np.sqrt(n)\n"
            "mde = (z_a + z_b) * se\n"
            "print(f'Daily vol: {vol*100:.3f}%/day, n = {n}')\n"
            "print(f'Standard error of the Valentine mean: {se*100:.4f}%/day ({se*1e4:.1f} bps)')\n"
            "print(f'Minimum detectable mean (80% power, 2-sided): {mde*100:.3f}%/day ({mde*1e4:.1f} bps)')\n"
            "print()\n"
            "print(f\"Observed excess: {R['excess_mean_bps']:+.2f} bps -- about \"\n"
            "      f\"{R['excess_mean_bps']/(mde*1e4):.1%} of the detectable threshold.\")\n"
            "print('Even a genuine Valentine premium below ~23 bps/day is invisible at n=76.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- excess +0.24 bps/day, t vs everyday drift +0.02, "
            "HAC t +0.02, perm p 0.98. The up-rate (47%) is *below* the 53% base rate. "
            "n=76 cannot resolve anything smaller than ~23 bps/day.\n"
            "- **Tradability `MIRAGE`** -- no vehicle beyond a one-day S&P hold; ~4 bps "
            "gross noise, halved by one-way costs, dominated by buy-and-hold.\n"
            "- **Busted** -- Feb 14's faint positivity is the market's everyday drift; "
            "there is no Valentine's Day magic.\n\n"
            "*Survivorship note:* the index level is back-revised by the vendor and the "
            "series is price-only -- both can only flatter, never rescue, a non-result."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- gross vs net\n\n"
            "The rule: buy the close before Feb 14, sell the Valentine close. One round-trip "
            "per year, 1 bp one-way (2 bps round-trip) charged on NAV."
        ),
        code(
            "tr = st.tradability(val_ret, cost_bps_oneway=1.0)\n"
            "print(f\"Events:               {tr['n_events']}\")\n"
            "print(f\"Gross mean per event: {tr['gross_mean_per_event']*1e4:+.2f} bps\")\n"
            "print(f\"Cost drag (round-trip): {tr['cost_drag_per_event']*1e4:.1f} bps\")\n"
            "print(f\"Net mean per event:   {tr['net_mean_per_event']*1e4:+.2f} bps\")\n"
            "print()\n"
            "print('Cumulative growth over the sample:')\n"
            "print(f\"  Gross: x{tr['gross_cum_growth']:.4f}  | Net: x{tr['net_cum_growth']:.4f}\")\n"
            "print()\n"
            "print('The gross event return IS the buy-and-hold day -- the rule only adds cost.')"
        ),
        md(
            "> One-way costs erase roughly half of a gross 'edge' that was already inside "
            "one day's noise band. There is no tradable Valentine's Day effect."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a Valentine premium *when one exists*? We plant a "
            "premium into a synthetic daily tape and sweep its size."
        ),
        code(
            "planted = [0, 5, 10, 25, 50]\n"
            "results = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_daily(start_year=1900, end_year=2299,\n"
            "                                   premium_bps=float(bps), seed=286)\n"
            "    ev_s = df_s.loc[df_s['is_valentine'], 'ret']\n"
            "    r = st.event_stats(ev_s, df_s['ret'], n_permutations=1000, seed=286)\n"
            "    results.append({'planted_bps': bps, 'excess_bps': round(r['excess_mean']*1e4, 2),\n"
            "                    'hac_t': r['hac_t_excess'], 'perm_p': r['perm_p']})\n"
            "res = pd.DataFrame(results)\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "colors = [GREEN if (row['hac_t'] > 2 and row['perm_p'] < 0.05) else RED\n"
            "          for _, row in res.iterrows()]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(res['planted_bps'].astype(str), res['hac_t'], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2 hurdle')\n"
            "ax.set_xlabel('Planted Valentine premium (bps/day)')\n"
            "ax.set_ylabel('Newey-West HAC t (excess)')\n"
            "ax.set_title('The engine finds the signal when it exists (green = HAC t>2 and p<0.05)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "The engine cleanly detects a planted premium from ~10 bps/day upward (HAC t > "
            "2, perm p < 0.05) and correctly finds nothing at premium = 0. The real tape's "
            "+0.24 bps is nowhere near the threshold. The verdict is a statement about the "
            "**market and the everyday drift**, not about the method."
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


def _execute(name):
    """Execute a notebook in place with nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    nb = nbf.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
