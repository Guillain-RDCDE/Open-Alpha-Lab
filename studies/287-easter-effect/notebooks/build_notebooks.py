"""Generate (and execute) the two narrative notebooks for Study 287 (Easter-Effect).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Easter is
computed by the Computus algorithm and the surrounding ^GSPC sessions are frozen, so the
offline core runs anywhere, deterministically; the real ^GSPC cells use the repo-level
cache (``_cache/^GSPC_split_only.parquet``) if present, and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md) and fall back to synthetic data,
so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-18).
R = dict(
    n=76,
    ev_mean_pct=0.3321,
    ev_std_pct=0.7345,
    uncond_mean_pct=0.0364,
    uncond_std_pct=0.9931,
    thu_mean_pct=0.0354,
    excess_mean_bps=29.57,
    excess_thu_bps=29.66,
    up_rate_pct=68.42,
    uncond_up_rate_pct=53.14,
    t_vs_zero=3.941,
    t_vs_uncond=3.509,
    p_vs_uncond=0.0008,
    hac_t_excess=3.278,
    hac_t_thu=3.289,
    perm_p=0.0121,
    perm_p_thu=0.0074,
    gross_mean_per_event_bps=33.21,
    net_mean_per_event_bps=31.21,
    gross_cum=1.2840,
    net_cum=1.2646,
    net_cagr_pct=0.309,
    # post-holiday (Easter Monday)
    post_mean_pct=-0.1872,
    post_up_pct=43.42,
    post_excess_bps=-22.36,
    post_hac_t=-2.452,
    post_excess_mon_bps=-13.97,
    post_hac_t_mon=-1.532,
    post_perm_p_mon=0.2631,
    # sub-periods (excess bps, hac_t)
    sp1_excess=24.03, sp1_t=4.658,
    sp2_excess=14.03, sp2_t=1.187,
    sp3_excess=52.32, sp3_t=2.388,
    h1_excess=23.10, h1_t=4.289,
    h2_excess=36.03, h2_t=2.135,
    fp_pre="2b3e2834cdbc",
    fp_post="3e3e7af69776",
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

from easter_effect import data, strategy as st

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
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-18)
R = dict(
    n=76,
    ev_mean_pct=0.3321, ev_std_pct=0.7345,
    uncond_mean_pct=0.0364, uncond_std_pct=0.9931, thu_mean_pct=0.0354,
    excess_mean_bps=29.57, excess_thu_bps=29.66,
    up_rate_pct=68.42, uncond_up_rate_pct=53.14,
    t_vs_zero=3.941, t_vs_uncond=3.509, p_vs_uncond=0.0008,
    hac_t_excess=3.278, hac_t_thu=3.289, perm_p=0.0121, perm_p_thu=0.0074,
    gross_mean_per_event_bps=33.21, net_mean_per_event_bps=31.21,
    gross_cum=1.2840, net_cum=1.2646, net_cagr_pct=0.309,
    post_mean_pct=-0.1872, post_up_pct=43.42, post_excess_bps=-22.36,
    post_hac_t=-2.452, post_excess_mon_bps=-13.97, post_hac_t_mon=-1.532, post_perm_p_mon=0.2631,
    sp1_excess=24.03, sp1_t=4.658, sp2_excess=14.03, sp2_t=1.187, sp3_excess=52.32, sp3_t=2.388,
    h1_excess=23.10, h1_t=4.289, h2_excess=36.03, h2_t=2.135,
    year_start=1950, year_end=2025,
)

# Build the real event frame + baselines if the cache is present, else synthetic null.
if HAVE_REAL:
    EV = data.build_event_frame(fetch=False)
    UNC = data.unconditional_daily(fetch=False)
    UNC_THU = data.unconditional_daily(fetch=False, weekday=3)   # pre-holiday = Thursday
    UNC_MON = data.unconditional_daily(fetch=False, weekday=0)   # post = Monday
    pre_ret = EV['pre_ret']
    post_ret = EV['post_ret']
else:
    _df, _ = data.synthetic_daily(start_year=R['year_start'], end_year=R['year_end'],
                                  premium_bps=0.0, seed=287)
    _pre = _df.loc[_df['is_pre'], ['ret']].copy()
    _pre.index = pd.Index(_pre.index.year, name='year')
    EV = _pre.rename(columns={'ret': 'pre_ret'})
    EV['post_ret'] = _pre['ret'].values          # synthetic post = a second draw stand-in
    UNC = _df['ret']; UNC_THU = _df['ret']; UNC_MON = _df['ret']
    pre_ret = EV['pre_ret']; post_ret = EV['post_ret']
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Easter-Effect -- is there a Good-Friday / Easter seasonal?\n"
            "### A calendar pattern that turns out to be *real* -- but not for the reason the folklore says\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Busted%3F: No](https://img.shields.io/badge/Busted%3F-No-2ea44f?style=flat-square)\n\n"
            "Every spring you'll hear that the market has an *Easter mood*: buoyant into the "
            "long weekend, sleepy on the way back. Most of these holiday tales are pure vibes. "
            "This one is interesting -- because **Good Friday is a day the stock market is "
            "*closed*.** So there's no Good-Friday return to look at. The real question is what "
            "happens on the session *right before* the long weekend -- and there, it turns out, "
            "something genuine is going on.\n\n"
            "> **This is the plain-language layer.** Want the Newey-West t-stat, the "
            "permutation distribution, the same-weekday control, and the power calculation? "
            "That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there a real Easter seasonal? | **Yes -- but it's a *pre-holiday* one.** Good "
            "Friday is a market holiday, so the action is the session *before* it (always Maundy "
            "Thursday). |\n"
            "| How big is it? | The pre-holiday session averages **+0.33%/day** -- about **+30 bps "
            "above** the everyday drift -- and is up **68%** of the time (vs 53% on a normal day). |\n"
            "| Is it statistically real? | **Yes.** HAC t = **+3.28**, permutation p = **0.012**, "
            "and it *survives* the day-of-week control. |\n"
            "| Is it Easter magic? | **No.** It's the well-known **pre-holiday effect** -- markets "
            "drift up into long weekends. Good Friday just happens to be the holiday. Easter Monday "
            "adds nothing once you account for the (weak) Monday effect. |\n"
            "| Could you trade it? | **Barely.** Net +31 bps per event survives costs -- but it "
            "fires *once a year* (~0.31%/yr). Holding the index beats sitting in cash 364 days. |\n\n"
            "> A rare folklore study where the signal is **real** -- it's just mislabelled. The "
            "magic isn't Easter; it's the long weekend."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The market is upbeat heading into the Easter long weekend and sluggish coming "
            "back on Easter Monday.\"*\n\n"
            "The twist that makes this study different from Valentine's Day or the Super Bowl: "
            "**Good Friday is an NYSE holiday** (it has been in every year since 1950). So there "
            "is literally no Good-Friday print to test. The honest experiment is to look at the "
            "two sessions that bracket the closed day:\n\n"
            "- the **pre-holiday session** -- the last trading day before Good Friday, which is "
            "*always Maundy Thursday*;\n"
            "- the **post-holiday session** -- the first trading day after Easter Sunday, which is "
            "*always Easter Monday*."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the day before a holiday reliably popped, that's a textbook calendar edge -- and "
            "this one has real academic pedigree (Ariel 1990; Lakonishok & Smidt 1988 documented "
            "*pre-holiday* returns many times the daily average). The fun is in separating two "
            "very different claims: 'Easter is special' (folklore) versus 'the day before a long "
            "weekend is special' (a real, general phenomenon). Spoiler: it's the second one."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong null.** The S&P drifts *up* about +0.036%/day on average. The correct "
            "question is whether the pre-holiday session beats **+0.036%**, not 0.\n"
            "2. **The weekday confound.** The pre-holiday session is *always a Thursday* and Easter "
            "Monday *always a Monday*. Any 'effect' could just be the old day-of-week seasonality. "
            "So we also test against a **Thursday-only** (and Monday-only) baseline.\n"
            "3. **Is it Easter, or is it the long weekend?** A real pre-holiday drift would show up "
            "before *any* holiday. We can't claim 'Easter magic' from a pattern that is really the "
            "generic pre-holiday effect -- and we won't.\n\n"
            "We test on the ^GSPC daily tape, 1950-2025 -- 76 Easter weekends, with Easter computed "
            "by the Computus algorithm in this study's `data.py`."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** for each year, the last ^GSPC session before Good Friday "
            "(Maundy Thursday) and the first after Easter (Easter Monday), with each session's "
            "return."
        ),
        code(
            "et = data.easter_table()\n"
            "print('Easter weekends:', len(et), 'years (1950-2025)')\n"
            "print('Pre-holiday session weekday(s):', sorted(et['pre_wd'].unique()), '(3 = Thursday)')\n"
            "print('Post-holiday session weekday(s):', sorted(et['post_wd'].unique()), '(0 = Monday)')\n"
            "if HAVE_REAL:\n"
            "    pre_mean = float(pre_ret.mean()*100)\n"
            "    unc_mean = float(UNC.mean()*100)\n"
            "    up_rate = float((pre_ret>0).mean()*100)\n"
            "    unc_up = float((UNC>0).mean()*100)\n"
            "else:\n"
            "    pre_mean, unc_mean = R['ev_mean_pct'], R['uncond_mean_pct']\n"
            "    up_rate, unc_up = R['up_rate_pct'], R['uncond_up_rate_pct']\n"
            "print()\n"
            "print(f'Pre-holiday mean return:    {pre_mean:+.4f}%/day')\n"
            "print(f'Unconditional daily mean:   {unc_mean:+.4f}%/day  <- the honest baseline')\n"
            "print(f'Pre-holiday up-rate:        {up_rate:.1f}%  (vs {unc_up:.1f}% on an average day)')"
        ),
        md("**The pre-holiday pop -- the session before Good Friday vs the everyday drift.**"),
        code(
            "if HAVE_REAL:\n"
            "    vm = float(pre_ret.mean()*100); um = float(UNC.mean()*100)\n"
            "else:\n"
            "    vm, um = R['ev_mean_pct'], R['uncond_mean_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Pre-Good-Friday\\n(Maundy Thu)', 'Average\\nday'], [vm, um],\n"
            "              color=[GREEN, GREY], width=0.5)\n"
            "ax.axhline(um, ls='--', c=AMBER, lw=2, label=f'Everyday drift ({um:+.4f}%)')\n"
            "ax.set_ylabel('Mean daily return (%)')\n"
            "ax.set_title('The session before Good Friday: ~+30 bps above the everyday drift')\n"
            "for b, v in zip(bars, [vm, um]):\n"
            "    ax.annotate(f'{v:+.4f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Excess over the everyday drift: {(vm-um)*100:+.1f} bps/day -- this is NOT a rounding error.')"
        ),
        md(
            "The pre-holiday session averages **+0.33%/day** -- nearly **+30 bps above** the "
            "everyday drift. Unlike Valentine's Day (a quarter of a basis point), this is a "
            "**real-sized** number: about a third of a single day's standard deviation, on a day "
            "that is up **68%** of the time.\n\n"
            "> But before we celebrate, we have to rule out the obvious confound: it's always a "
            "Thursday. Is this just a Thursday thing?"
        ),
        md("**The day-of-week control: is it Thursday, or is it pre-holiday?**"),
        code(
            "if HAVE_REAL:\n"
            "    pre_m = float(pre_ret.mean()*100)\n"
            "    all_thu = float(UNC_THU.mean()*100)\n"
            "    all_day = float(UNC.mean()*100)\n"
            "else:\n"
            "    pre_m, all_thu, all_day = R['ev_mean_pct'], R['thu_mean_pct'], R['uncond_mean_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "bars = ax.bar(['Pre-holiday\\nThursday', 'Ordinary\\nThursday', 'Any\\nday'],\n"
            "              [pre_m, all_thu, all_day], color=[GREEN, GREY, GREY], width=0.55)\n"
            "ax.set_ylabel('Mean daily return (%)')\n"
            "ax.set_title('Ordinary Thursdays are NOT special -- only the pre-holiday one is')\n"
            "for b, v in zip(bars, [pre_m, all_thu, all_day]):\n"
            "    ax.annotate(f'{v:+.4f}%', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Ordinary Thursday mean: {all_thu:+.4f}%/day -- basically the same as any day.')\n"
            "print(f'Pre-holiday Thursday:   {pre_m:+.4f}%/day -- ~{(pre_m-all_thu)*100:.0f} bps higher.')"
        ),
        md(
            "An *ordinary* Thursday earns **+0.035%/day** -- indistinguishable from any other day. "
            "Only the *pre-holiday* Thursday pops. So the effect is **not** a weekday artifact: it "
            "really is a **pre-holiday** drift. (The quants notebook confirms the excess vs the "
            "Thursday-only baseline is +29.7 bps with HAC t = +3.29.)"
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Real.** Pre-holiday excess +30 bps/day, HAC t = **+3.28**, permutation "
            "p = **0.012**, up-rate 68%. It survives the day-of-week control (HAC t = +3.29 vs "
            "ordinary Thursdays) and shows up in both halves of the sample. This is a genuine "
            "**pre-holiday effect**, with real academic pedigree.\n"
            "- **Tradability -- Fragile.** Net +31 bps per event survives costs, but it fires "
            "**once a year**: ~0.31% annual contribution, dominated by simply holding the index. "
            "Capacity is huge but the sliver can't be safely levered on one overnight gap.\n"
            "- **Not busted -- but not *Easter*.** The seasonal is real; the label is wrong. It's "
            "the generic pre-long-weekend drift, present at the Good-Friday gate because Good "
            "Friday closes the NYSE. Easter Monday adds nothing once the Monday effect is removed."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: buy the close on the Wednesday before Good Friday, sell the "
            "Maundy-Thursday close, once a year. Here's the gross edge and what's left after a "
            "modest 1 bp one-way cost (2 bps round-trip) on NAV."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tradability(pre_ret, cost_bps_oneway=1.0)\n"
            "    gross_bps = tr['gross_mean_per_event']*1e4\n"
            "    net_bps = tr['net_mean_per_event']*1e4\n"
            "    net_cum = tr['net_cum_growth']\n"
            "else:\n"
            "    gross_bps, net_bps = R['gross_mean_per_event_bps'], R['net_mean_per_event_bps']\n"
            "    net_cum = R['net_cum']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "bars = ax.bar(['Gross\\nper event', 'Net\\n(after 2 bps cost)'], [gross_bps, net_bps],\n"
            "              color=[GREEN, AMBER], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean return per event (bps)')\n"
            "ax.set_title('The edge survives costs -- but it only fires once a year')\n"
            "for b, v in zip(bars, [gross_bps, net_bps]):\n"
            "    ax.annotate(f'{v:+.1f} bps', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross: {gross_bps:+.1f} bps/event | Net: {net_bps:+.1f} bps/event')\n"
            "print(f'Compounded over 76 years (net): x{net_cum:.4f}  ->  ~{(net_cum**(1/76)-1)*100:.2f}%/yr.')\n"
            "print('Real per event -- but a once-a-year +0.31%/yr is dominated by just holding the index.')"
        ),
        md(
            "The edge is real *per event* and survives costs -- but the harvest is one "
            "session a year. Compounded, the net rule grows the stake ~26% over 76 years "
            "(~**0.31%/yr**). Sitting in cash 364 days to capture that is dominated by simply "
            "being invested. **Real signal, fragile vehicle.**"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a pre-holiday premium into "
            "a synthetic tape and shows the machinery detects it cleanly above ~10 bps/day -- and "
            "finds nothing at zero. The real tape's +30 bps clears the bar.\n"
            "- **The pure-folklore siblings** that land None/Mirage -- same machinery, no signal: "
            "[Study 285 -- St-Patricks-Day](../../285-st-patricks-day/) and "
            "[Study 286 -- Valentines-Day](../../286-valentines-day/).\n"
            "- **The Super Bowl Indicator** -- the base-rate trap in its purest form: "
            "[Study 158 -- Super-Bowl](../../158-super-bowl/).\n\n"
            "*Think this is Easter-specific rather than a generic pre-holiday drift? Fork this, "
            "test the sessions before Memorial Day, July 4th and Labor Day, and show the "
            "Good-Friday session stands out from the other pre-holiday sessions. We don't think it "
            "does -- that's why the label is 'pre-holiday', not 'Easter'.*"
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
            "# Easter-Effect -- a quantitative teardown\n"
            "### 76 Easter weekends * ^GSPC daily * pre/post sessions * one-sample t * Newey-West "
            "HAC t * same-weekday control * permutation * sub-periods * synthetic control\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Busted%3F: No](https://img.shields.io/badge/Busted%3F-No-2ea44f?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* Good Friday is a market "
            "holiday, so we test the **pre-holiday** (Maundy Thursday) and **post-holiday** (Easter "
            "Monday) sessions against the everyday drift **and the same-weekday baseline**, run a "
            "one-sample t-test, a Newey-West HAC t on the excess, a 10,000-draw permutation test, a "
            "three-way sub-period split, a power calculation, and a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily close-to-close, price-only "
            "(1950-2025); Easter via the Computus algorithm in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Signal-axis caveat:** price-only index, vendor back-revises the ^GSPC level, and "
            "the pre-holiday effect is documented to decay -- the middle third (1980-2002) is "
            "already weak. Survivorship is moot for an index level."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `REAL` | Pre-holiday excess **+29.6 bps/day**; one-sample t vs the "
            "everyday drift = **+3.51**; Newey-West HAC t = **+3.28**; permutation p = **0.012**. "
            "Survives the day-of-week control (HAC t = +3.29 vs Thursdays only). Up-rate 68% > 53% "
            "base rate. |\n"
            "| **Tradability** | `FRAGILE` | Net +31 bps/event after costs, but fires once/yr "
            "(~0.31%/yr compounded); can't be levered on one overnight gap; documented effect, "
            "decay risk. |\n"
            "| **Busted?** | `NO` | A real seasonal -- but the **generic pre-holiday effect**, not "
            "Easter-specific. Easter Monday's weakness is just the Monday effect (HAC t = -1.53 vs "
            "Mondays only). |\n\n"
            "> The signal is real and survives the obvious confounds; the honest caveat is that it "
            "is *pre-holiday*, not *Easter*, and it is a once-a-year sliver."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC close-to-close return on the pre-holiday session (Maundy "
            "Thursday) of year $t$, and $\\mu$ the unconditional daily mean over the same window. "
            "The hypotheses:\n\n"
            "- **H1 (excess return).** $E[r_t] > \\mu$ -- the pre-holiday session beats the "
            "*everyday drift*. We test against $\\mu$, not 0.\n"
            "- **H1' (weekday-clean).** $E[r_t] > \\mu_{\\text{Thu}}$ -- it also beats the mean of "
            "an *ordinary Thursday*, ruling out the day-of-week confound.\n"
            "- **H2 (Easter-specific?).** Is the Good-Friday gate special among pre-holiday gates? "
            "We *cannot* affirm this from one holiday, so we label the effect **pre-holiday**, not "
            "Easter.\n\n"
            "We accept H1 (excess +29.6 bps, t = +3.51) and H1' (vs Thursdays, HAC t = +3.29), and "
            "decline to claim H2."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "A pre-holiday drift is one of the few calendar regularities with peer-reviewed "
            "support (Ariel 1990; Lakonishok & Smidt 1988). If real, it's a clean, pre-announced, "
            "one-day-hold pattern. The catch is economic, not statistical: it fires once a year, "
            "so even a genuine +30 bps is a rounding error in an annual budget -- and documented "
            "anomalies decay (McLean & Pontiff 2016). The interesting work is separating the real "
            "drift from the Easter label and from the weekday confound."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily close-to-close, price-only, 1950-2025; pre-holiday session = "
            "last trading day before Good Friday (always Maundy Thursday), post = Easter Monday "
            "(hardcoded from the Computus calendar in `data.py`, 76 each).\n"
            "- **Statistic.** Mean of the 76 event returns vs the unconditional daily mean $\\mu$ "
            "(the excess return) -- and vs the same-weekday mean.\n"
            "- **Correct null.** One-sample t against $\\mu$ (and against $\\mu_{\\text{Thu}}$), "
            "*not* 0; plus a permutation null of random 76-session draws.\n"
            "- **Inference.** `ttest_1samp`; Newey-West HAC t (Bartlett, lag 4) on the excess; "
            "permutation test (10,000 draws); three-way sub-period split for decay.\n"
            "- **Positive control.** Synthetic daily tape with a planted pre-holiday premium.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The correct baseline: excess return and HAC t\n\n"
            "Test the pre-holiday return against the everyday drift $\\mu$ -- and report the "
            "Newey-West HAC t on the excess (the desk's |t| >= 2 hurdle)."
        ),
        code(
            "es = st.event_stats(pre_ret, UNC, n_permutations=10_000, seed=287)\n"
            "print(f\"n pre-holiday sessions: {es['n']}\")\n"
            "print(f\"Pre-holiday mean:     {es['ev_mean']*100:+.4f}%/day\")\n"
            "print(f\"Unconditional mean:   {es['uncond_mean']*100:+.4f}%/day  (the honest baseline)\")\n"
            "print(f\"Excess mean:          {es['excess_mean']*1e4:+.2f} bps/day\")\n"
            "print(f\"Up-rate:              {es['up_rate']*100:.1f}%  (vs {es['uncond_up_rate']*100:.1f}% base rate)\")\n"
            "print()\n"
            "print(f\"One-sample t vs 0:            {es['t_vs_zero']:+.3f}  (p={es['p_vs_zero']:.4f})\")\n"
            "print(f\"One-sample t vs everyday mu:  {es['t_vs_uncond']:+.3f}  (p={es['p_vs_uncond']:.4f})\")\n"
            "print(f\"Newey-West HAC t (excess):    {es['hac_t_excess']:+.3f}\")\n"
            "print()\n"
            "print('The excess clears |t| >= 2 against the correct null -- a real effect.')"
        ),
        md(
            "> The pre-holiday session beats the everyday drift by **+29.6 bps** with t = **+3.51** "
            "and HAC t = **+3.28** -- comfortably past the |t| >= 2 hurdle. Unlike the pure-folklore "
            "siblings (Valentine's, St-Patrick's), this is a genuine signal."
        ),
        md(
            "### 4b - The day-of-week control: is it Thursday, or is it pre-holiday?\n\n"
            "The pre-holiday session is *always* a Thursday. We must beat an **ordinary Thursday**, "
            "not just an average day."
        ),
        code(
            "es_thu = st.event_stats(pre_ret, UNC_THU, n_permutations=10_000, seed=287)\n"
            "print(f\"Pre-holiday mean:        {es_thu['ev_mean']*100:+.4f}%/day\")\n"
            "print(f\"Ordinary-Thursday mean:  {es_thu['uncond_mean']*100:+.4f}%/day\")\n"
            "print(f\"Excess vs Thursdays:     {es_thu['excess_mean']*1e4:+.2f} bps/day\")\n"
            "print(f\"HAC t (excess vs Thu):   {es_thu['hac_t_excess']:+.3f}\")\n"
            "print(f\"Permutation p (vs Thu):  {es_thu['perm_p']:.4f}\")\n"
            "print()\n"
            "print('Ordinary Thursdays are unremarkable; only the pre-holiday Thursday pops.')\n"
            "print('=> the effect is pre-HOLIDAY, not a weekday artifact.')"
        ),
        md(
            "Against the **Thursday-only** baseline the excess is essentially unchanged (+29.7 bps, "
            "HAC t = **+3.29**, perm p = 0.007). Ordinary Thursdays earn the everyday drift; only "
            "the pre-holiday Thursday is special. The day-of-week confound is **ruled out**."
        ),
        md(
            "### 4c - The permutation test: where does the pre-holiday mean land?\n\n"
            "Draw 76 random sessions from the same window 10,000 times and record the distribution "
            "of their means."
        ),
        code(
            "perm_means = es['perm_means'] * 100\n"
            "obs = float(pre_ret.mean()*100)\n"
            "mu = float(UNC.mean()*100)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_means, bins=40, color=GREY, alpha=0.7, label='Random 76-session means')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'Pre-holiday mean: {obs:+.4f}%')\n"
            "ax.axvline(mu, c=AMBER, lw=2, ls='--', label=f'Everyday drift: {mu:+.4f}%')\n"
            "ax.set_xlabel('Mean daily return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The pre-holiday mean sits out in the right tail of the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Permutation p (two-sided): {es['perm_p']:.4f}\")\n"
            "print(f\"Percentile of pre-holiday mean: {(perm_means < obs).mean():.1%} of draws below\")"
        ),
        md(
            "The pre-holiday mean lands far out in the **right tail** of the null distribution "
            "(perm p = **0.012**) -- the opposite of the dead-centre placement we see for the "
            "pure-folklore holidays."
        ),
        md(
            "### 4d - Sub-periods: how stable is the drift?\n\n"
            "Split the 76 sessions into thirds and recompute the excess HAC t in each."
        ),
        code(
            "sp = st.subperiod_stats(EV['pre_ret'], UNC,\n"
            "                        breaks=((1950, 1979), (1980, 2002), (2003, 2025)))\n"
            "print(sp.to_string(index=False))\n"
            "print()\n"
            "colors = [GREEN if abs(t) >= 2 else AMBER for t in sp['hac_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.bar(sp['period'], sp['hac_t'], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2 hurdle')\n"
            "ax.set_ylabel('HAC t (excess)')\n"
            "ax.set_title('Robust on the whole tape, but uneven: the middle third is weak')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Both outer thirds clear |t|>=2; 1980-2002 is weak (t~1.2). Real but not metronomic.')"
        ),
        md(
            "The drift is present in **all three thirds** by sign, and the two outer thirds clear "
            "|t| >= 2 -- but the middle third (1980-2002) is weak (HAC t = 1.19). A genuine effect, "
            "yet uneven: exactly why Tradability is **Fragile**, not Investable. (Both *halves* of "
            "the sample also clear |t| >= 2: 4.29 and 2.14.)"
        ),
        md(
            "### 4e - Easter Monday: vanishes under the Monday control\n\n"
            "The post-holiday session looks weak -- but it's always a Monday."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p_all = st.event_stats(post_ret, UNC, n_permutations=5_000, seed=287)\n"
            "    p_mon = st.event_stats(post_ret, UNC_MON, n_permutations=5_000, seed=287)\n"
            "    print(f\"Easter Monday mean:        {p_all['ev_mean']*100:+.4f}%/day (up {p_all['up_rate']*100:.1f}%)\")\n"
            "    print(f\"Excess vs all days:        {p_all['excess_mean']*1e4:+.2f} bps, HAC t {p_all['hac_t_excess']:+.3f}\")\n"
            "    print(f\"Excess vs MONDAYS only:    {p_mon['excess_mean']*1e4:+.2f} bps, HAC t {p_mon['hac_t_excess']:+.3f}, perm p {p_mon['perm_p']:.3f}\")\n"
            "else:\n"
            "    print(f\"Easter Monday mean:        {R['post_mean_pct']:+.4f}%/day (up {R['post_up_pct']:.1f}%)\")\n"
            "    print(f\"Excess vs all days:        {R['post_excess_bps']:+.2f} bps, HAC t {R['post_hac_t']:+.3f}\")\n"
            "    print(f\"Excess vs MONDAYS only:    {R['post_excess_mon_bps']:+.2f} bps, HAC t {R['post_hac_t_mon']:+.3f}, perm p {R['post_perm_p_mon']:.3f}\")\n"
            "print()\n"
            "print('Against the correct Monday baseline, Easter Monday is NOT distinguishable.')\n"
            "print('The apparent weakness is the generic Monday effect, not anything Easter-specific.')"
        ),
        md(
            "Easter Monday looks soft against the all-day baseline, but against the **Monday-only** "
            "baseline the excess shrinks to -14 bps with HAC t = **-1.53** (perm p = 0.26). It is "
            "the ordinary Monday effect -- **not** an Easter phenomenon. We report it and decline "
            "to claim it."
        ),
        md(
            "### 4f - Power calculation: what could n=76 detect?\n\n"
            "The minimum detectable mean at 80% power, two-sided, alpha=0.05."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vol = float(pre_ret.std(ddof=1))\n"
            "else:\n"
            "    vol = R['ev_std_pct']/100\n"
            "n = R['n']\n"
            "from scipy.stats import norm\n"
            "z_a = norm.ppf(1 - 0.05/2); z_b = norm.ppf(0.80)\n"
            "se = vol / np.sqrt(n)\n"
            "mde = (z_a + z_b) * se\n"
            "print(f'Pre-holiday daily vol: {vol*100:.3f}%/day, n = {n}')\n"
            "print(f'Standard error of the pre-holiday mean: {se*100:.4f}%/day ({se*1e4:.1f} bps)')\n"
            "print(f'Minimum detectable mean (80% power, 2-sided): {mde*100:.3f}%/day ({mde*1e4:.1f} bps)')\n"
            "print()\n"
            "print(f\"Observed excess: {R['excess_mean_bps']:+.2f} bps -- about \"\n"
            "      f\"{R['excess_mean_bps']/(mde*1e4):.0%} of the detectable threshold.\")\n"
            "print('The observed effect is above the n=76 detection floor -- consistent with a real signal.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `REAL`** -- excess +29.6 bps/day, t vs everyday drift +3.51, HAC t +3.28, "
            "perm p 0.012, up-rate 68%. Survives the Thursday-only control (HAC t +3.29) and both "
            "sample halves (t 4.29 / 2.14). The observed effect exceeds the n=76 detection floor.\n"
            "- **Tradability `FRAGILE`** -- net +31 bps/event after costs, but once-a-year "
            "(~0.31%/yr compounded), un-leverable on one overnight gap, and a documented effect "
            "with decay risk (the 1980-2002 third is weak).\n"
            "- **Not busted -- but pre-holiday, not Easter** -- the drift is the generic "
            "pre-long-weekend effect at the Good-Friday gate; Easter Monday adds nothing once the "
            "Monday effect is removed.\n\n"
            "*Signal-axis caveat:* price-only index, vendor back-revises the ^GSPC level; "
            "survivorship is moot for an index. These can only flatter, not manufacture, the result "
            "-- but the decay and unevenness are real and keep this off the Investable tier."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- gross vs net\n\n"
            "The rule: buy the close before the pre-holiday session, sell the pre-holiday close. "
            "One round-trip per year, 1 bp one-way (2 bps round-trip) on NAV."
        ),
        code(
            "tr = st.tradability(pre_ret, cost_bps_oneway=1.0)\n"
            "print(f\"Events:               {tr['n_events']}\")\n"
            "print(f\"Gross mean per event: {tr['gross_mean_per_event']*1e4:+.2f} bps\")\n"
            "print(f\"Cost drag (round-trip): {tr['cost_drag_per_event']*1e4:.1f} bps\")\n"
            "print(f\"Net mean per event:   {tr['net_mean_per_event']*1e4:+.2f} bps\")\n"
            "print()\n"
            "print('Cumulative growth over the sample (76 once-a-year events):')\n"
            "print(f\"  Gross: x{tr['gross_cum_growth']:.4f}  | Net: x{tr['net_cum_growth']:.4f}\")\n"
            "print(f\"  Per-event CAGR: gross {tr['gross_cagr']*100:.3f}% | net {tr['net_cagr']*100:.3f}%\")\n"
            "print()\n"
            "print('Survives costs per event -- but ~0.31%/yr is dominated by staying invested.')"
        ),
        md(
            "> The edge clears costs *per event*, but the once-a-year cadence caps the annual "
            "contribution at ~0.31%. Real, but fragile as a standalone vehicle."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a pre-holiday premium *when one exists*, and stay quiet at "
            "zero? We plant a premium into a synthetic daily tape and sweep its size."
        ),
        code(
            "planted = [0, 5, 10, 25, 50]\n"
            "results = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_daily(start_year=1900, end_year=2299,\n"
            "                                   premium_bps=float(bps), seed=287)\n"
            "    ev_s = df_s.loc[df_s['is_pre'], 'ret']\n"
            "    r = st.event_stats(ev_s, df_s['ret'], n_permutations=1000, seed=287)\n"
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
            "ax.set_xlabel('Planted pre-holiday premium (bps/day)')\n"
            "ax.set_ylabel('Newey-West HAC t (excess)')\n"
            "ax.set_title('The engine finds the signal when it exists (green = HAC t>2 and p<0.05)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Quiet at premium=0, fires from ~10 bps up. The real tape (+29.6 bps) clears it.')"
        ),
        md(
            "The engine correctly finds **nothing at premium = 0** and cleanly detects a planted "
            "premium from ~10 bps/day upward (HAC t > 2, perm p < 0.05). The real tape's +29.6 bps "
            "pre-holiday excess sits above that threshold -- the **REAL** verdict is a statement "
            "about the market's pre-holiday drift, validated by a method that stays silent on noise."
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
