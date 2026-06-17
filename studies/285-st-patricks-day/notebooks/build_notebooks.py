"""Generate (and execute) the two narrative notebooks for Study 285 (St-Patricks-Day).

    python notebooks/build_notebooks.py

Builds both notebooks then executes them in place with nbconvert (no
``--allow-errors``). The synthetic figures run anywhere, offline and
deterministic; the real-tape cells use the shared ``_cache/^GSPC_split_only.parquet``
if present and otherwise fall back to the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebooks re-run for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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
    fp="df3897aef897",
    year_start=1928, year_end=2026, n_days=24728,
    n_stpat=99,
    mean_stpat_bps=14.8, tstat_stpat_hac=1.13,
    mean_all_bps=2.4, contrast_vs_all_bps=12.4,
    tstat_welch_all=0.93, pval_welch_all=0.353,
    mean_march_other_bps=0.7, contrast_vs_march_bps=14.1,
    tstat_welch_march=1.04, pval_welch_march=0.299,
    n_placebo=99, mean_placebo_bps=3.5, pval_placebo_march=0.877,
    pval_bonferroni_17=1.0,
    # sub-periods
    sub_a_lbl="1928-1960", sub_a_n=33, sub_a_bps=-9.9, sub_a_t=-0.43,
    sub_b_lbl="1961-1990", sub_b_n=30, sub_b_bps=-12.2, sub_b_t=-0.89,
    sub_c_lbl="1991-2026", sub_c_n=36, sub_c_bps=59.9, sub_c_t=3.27,
    # strategy
    net_mean_bps=12.8, net_tstat_hac=0.98, strat_per_event_pct=0.128,
    # multiple-comparisons sweep (anchor_day: (n, mean_bps, p_raw))
    mc_31=(67, -24.8, 0.107), mc_10=(99, 23.1, 0.265), mc_17=(99, 14.8, 0.299),
    mc_03=(99, 8.6, 0.433), mc_24=(99, 3.5, 0.877),
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
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

from st_patricks_day import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("GSPC cache present:", HAVE_REAL)
"""

R_CELL = f"""\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = {R!r}
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# St-Patricks-Day -- is there a 'wearin' o' the green' rally?\n"
            "### A folk calendar bump, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every March you'll hear that the stock market gets a **St. Patrick's Day bump** -- "
            "the 'wearin' o' the green' rally, where equities are supposed to do unusually well "
            "on (or around) March 17. This notebook runs the only test that matters: across "
            "almost a century of S&P 500 history, is the St. Patrick's session actually greener "
            "than any other trading day, or is it just one lucky number in a calendar full of them?\n\n"
            "> **This is the plain-language layer.** Want the HAC t-stats, the placebo, the "
            "multiple-comparisons sweep and the sub-period split? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the St. Patrick's session greener than average? | **Barely.** Mean **+14.8 bps** "
            "vs **+2.4 bps** for all other days -- but HAC t = **1.13**, well below the t=2 bar. |\n"
            "| Does it beat other March days? | **No, not significantly.** +14.1 bps richer than "
            "the rest of March, Welch p = **0.30**. |\n"
            "| Is it special, or just one lucky day? | A snooper testing five March anchor days "
            "finds nothing survives Bonferroni; the placebo (March 24) is a flat +3.5 bps. |\n"
            "| Where does the 'bump' come from? | **One regime.** It is +60 bps in 1991-2026 but "
            "*negative* in 1928-1990. A genuine signal doesn't flip sign across history. |\n"
            "| Could you trade it? | **No.** One day a year, ~13 bps net of costs, dominated by "
            "just holding the index. |\n\n"
            "> The St. Patrick's Day bump is folklore: a mildly positive day that owes its whole "
            "reputation to a single lucky modern regime and the fact that markets drift up anyway."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The market rallies on St. Patrick's Day -- the 'wearin' o' the green' bump. "
            "Buy the S&P on March 17 and you'll catch a reliably green session.\"*\n\n"
            "Unlike most anomalies, this one has **no academic paper** behind it -- there is no "
            "peer-reviewed 'St. Patrick's Day effect'. It is pure calendar superstition, in the "
            "same family as the Santa Claus rally and the Halloween indicator, but thinner: a "
            "single fixed civil-calendar date, March 17."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a single named holiday reliably delivered an above-average return, it would be the "
            "easiest trade in finance: hold the index for one day a year and skip the rest. The "
            "interesting question is *why* a perfectly ordinary day acquires a green reputation -- "
            "and the answer is a clinic in how calendar folklore is born from a lucky sub-sample "
            "and the market's natural upward drift."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Four traps to avoid:\n\n"
            "1. **The drift baseline.** Stocks drift up ~+2.4 bps/day unconditionally. Any single "
            "day will *look* green on average. The right question is whether March 17 beats that "
            "baseline by a statistically real margin -- HAC t >= 2, not just a positive number.\n"
            "2. **March seasonality.** March may itself be a good (or bad) month. We compare the "
            "St. Patrick session against *other March days*, not just against the whole year, to "
            "strip out generic spring seasonality.\n"
            "3. **A placebo.** We test March 24 (one week later) with the same machinery. If a "
            "folklore-free day looks just as green, the St. Patrick result is noise.\n"
            "4. **Multiple comparisons.** March 17 was picked *because* of the holiday. A snooper "
            "testing many March anchor days would find the most extreme by chance, so we Bonferroni-"
            "correct across a grid.\n\n"
            "We test all of this on ^GSPC daily returns from 1928 to 2026 (99 St. Patrick sessions)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the headline:** the St. Patrick's session mean vs the unconditional drift "
            "and vs the rest of March."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc(fetch=False)\n"
            "    pri = st.stpat_comparison(df)\n"
            "    n_ev = pri['n_stpat']\n"
            "    m_sp = pri['mean_stpat_bps']\n"
            "    m_all = pri['mean_all_bps']\n"
            "    m_mar = pri['mean_march_other_bps']\n"
            "    t_hac = pri['tstat_stpat_hac']\n"
            "else:\n"
            "    n_ev = R['n_stpat']\n"
            "    m_sp = R['mean_stpat_bps']; m_all = R['mean_all_bps']\n"
            "    m_mar = R['mean_march_other_bps']; t_hac = R['tstat_stpat_hac']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['St. Patrick\\nsession', 'All other\\ndays', 'Other March\\ndays'],\n"
            "              [m_sp, m_all, m_mar], color=[GREEN, GREY, AMBER], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily log-return (bps)')\n"
            "ax.set_title(f'St. Patrick mean +{m_sp:.1f} bps -- but HAC t = {t_hac:.2f} (< 2)')\n"
            "for b, v in zip(bars, [m_sp, m_all, m_mar]):\n"
            "    ax.annotate(f'{v:+.1f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'St. Patrick sessions: {n_ev} (1928-2026)')\n"
            "print(f'Mean: {m_sp:+.1f} bps  |  all other days: {m_all:+.1f} bps  |  HAC t = {t_hac:.2f}')\n"
            "print('A +14.8 bps day sounds great until you see the t-stat: noise.')"
        ),
        md(
            "The St. Patrick's session does average **+14.8 bps** -- about six times the +2.4 bps "
            "unconditional drift. But the HAC t-stat is only **1.13**: with 99 noisy daily "
            "observations (~131 bps daily standard deviation), a +14.8 bps mean is comfortably "
            "inside what chance produces. It does not clear the t=2 bar."
        ),
        md("**Is the bump real, or does it live in one lucky era?**"),
        code(
            "if HAVE_REAL:\n"
            "    sub = st.subperiod_table(df)\n"
            "    labels = sub['period'].tolist()\n"
            "    vals = sub['mean_bps'].tolist()\n"
            "    tsts = sub['tstat_hac'].tolist()\n"
            "else:\n"
            "    labels = [R['sub_a_lbl'], R['sub_b_lbl'], R['sub_c_lbl']]\n"
            "    vals = [R['sub_a_bps'], R['sub_b_bps'], R['sub_c_bps']]\n"
            "    tsts = [R['sub_a_t'], R['sub_b_t'], R['sub_c_t']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [GREEN if v > 0 else RED for v in vals]\n"
            "bars = ax.bar(labels, vals, color=colors, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('St. Patrick mean (bps)')\n"
            "ax.set_title('The bump is entirely a post-1990 phenomenon -- negative before')\n"
            "for b, v, t in zip(bars, vals, tsts):\n"
            "    ax.annotate(f'{v:+.0f} bps\\n(t={t:+.2f})',\n"
            "                (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1928-1960:', f'{vals[0]:+.1f} bps', '| 1961-1990:', f'{vals[1]:+.1f} bps',\n"
            "      '| 1991-2026:', f'{vals[2]:+.1f} bps')\n"
            "print('A real signal does not flip sign across regimes. This one does.')"
        ),
        md(
            "Here is the kill shot. The St. Patrick's bump is **+59.9 bps** in 1991-2026 (t = 3.27) "
            "but **negative** in both 1928-1960 (-9.9 bps) and 1961-1990 (-12.2 bps). The famous "
            "'green rally' is really just the last 35 years getting lucky on one calendar slot. "
            "A signal that exists only in one regime and reverses in the others is a regime "
            "artifact, not an exploitable edge."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Full-sample mean +14.8 bps but HAC t = **1.13**; vs other "
            "March days Welch p = **0.30**. The whole effect lives in one post-1990 sub-period and "
            "is negative before it -- the hallmark of a regime artifact, not a stable signal.\n"
            "- **Tradability -- Mirage.** One day a year, ~13 bps net of costs. There is no vehicle "
            "that beats simply holding the index, which captures the same upward drift every day.\n"
            "- **Busted.** A placebo a week later (March 24) is a flat +3.5 bps, and no March "
            "anchor day survives a Bonferroni correction. March 17 is just one lucky number."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' is to sit in cash all year and be long the index only on the St. "
            "Patrick's session. One round-trip a year, so costs are tiny -- but so is the prize."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pnl = st.strategy_pnl(df, one_way_bps=1.0)\n"
            "    net_bps = pnl['net_mean_bps']\n"
            "    net_t = pnl['net_tstat_hac']\n"
            "    bah = pnl['bah_daily_ann_pct']\n"
            "else:\n"
            "    net_bps = R['net_mean_bps']; net_t = R['net_tstat_hac']; bah = None\n"
            "\n"
            "print(f'St. Patrick trade (net of 2 bps round-trip): {net_bps:+.1f} bps/event, HAC t = {net_t:.2f}')\n"
            "if bah is not None:\n"
            "    print(f'Buy-and-hold the index: {bah:+.1f}%/yr -- earned every single day, not one.')\n"
            "print()\n"
            "print('Being in the market ONE day a year forfeits ~250 other drift days.')\n"
            "print('The only real trade here is: do not build a portfolio around a holiday.')"
        ),
        md(
            "Net of a 2 bps round-trip the bump is ~13 bps for one day -- and not statistically "
            "distinguishable from zero. Buy-and-hold earns the market's drift every day of the year; "
            "the St. Patrick trade earns it once. There is no edge to harvest."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a synthetic St. Patrick "
            "bump and shows the same engine lights up (HAC t >> 2) when an effect truly exists. "
            "The real tape has nothing like that.\n"
            "- **Same family:** other fixed-date calendar folklore -- the [Super-Bowl indicator]"
            "(../../158-super-bowl/), [Friday-the-13th](../../163-friday-13th/), and the "
            "[year-ending-in-five](../../161-year-ending-five/) lore -- all share the small-n, "
            "multiple-comparisons trap.\n\n"
            "*Think March 17 really is special? Fork this, pre-register the date, and show a "
            "stable HAC t >= 2 that survives the placebo and the sub-period split. That's the bar -- "
            "and it isn't cleared.*"
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
            "# St-Patricks-Day -- a quantitative teardown\n"
            "### 99 sessions * ^GSPC daily * HAC Newey-West * placebo * Bonferroni * sub-period split\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test the St. Patrick's "
            "Day bump on ^GSPC daily returns with a HAC (Newey-West) t-stat on the event-day mean, "
            "a Welch contrast against all other days and against other March days, a March-24 "
            "placebo, a Bonferroni-corrected day-of-March sweep, a sub-period split, and a synthetic "
            "positive control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily close (price-only log-returns, "
            "1928--2026) from the shared `_cache/^GSPC_split_only.parquet`; St. Patrick sessions "
            "derived by date arithmetic in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship:** none -- ^GSPC is an index, not a survivor-selected basket. "
            "**Returns are price-only** (no dividends); a total-return tape would lift *every* day "
            "equally and not change the contrast."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | St. Patrick mean **+14.8 bps**, HAC t = **1.13**; vs other "
            "March days Welch p = **0.299**. |\n"
            "| **Tradability** | `MIRAGE` | One round-trip/yr, ~13 bps net, HAC t = **0.98**; "
            "dominated by holding the index. |\n"
            "| **Busted?** | `YES` | Effect is +60 bps post-1990, *negative* pre-1990; placebo "
            "(Mar 24) flat; nothing survives Bonferroni. |\n\n"
            "> The unconditional drift and a single lucky modern regime do all the work. "
            "March 17 carries no statistically distinguishable bump."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC daily log-return and $E_t \\in \\{0,1\\}$ indicate the St. "
            "Patrick's Day trading session (first session on/after March 17). The hypotheses:\n\n"
            "- **H1 (level).** $E[r_t \\mid E_t=1] > E[r_t]$ with HAC t >= 2 -- the event day beats "
            "the unconditional drift.\n"
            "- **H2 (vs March).** $E[r_t \\mid E_t=1] > E[r_t \\mid \\text{other March day}]$ -- the "
            "bump survives stripping out generic March seasonality.\n"
            "- **H3 (specificity).** The placebo (March 24) shows no comparable bump, and March 17 "
            "survives a Bonferroni correction across candidate March anchor days.\n"
            "- **H4 (stability).** The effect is reasonably stable across sub-periods.\n\n"
            "We reject H1, H2, H3 and H4 on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The St. Patrick's bump is folklore with no mechanism: a fixed civil-calendar date has "
            "no reason to move corporate cash flows. If H1--H4 held it would be a pure free lunch. "
            "The instructive finding is *how* an ordinary day earns a green reputation -- a lucky "
            "modern sub-sample, the market's upward drift, and the implicit multiple testing of "
            "every named day on the calendar."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily close, 1928--2026; log-returns; St. Patrick sessions derived "
            "by date arithmetic (first trading day on/after March 17), 99 events.\n"
            "- **Event-day inference.** Newey-West HAC t-stat on the event-day mean (lag = "
            "$\\lfloor 4(n/100)^{2/9}\\rfloor$); Welch t-test on the contrast vs all-other-days "
            "and vs other-March-days.\n"
            "- **Placebo.** March 24 (one week later), same machinery.\n"
            "- **Multiple comparisons.** Day-of-March sweep over {3,10,17,24,31}, Bonferroni-corrected.\n"
            "- **Robustness.** Three-regime sub-period split.\n"
            "- **Positive control.** Synthetic tape with a planted bump confirms the engine fires "
            "when an effect exists.\n"
            "- **Lag/cost honesty.** Execution at the close of the labelled session (label known "
            "pre-open); strategy P&L charged 2 bps round-trip; price-only (no dividends)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Event-day HAC t-stat vs the two baselines\n\n"
            "A positive mean is not a signal. The HAC t-stat is."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc(fetch=False)\n"
            "    pri = st.stpat_comparison(df)\n"
            "    fp = data.fingerprint(df)\n"
            "    print(f'Data stamp: {df.index.min().date()} -> {df.index.max().date()}'\n"
            "          f'  n_days={len(df)}  fingerprint={fp}')\n"
            "else:\n"
            "    pri = dict(n_stpat=R['n_stpat'], mean_stpat_bps=R['mean_stpat_bps'],\n"
            "               tstat_stpat_hac=R['tstat_stpat_hac'], mean_all_bps=R['mean_all_bps'],\n"
            "               contrast_vs_all_bps=R['contrast_vs_all_bps'], tstat_welch_all=R['tstat_welch_all'],\n"
            "               pval_welch_all=R['pval_welch_all'], mean_march_other_bps=R['mean_march_other_bps'],\n"
            "               contrast_vs_march_bps=R['contrast_vs_march_bps'], tstat_welch_march=R['tstat_welch_march'],\n"
            "               pval_welch_march=R['pval_welch_march'])\n"
            "    print(f'[frozen] fingerprint={R[\"fp\"]}  n_days={R[\"n_days\"]}')\n"
            "\n"
            "print(f\"n events: {pri['n_stpat']}\")\n"
            "print(f\"St. Patrick mean: {pri['mean_stpat_bps']:+.1f} bps   HAC t = {pri['tstat_stpat_hac']:+.2f}\")\n"
            "print(f\"vs all other days:  contrast {pri['contrast_vs_all_bps']:+.1f} bps  Welch t = {pri['tstat_welch_all']:+.2f}  p = {pri['pval_welch_all']:.3f}\")\n"
            "print(f\"vs other March days: contrast {pri['contrast_vs_march_bps']:+.1f} bps  Welch t = {pri['tstat_welch_march']:+.2f}  p = {pri['pval_welch_march']:.3f}\")\n"
            "print()\n"
            "print('All three t-stats are below 2. The bump is not statistically real.')"
        ),
        md(
            "> The event-day mean (+14.8 bps) is positive but its HAC t-stat is **1.13**. Against "
            "other March days the contrast is +14.1 bps with Welch p = **0.299**. Nothing clears "
            "the inference bar."
        ),
        md(
            "### 4b - The placebo and the multiple-comparisons sweep\n\n"
            "March 17 was chosen *because* of the holiday. Test the neighbourhood and correct for it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac = st.placebo_comparison(df)\n"
            "    mc = st.day_of_march_sweep(df)\n"
            "    plac_bps = plac['mean_placebo_bps']; plac_p = plac['pval_welch_march']\n"
            "else:\n"
            "    plac_bps = R['mean_placebo_bps']; plac_p = R['pval_placebo_march']\n"
            "    mc = pd.DataFrame([\n"
            "        {'anchor_day': 31, 'n': R['mc_31'][0], 'mean_bps': R['mc_31'][1], 'pval_raw': R['mc_31'][2]},\n"
            "        {'anchor_day': 10, 'n': R['mc_10'][0], 'mean_bps': R['mc_10'][1], 'pval_raw': R['mc_10'][2]},\n"
            "        {'anchor_day': 17, 'n': R['mc_17'][0], 'mean_bps': R['mc_17'][1], 'pval_raw': R['mc_17'][2]},\n"
            "        {'anchor_day': 3,  'n': R['mc_03'][0], 'mean_bps': R['mc_03'][1], 'pval_raw': R['mc_03'][2]},\n"
            "        {'anchor_day': 24, 'n': R['mc_24'][0], 'mean_bps': R['mc_24'][1], 'pval_raw': R['mc_24'][2]},\n"
            "    ])\n"
            "    mc['pval_bonferroni'] = (mc['pval_raw'] * len(mc)).clip(upper=1.0)\n"
            "\n"
            "print(f'Placebo (Mar 24): {plac_bps:+.1f} bps  vs-March Welch p = {plac_p:.3f}  (flat, as expected)')\n"
            "print()\n"
            "print('Day-of-March sweep (Bonferroni across 5 anchor days):')\n"
            "cols = ['anchor_day','n','mean_bps','pval_raw','pval_bonferroni']\n"
            "print(mc[cols].to_string(index=False))\n"
            "print()\n"
            "print('No anchor day -- including the 17th -- survives Bonferroni. March 17 is not special.')"
        ),
        md(
            "> The placebo a week later is a flat +3.5 bps (p = 0.88). In the day-of-March sweep, "
            "the most extreme slot is the 31st (a *negative* -24.8 bps), and even it does not clear "
            "Bonferroni. March 17's raw p of 0.30 becomes ~1.0 corrected. Specificity: rejected."
        ),
        md(
            "### 4c - Sub-period split: where the 'bump' actually lives\n\n"
            "Stability across regimes is the difference between a signal and an artifact."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = st.subperiod_table(df)\n"
            "else:\n"
            "    sub = pd.DataFrame([\n"
            "        {'period': R['sub_a_lbl'], 'n': R['sub_a_n'], 'mean_bps': R['sub_a_bps'], 'tstat_hac': R['sub_a_t']},\n"
            "        {'period': R['sub_b_lbl'], 'n': R['sub_b_n'], 'mean_bps': R['sub_b_bps'], 'tstat_hac': R['sub_b_t']},\n"
            "        {'period': R['sub_c_lbl'], 'n': R['sub_c_n'], 'mean_bps': R['sub_c_bps'], 'tstat_hac': R['sub_c_t']},\n"
            "    ])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [GREEN if t >= 2 else AMBER if t >= 0 else RED for t in sub['tstat_hac']]\n"
            "ax.bar(sub['period'], sub['tstat_hac'], color=colors)\n"
            "ax.axhline(2.0, c='k', ls='--', lw=0.9, label='t = 2 inference bar')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('HAC t-stat (St. Patrick session)')\n"
            "ax.set_title('Only the 1991-2026 slice clears t=2 -- the rest is negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(sub.to_string(index=False))\n"
            "print()\n"
            "print('A single positive regime flanked by two negative ones = regime artifact.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- event mean +14.8 bps, HAC t = 1.13; vs other March days "
            "Welch p = 0.299. The effect is entirely a 1991-2026 phenomenon (t = 3.27) and "
            "*negative* in both prior regimes.\n"
            "- **Tradability `MIRAGE`** -- one round-trip/yr, ~13 bps net, HAC t = 0.98; "
            "no vehicle beats holding the index.\n"
            "- **Busted** -- the placebo is flat, nothing survives Bonferroni, and the sub-period "
            "split exposes the bump as a regime artifact."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the cost-aware P&L\n\n"
            "Long the index only on the St. Patrick's session; charged a 2 bps round-trip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pnl = st.strategy_pnl(df, one_way_bps=1.0)\n"
            "    print(f\"Gross: {pnl['gross_mean_bps']:+.1f} bps/event  HAC t = {pnl['gross_tstat_hac']:+.2f}\")\n"
            "    print(f\"Net (2 bps round-trip): {pnl['net_mean_bps']:+.1f} bps/event  HAC t = {pnl['net_tstat_hac']:+.2f}\")\n"
            "    print(f\"Buy-and-hold drift: {pnl['bah_daily_ann_pct']:+.1f}%/yr -- earned every day, not one.\")\n"
            "else:\n"
            "    print(f\"Net (2 bps round-trip): {R['net_mean_bps']:+.1f} bps/event  HAC t = {R['net_tstat_hac']:+.2f}\")\n"
            "print()\n"
            "print('One day of exposure a year cannot compound into a portfolio. No trade.')"
        ),
        md(
            "> Net of costs the bump is ~13 bps for a single session and statistically zero "
            "(HAC t = 0.98). Buy-and-hold captures the market's drift on all ~250 trading days; "
            "the holiday trade captures it on one. There is nothing to harvest."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect a St. Patrick bump *when one exists*? Plant one and sweep its size."
        ),
        code(
            "planted = [0.0, 0.5, 1.0, 2.0, 4.0]   # effect in daily-vol units\n"
            "rows = []\n"
            "for eff in planted:\n"
            "    df_s, truth = data.synthetic_daily(n_years=99, stpat_effect=eff, seed=285)\n"
            "    r = st.stpat_comparison(df_s)\n"
            "    rows.append({'effect': eff, 'mean_bps': r['mean_stpat_bps'],\n"
            "                 'hac_t': r['tstat_stpat_hac'], 'welch_p_all': r['pval_welch_all']})\n"
            "res = pd.DataFrame(rows)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [GREEN if t >= 2 else RED for t in res['hac_t']]\n"
            "ax.bar([str(e) for e in res['effect']], res['hac_t'], color=colors)\n"
            "ax.axhline(2.0, c='k', ls='--', lw=0.9, label='t = 2')\n"
            "ax.set_xlabel('Planted St. Patrick effect (daily-vol units)')\n"
            "ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine finds the bump when it is real (green = HAC t >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "print('Null (effect=0) reads ~0; a planted bump lights up. The real tape stays dark.')"
        ),
        md(
            "The engine correctly reads ~zero on the null and crosses t=2 once a real bump is "
            "planted. The real ^GSPC tape -- HAC t = 1.13, concentrated in one regime -- is nowhere "
            "near the detection threshold. The verdict is about the **data**, not the method."
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
    import subprocess
    import sys
    path = os.path.join(HERE, name)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {name}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {name}")
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
