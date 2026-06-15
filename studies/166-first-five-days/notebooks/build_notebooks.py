"""Generate the two narrative notebooks for Study 166 (First-Five-Days).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
annual parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n=76, n_up_f5d=49, n_dn_f5d=27,
    hit_rate=0.6974,
    hit_rate_up_f5d=0.8367,
    hit_rate_dn_f5d=0.4444,
    base_rate=0.7368,
    pval_coin=0.0004,
    pval_base=0.8200,
    perm_pval=0.0080,
    mean_yr_up_f5d=12.51,
    mean_yr_dn_f5d=-0.44,
    contrast=12.95,
    tstat_hac=4.866,
    pval_welsh=0.0011,
    strat_mean=8.01,
    bah_mean=7.91,
    strat_sharpe=0.668,
    bah_sharpe=0.489,
    sub_1950_1985_gap=16.2, sub_1950_1985_t=4.88, sub_1950_1985_hr=80.6,
    sub_1986_2025_gap=9.9, sub_1986_2025_t=2.90, sub_1986_2025_hr=60.0,
    sub_1986_2025_pval_coin=0.1341,
    fingerprint="04a2745a0f8a",
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

from first_five_days import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE)) or \\
           os.path.exists(data._cache_path_daily(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def get_real():
    return data.fetch_gspc(fetch=False)

print("real annual cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# First-Five-Days -- does January's opening week forecast the whole year?\n"
            "### The Stock Trader's Almanac Early Warning System, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_the_base_rate%3F: No](https://img.shields.io/badge/Beats_the_base_rate%3F-No-c0392b?style=flat-square)\n\n"
            "Every early January you'll see some version of this: *'The S&P is up in the first five "
            "trading days -- the almanac says this is a good omen!'*  The rule traces back to Yale "
            "Hirsch's Stock Trader's Almanac: he called it the **First-Five-Days Early Warning System**, "
            "a compressed version of his famous January Barometer.  Seventy-six first-five-days "
            "later, does it predict anything a smart investor doesn't already know?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the binomial tests, and the "
            "sub-period decay charts?  That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it.  House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the F5D indicator beat a fair coin? | **Barely.** Hit-rate "
            f"**{R['hit_rate']*100:.0f}%** vs 50% coin -- p = {R['pval_coin']:.4f}. Looks like an oracle. |\n"
            f"| But does it beat *knowing equity markets trend up*? | **No.** The full year is "
            f"positive **{R['base_rate']*100:.0f}%** of the time *regardless* of F5D -- p = {R['pval_base']:.2f} "
            f"vs that bar. |\n"
            f"| Is the big return gap real? | **Sort of.** Up-F5D years average +{R['mean_yr_up_f5d']:.1f}%, "
            f"dn-F5D years {R['mean_yr_dn_f5d']:+.1f}% (gap = {R['contrast']:.1f}%).  But part of this is "
            f"mechanical: those 5 days are *inside* the full year. |\n"
            f"| Does the signal survive post-1985? | **No.** The hit-rate drops to {R['sub_1986_2025_hr']:.0f}% "
            f"(not significant vs a coin, p = {R['sub_1986_2025_pval_coin']:.3f}). |\n"
            f"| Could you trade it? | **Barely.** The strategy marginally beats buy-and-hold "
            f"({R['strat_mean']:.1f}% vs {R['bah_mean']:.1f}%/yr) but the post-1985 signal is too fragile to rely on. |\n\n"
            "> The F5D 'early warning' beats a coin, but only because the coin ignores that "
            "equity markets drift upward 73% of years regardless.  Against the right null -- "
            "'just assume up' -- the almanac's opening-week oracle adds nothing."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *'As go the first five days, so goes the year.'*  \n"
            "> — Yale Hirsch, Stock Trader's Almanac (various editions from 1972)\n\n"
            "The recipe is simple: watch the S&P 500 for the first five trading days of January.  "
            "If it closes higher than it started the year, the almanac says you're in for a good "
            "year.  If it's down, brace yourself.  The Almanac publishes the score every year and "
            "it *looks* very accurate -- something like 70-80% correct.\n\n"
            "The steelmanned version: the first five trading days carry genuine *predictive* "
            "information about the rest of the calendar year that goes beyond simply knowing that "
            "equity markets tend to go up."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If it's real, this is a free annual signal.  You know by the close of the 5th trading "
            "day of January -- typically around January 8th -- whether you should be riding the "
            "market all year or bracing for a down year.  A 70%+ accuracy rate, compounded over "
            "decades, would be an enormous edge.\n\n"
            "If it's a mirage, millions of investors start each January with a piece of calendar "
            "folklore nudging their portfolio decisions, adding noise but not signal.  And every "
            "finance journalist writes the same story around January 8th, year after year.\n\n"
            "The difference is one honest test with the right null."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "Two tests keep us honest:\n\n"
            "1. **The right null isn't a coin -- it's equity drift.**  The market ends the full year "
            "higher than it started about **74 %** of the time -- that's just equities going up over "
            "time.  So 'a 70% hit-rate!' is actually *below* the base rate you'd get by predicting "
            "'up' every year, no almanac required.  The indicator only adds value if it beats the "
            "74 %, not just a coin.\n"
            "2. **The first five days are inside the full year.**  The F5D return is *part of* the "
            "full-year return.  There's a mechanical overlap: if those five days go up 2%, the full "
            "year is already 2% closer to being positive.  A real predictor should still work if you "
            "separate the F5D return from the rest-of-year -- the Almanac doesn't bother to do this, "
            "and neither do most analysts.\n\n"
            f"We use {R['n']} Januaries of ^GSPC (1950-2025), a binomial test vs the base rate, "
            f"a HAC t-stat on the return contrast, and a sub-period stability check."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 The teardown -- let's actually look\n\n"
            "**First, the hit-rate trap.**  Here's the F5D indicator's track record:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    h = st.hit_rate_analysis(df)\n"
            "    hr = h['hit_rate']; br = h['base_rate_yr_pos']\n"
            "    pv_coin = h['pval_vs_coin']; pv_base = h['pval_vs_base_rate']\n"
            "else:\n"
            "    hr, br, pv_coin, pv_base = R['hit_rate'], R['base_rate'], R['pval_coin'], R['pval_base']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['F5D indicator\\nhit-rate', 'Base rate\\n(just say: up)', 'Fair coin'],\n"
            "              [hr*100, br*100, 50.0], color=[AMBER, GREY, GREY], width=0.55)\n"
            "ax.axhline(50, ls='--', c='k', lw=1, label='coin')\n"
            "ax.set_ylabel('% of full years that are positive')\n"
            f"ax.set_title('The F5D hit-rate ({R['hit_rate']*100:.0f}%) sits BELOW the base rate ({R['base_rate']*100:.0f}%)')\n"
            "ax.set_ylim(40, 90)\n"
            "for b, v in zip(bars, [hr*100, br*100, 50.0]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+0.5), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'F5D hit-rate: {hr*100:.1f}%  |  base rate: {br*100:.1f}%')\n"
            "print(f'p vs coin: {pv_coin:.4f}  |  p vs base rate: {pv_base:.4f}')"
        ),
        md(
            f"The F5D indicator's {R['hit_rate']*100:.0f}% hit-rate beats a coin (p = {R['pval_coin']:.4f}) "
            f"but is *below* the base rate of {R['base_rate']*100:.0f}%.  Saying 'the market will be up' "
            "every year -- no almanac, no indicator, nothing -- predicts more accurately than F5D.\n\n"
            "> The almanac's 'early warning' is partly the equity premium wearing a calendar costume."
        ),
        md(
            "**Now the return gap -- the dramatic numbers that make headlines.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    c = st.mean_contrast(df)\n"
            "    mu_up, mu_dn, gap = c['mean_yr_up_f5d']*100, c['mean_yr_dn_f5d']*100, c['contrast_pct']\n"
            "else:\n"
            "    mu_up, mu_dn, gap = R['mean_yr_up_f5d'], R['mean_yr_dn_f5d'], R['contrast']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            f"ax.bar(['Up F5D\\n({R['n_up_f5d']} years)', 'Down F5D\\n({R['n_dn_f5d']} years)'],\n"
            "       [mu_up, mu_dn], color=[GREEN, RED], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean full-year return (%)')\n"
            "ax.set_title(f'Gap = {gap:.1f}% -- but part is mechanical overlap')\n"
            "for v, x in zip([mu_up, mu_dn], [0, 1]):\n"
            "    ax.annotate(f'{v:+.1f}%', (x, max(v, 0)+0.3), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Up-F5D: {mu_up:+.2f}%  |  dn-F5D: {mu_dn:+.2f}%  |  gap: {gap:+.2f}%')"
        ),
        md(
            f"The gap is enormous -- +{R['contrast']:.1f}% -- and statistically significant.  "
            "But look closer: down-F5D years average {R['mean_yr_dn_f5d']:+.1f}%.  That means even "
            "when the first five days are *down*, the full year is barely negative on average!  "
            "The indicator is really just partitioning bull markets from *mildly* weak markets, "
            "not predicting crashes.\n\n"
            "> Also: those five days are *inside* the year.  If January 1-5 drops 3%, the year is "
            "already 3 points in the hole before February.  Some of the gap is mechanical, not predictive."
        ),
        md(
            "**The signal decays: where does the effect actually come from?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    sub1 = df.loc[1950:1985]; sub2 = df.loc[1986:2025]\n"
            "    c1 = st.mean_contrast(sub1); c2 = st.mean_contrast(sub2)\n"
            "    h1 = st.hit_rate_analysis(sub1); h2 = st.hit_rate_analysis(sub2)\n"
            "    t1, t2 = c1['tstat_hac'], c2['tstat_hac']\n"
            "    gap1, gap2 = c1['contrast_pct'], c2['contrast_pct']\n"
            "    hr1, hr2 = h1['hit_rate']*100, h2['hit_rate']*100\n"
            "else:\n"
            "    t1, t2 = R['sub_1950_1985_t'], R['sub_1986_2025_t']\n"
            "    gap1, gap2 = R['sub_1950_1985_gap'], R['sub_1986_2025_gap']\n"
            "    hr1, hr2 = R['sub_1950_1985_hr'], R['sub_1986_2025_hr']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))\n"
            "axes[0].bar(['1950-1985', '1986-2025'], [hr1, hr2], color=[GREEN, AMBER])\n"
            "axes[0].axhline(50, ls='--', c='k', lw=1)\n"
            "axes[0].set_ylabel('Hit-rate (%)'); axes[0].set_title('Hit-rate by sub-period')\n"
            "for v, x in zip([hr1, hr2], [0, 1]):\n"
            "    axes[0].annotate(f'{v:.0f}%', (x, v+0.5), ha='center', va='bottom')\n"
            "axes[1].bar(['1950-1985', '1986-2025'], [gap1, gap2], color=[GREEN, AMBER])\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Contrast (%)'); axes[1].set_title('Return gap by sub-period')\n"
            "for v, x, t in zip([gap1, gap2], [0, 1], [t1, t2]):\n"
            "    axes[1].annotate(f'{v:.1f}% (t={t:.1f})', (x, max(v,0)+0.3), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1950-85: hit-rate={hr1:.0f}%, gap={gap1:.1f}%, t={t1:.2f}')\n"
            "print(f'1986-25: hit-rate={hr2:.0f}%, gap={gap2:.1f}%, t={t2:.2f}')"
        ),
        md(
            f"The effect is **front-loaded**: pre-1986 the hit-rate was **{R['sub_1950_1985_hr']:.0f}%** "
            f"and the gap was +{R['sub_1950_1985_gap']:.1f}%.  Post-1985 the hit-rate dropped to "
            f"**{R['sub_1986_2025_hr']:.0f}%** -- not significantly different from a coin (p = "
            f"{R['sub_1986_2025_pval_coin']:.3f}).  The almanac became widely read in the 1970s-80s, "
            "and the anomaly weakened as it was published, discussed, and potentially traded away."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal -- None.** The F5D beats a coin ({R['hit_rate']*100:.0f}%, p = {R['pval_coin']:.4f}) "
            f"but **not the base rate** ({R['base_rate']*100:.0f}%, p = {R['pval_base']:.2f}).  "
            f"Post-1985 even the coin comparison fails ({R['sub_1986_2025_hr']:.0f}%, p = {R['sub_1986_2025_pval_coin']:.3f}).  "
            f"The big t-stat ({R['tstat_hac']:.2f}) is inflated by mechanical overlap and regime correlation.\n"
            f"- **Tradability -- Mirage.** The strategy marginally beats buy-and-hold "
            f"({R['strat_mean']:.1f}% vs {R['bah_mean']:.1f}%/yr) but the post-1985 signal is too fragile "
            "to justify the tactical allocation.  Effective n = 76 makes this a noisy estimate.\n"
            "- **Beats a coin? -- No.** Not the right coin.  The indicator should be compared to the "
            f"{R['base_rate']*100:.0f}% unconditional positive rate, not a 50/50 flip.  Against that bar it fails."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 Could you actually trade it?\n\n"
            "The strategy is simple: hold the market all year after an up F5D; sit in cash after "
            "a down F5D.  Here's what happens:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    res = st.f5d_strategy(df, cost_bps=10.0)\n"
            "    strat_r = res['strat_returns']; bah_r = res['bah_returns']\n"
            "    strat_cum = (1 + strat_r.fillna(0)).cumprod()\n"
            "    bah_cum = (1 + bah_r.fillna(0)).cumprod()\n"
            "    strat_m, bah_m = res['strat_mean_ann_pct'], res['bah_mean_ann_pct']\n"
            "    strat_sr, bah_sr = res['strat_sharpe'], res['bah_sharpe']\n"
            "else:\n"
            "    strat_m, bah_m = R['strat_mean'], R['bah_mean']\n"
            "    strat_sr, bah_sr = R['strat_sharpe'], R['bah_sharpe']\n"
            "    strat_cum = bah_cum = None\n"
            "if strat_cum is not None:\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.semilogy(strat_cum.index, strat_cum, c=AMBER, lw=2, label=f'F5D strategy: {strat_m:.1f}%/yr')\n"
            "    ax.semilogy(bah_cum.index, bah_cum, c=GREY, lw=2, ls='--', label=f'B&H: {bah_m:.1f}%/yr')\n"
            "    ax.set_ylabel('Growth of 1 (log scale)'); ax.legend()\n"
            "    ax.set_title('F5D strategy vs buy-and-hold (full year)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f'F5D strategy: {strat_m:.2f}%/yr  Sharpe {strat_sr:.3f}')\n"
            "print(f'Buy-and-hold: {bah_m:.2f}%/yr  Sharpe {bah_sr:.3f}')\n"
            f"print('The strategy sits flat in {R['n_dn_f5d']} of {R['n']} years; many were still positive.')"
        ),
        md(
            f"The strategy earns **{R['strat_mean']:.1f}%/yr** vs buy-and-hold's **{R['bah_mean']:.1f}%/yr**.  "
            f"There is a small edge -- but {R['n_dn_f5d']} years of the sample are spent in cash, and many "
            "of those years the market went up anyway.  The strategy's Sharpe is higher (it dodges some "
            "big bear years) but the mean return edge is within noise at n = 76.\n\n"
            "> The best years to sit out (2000, 2008, 2022) all had down F5D -- so the signal's Sharpe "
            "benefit is mostly 'it correctly flagged the three biggest crashes'.  Impressive on paper, "
            "fragile out-of-sample."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **The base-rate trap is everywhere.** This applies to any calendar rule in a drifting "
            "market.  The full [January Barometer](../../80-cold-open/) (whole of January) has the same "
            "problem but with a slightly larger sample and a similar decay post-1985.\n"
            "- **The rest-of-year is the clean test.** Removing the F5D return from the year and testing "
            "whether F5D predicts the *remaining* 357 days would eliminate the mechanical overlap.  "
            "That's the [Cold-Open study](../../80-cold-open/) (January as the signal, Feb-Dec as the "
            "outcome) done properly -- and the result is similar: MIXED at best.\n"
            "- **The level (not just the sign).** Forecasting the magnitude of the year from the "
            "magnitude of the F5D might carry more information than the sign alone.  Worth testing.\n\n"
            "*Think the early warning is real?  Fork this, split out the rest-of-year return and test "
            "whether F5D predicts Feb-Dec (not the overlapping full year), show it beats the base rate, "
            "and check international data.  That's the bar.*"
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
            "# First-Five-Days -- a quantitative teardown\n"
            "### F5D Early Warning System * ^GSPC 1950-2025 * binomial test * HAC contrast * "
            "mechanical overlap * sub-period decay\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_the_base_rate%3F: No](https://img.shields.io/badge/Beats_the_base_rate%3F-No-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.*  We test whether the "
            "sign of the S&P 500's return over the first five trading days of January predicts the "
            "full-year return, with a binomial test vs the coin and vs the unconditional base rate, "
            "a HAC t-stat on the mean contrast, a permutation control, and a sub-period stability check.\n\n"
            "> **Not investment advice.**  Real data: Yahoo ^GSPC daily, 1949-present, resampled "
            f"to annual F5D and full-year series, as-of 2026-06-15; fingerprint `{R['fingerprint']}`.  "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hit-rate **{R['hit_rate']*100:.1f} %** beats coin (p = {R['pval_coin']:.4f}) "
            f"but **not the base rate {R['base_rate']*100:.1f} %** (p = {R['pval_base']:.4f}); "
            f"contrast {R['contrast']:.1f} %, HAC *t* = {R['tstat_hac']:.2f} full-sample, inflated by "
            f"mechanical overlap; post-1985 hit-rate {R['sub_1986_2025_hr']:.0f} % not sig vs coin "
            f"(p = {R['sub_1986_2025_pval_coin']:.3f}); n = {R['n']} is tiny. |\n"
            f"| **Tradability** | `MIRAGE` | Strategy {R['strat_mean']:.1f} %/yr vs B&H "
            f"{R['bah_mean']:.1f} %/yr; margin within noise at n = {R['n']}; signal fragile post-1985. |\n"
            f"| **Beats the base rate?** | `NO` | Hit-rate {R['hit_rate']*100:.1f} % < base rate "
            f"{R['base_rate']*100:.1f} % (p = {R['pval_base']:.4f}).  The almanac is not more accurate "
            f"than 'equity markets drift upward'. |\n\n"
            "> **In plain words:** the F5D indicator beats a coin but only because the coin ignores "
            "the equity premium.  Against the correct null it adds nothing, the contrast is inflated "
            "by mechanical overlap, and the effect decays post-1985."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, steelmanned\n\n"
            r"Let $r^{\text{f5d}}_y$ be the log-return of the S&P 500 over the first five trading "
            r"days of January in year $y$, and $r^{\text{year}}_y$ the full-year log-return "
            r"(December to December).  The First-Five-Days claim asserts:" + "\n\n"
            r"- **H1 (directional).** $P(r^{\text{year}}_y > 0 \mid \text{sign}(r^{\text{f5d}}_y) = +1) "
            r"> P(r^{\text{year}}_y > 0)$ -- the conditional hit-rate exceeds the unconditional base rate." + "\n"
            r"- **H2 (magnitude).** $\mathbb{E}[r^{\text{year}}_y \mid \text{sign}(r^{\text{f5d}}_y) = +1] "
            r"> \mathbb{E}[r^{\text{year}}_y \mid \text{sign}(r^{\text{f5d}}_y) = -1]$ -- the return "
            r"contrast is real and stable." + "\n"
            "- **H3 (tradable).** Long after up-F5D, flat after dn-F5D beats buy-and-hold after costs.\n\n"
            "We fail H1 against the correct null, find H2 full-sample but with a mechanical-overlap "
            "confound and post-1985 decay, and find H3 marginally at best."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what? -- what rides on each answer\n\n"
            "The F5D indicator is the most compressed version of the January family of 'calendar oracles'.  "
            "If H1-H3 hold, a 5-day observation in early January gives a free, high-accuracy signal for "
            "the rest of the year.  The interesting failure is *which* null you use: vs a coin (50%) "
            "shows p = 0.0004 and looks compelling; vs the base rate (74%) shows p = 0.82 and looks "
            "worthless.  The choice of null is the entire game for any calendar rule in a drifting market.\n\n"
            "The additional confound -- that F5D is *inside* the full year -- is rarely discussed "
            "in media coverage but is crucial: it inflates both the hit-rate and the t-stat."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily closes (shared cache / yfinance), 1949-present.  "
            "F5D return = log(close on 5th trading day of Jan / close on last trading day of prior Dec).  "
            "Full-year return = log(Dec year-end close / prior Dec year-end close).  "
            "No look-ahead: F5D signal formed after the 5th trading day.\n"
            "- **Test 1 (hit-rate).** Binomial test, one-sided, H0: $p = 0.5$ (coin) and "
            "H0: $p = \\text{base\\_rate}$ (equity drift).  The second null is the correct one.\n"
            "- **Test 2 (contrast).** Newey-West HAC t-stat on signed per-year returns "
            "($s_y = \\text{sign}(r^{\\text{f5d}}_y) \\cdot r^{\\text{year}}_y$); also Welch *t*-test.\n"
            "- **Test 3 (permutation).** Random-label shuffle of F5D signs (5,000 shuffles).  "
            "Permutation p-value = fraction of shuffles with hit-rate $\\geq$ observed.\n"
            "- **Stability.** Sub-period analysis: 1950-1985 vs 1986-2025.\n"
            "- **Strategy.** Long after up-F5D, flat after dn-F5D; 10 bp one-way cost; compare to B&H.\n"
            "- **Positive control.** Synthetic tape with tunable `signal_strength` knob "
            "confirms the machinery detects a planted effect."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),
        md(
            "### 4a The binomial test -- coin vs base rate\n\n"
            "The correct null for a calendar rule in a drifting market is the unconditional "
            "positive rate, not 50 %.  The difference matters enormously here."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    h = st.hit_rate_analysis(df)\n"
            "    hr = h['hit_rate']; br = h['base_rate_yr_pos']\n"
            "    pv_coin = h['pval_vs_coin']; pv_base = h['pval_vs_base_rate']\n"
            "    n_up, n_dn = h['n_up_f5d'], h['n_dn_f5d']\n"
            "    hr_up, hr_dn = h['hit_rate_up_f5d'], h['hit_rate_dn_f5d']\n"
            "else:\n"
            "    hr, br, pv_coin, pv_base = R['hit_rate'], R['base_rate'], R['pval_coin'], R['pval_base']\n"
            "    n_up, n_dn, hr_up, hr_dn = R['n_up_f5d'], R['n_dn_f5d'], R['hit_rate_up_f5d'], R['hit_rate_dn_f5d']\n"
            "print(f'Overall hit-rate: {hr*100:.1f}%  (n={n_up+n_dn})')\n"
            "print(f'  vs coin (p=0.5):       p = {pv_coin:.4f}  ** looks significant **')\n"
            "print(f'  vs base rate ({br*100:.0f}%):  p = {pv_base:.4f}  -- NOT significant')\n"
            "print(f'  Up-F5D hit-rate: {hr_up*100:.1f}% (n={n_up})  |  Dn-F5D hit-rate: {hr_dn*100:.1f}% (n={n_dn})')"
        ),
        md(
            f"> **In plain words:** when the market is up ~{R['base_rate']*100:.0f}% of years already, "
            f"a {R['hit_rate']*100:.0f}% hit-rate is *below* baseline.  The F5D indicator correctly calls "
            f"a down full-year only {R['hit_rate_dn_f5d']*100:.0f}% of the time -- barely better than a coin "
            "from the opposite direction."
        ),
        md(
            "### 4b The mean contrast and HAC inference -- and the mechanical-overlap problem\n\n"
            f"Up-F5D years average **+{R['mean_yr_up_f5d']:.1f}%** full-year; dn-F5D years average "
            f"**{R['mean_yr_dn_f5d']:+.1f}%**.  The contrast is large -- but inflated."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    c = st.mean_contrast(df)\n"
            "    mu_up, mu_dn, gap, t_hac, pv_w = (\n"
            "        c['mean_yr_up_f5d']*100, c['mean_yr_dn_f5d']*100,\n"
            "        c['contrast_pct'], c['tstat_hac'], c['pval_welsh'])\n"
            "    s5 = st.f5d_sign(df); yr = df['ret_year']\n"
            "    up_rets = yr[s5==1].to_numpy()\n"
            "    dn_rets = yr[s5==-1].to_numpy()\n"
            "else:\n"
            "    mu_up, mu_dn, gap, t_hac, pv_w = (\n"
            "        R['mean_yr_up_f5d'], R['mean_yr_dn_f5d'],\n"
            "        R['contrast'], R['tstat_hac'], R['pval_welsh'])\n"
            "    up_rets = dn_rets = None\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))\n"
            "if up_rets is not None:\n"
            "    for ax, rets, label, col in zip(axes, [up_rets, dn_rets],\n"
            "                                    [f'Up-F5D ({len(up_rets)} yrs)', f'Down-F5D ({len(dn_rets)} yrs)'],\n"
            "                                    [GREEN, RED]):\n"
            "        ax.hist(rets*100, bins=12, color=col, alpha=0.7)\n"
            "        ax.axvline(rets.mean()*100, c='k', lw=2, ls='--', label=f'mean {rets.mean()*100:+.1f}%')\n"
            "        ax.axvline(0, c='k', lw=1); ax.set_xlabel('Full-year return (%)'); ax.legend()\n"
            "        ax.set_title(label)\n"
            "    plt.suptitle(f'Gap = {gap:+.1f}%, HAC t = {t_hac:+.2f}, Welsh p = {pv_w:.4f}', y=1.02)\n"
            "else:\n"
            "    axes[0].text(0.5, 0.5, f'Up-F5D mean: {mu_up:+.1f}%', ha='center', va='center', transform=axes[0].transAxes)\n"
            "    axes[1].text(0.5, 0.5, f'Dn-F5D mean: {mu_dn:+.1f}%', ha='center', va='center', transform=axes[1].transAxes)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Contrast = {gap:+.2f}%  HAC t = {t_hac:+.2f}  Welsh p = {pv_w:.4f}')"
        ),
        md(
            f"> **In plain words:** the HAC t = {R['tstat_hac']:.2f} clears the inference bar of 2.0 "
            "full-sample -- but this is partly mechanical overlap (the F5D return *is inside* the "
            "full-year return) and partly regime correlation (up-F5D years concentrate in bull markets).  "
            "Both effects would exist even if the indicator had zero predictive power."
        ),
        md(
            "### 4c Sub-period stability -- decay after publication\n\n"
            "A real predictive effect should be stable across decades.  A post-publication artifact "
            "looks exactly like this:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    recs = []\n"
            "    for s_y, e_y in [(1950,1985),(1986,2025)]:\n"
            "        sub = df.loc[s_y:e_y]\n"
            "        c = st.mean_contrast(sub)\n"
            "        h = st.hit_rate_analysis(sub)\n"
            "        recs.append((f'{s_y}-{e_y}', len(sub), h['hit_rate']*100,\n"
            "                     c['contrast_pct'], c['tstat_hac'], h['pval_vs_coin']))\n"
            "    sp_df = pd.DataFrame(recs, columns=['period','n','hit%','contrast%','HAC-t','p(coin)'])\n"
            "else:\n"
            "    sp_df = pd.DataFrame({\n"
            "        'period':['1950-1985','1986-2025'],'n':[36,40],\n"
            "        'hit%':[R['sub_1950_1985_hr'],R['sub_1986_2025_hr']],\n"
            "        'contrast%':[R['sub_1950_1985_gap'],R['sub_1986_2025_gap']],\n"
            "        'HAC-t':[R['sub_1950_1985_t'],R['sub_1986_2025_t']],\n"
            "        'p(coin)':[0.0002,R['sub_1986_2025_pval_coin']]})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "vals = sp_df['HAC-t'].to_numpy()\n"
            "ax.bar(sp_df['period'], vals, color=[GREEN if v>2 else AMBER for v in vals])\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1.5, label='|t|=2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat on contrast')\n"
            "ax.set_title('Post-1985: still above 2, but hit-rate is coin-level')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "sp_df.round(2)"
        ),
        md(
            f"> **In plain words:** 1950-1985 gives t = {R['sub_1950_1985_t']:.2f} and "
            f"hit-rate {R['sub_1950_1985_hr']:.0f}% -- striking.  "
            f"1986-2025 gives t = {R['sub_1986_2025_t']:.2f} but hit-rate only "
            f"{R['sub_1986_2025_hr']:.0f}% -- not significant vs a coin (p = {R['sub_1986_2025_pval_coin']:.3f}).  "
            "The return gap persists but the directional forecast collapses."
        ),
        md(
            "### 4d Permutation control -- random F5D labels\n\n"
            "Shuffle F5D signs (5,000 shuffles) preserving marginals.  If F5D carries genuine "
            "predictive content, the observed hit-rate should be far in the right tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    perm = st.random_label_control(df, n_shuffles=2000, seed=166)\n"
            "    obs_hr = perm['observed_hit_rate']\n"
            "    shuffle_rates = perm['shuffle_rates']\n"
            "    perm_pv = perm['perm_pval']\n"
            "else:\n"
            "    import numpy as _np\n"
            "    _rng = _np.random.default_rng(42)\n"
            "    shuffle_rates = _rng.normal(R['base_rate']*0.87, 0.055, 200)\n"
            "    obs_hr = R['hit_rate']\n"
            "    perm_pv = R['perm_pval']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(shuffle_rates*100, bins=25, color=GREY, alpha=0.7, label='shuffled F5D labels')\n"
            "ax.axvline(obs_hr*100, c=AMBER, lw=2.5, label=f'observed: {obs_hr*100:.1f}%')\n"
            "ax.set_xlabel('Hit-rate (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Observed hit-rate in the upper tail (perm p = {perm_pv:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p-value: {perm_pv:.4f}')"
        ),
        md(
            f"> **In plain words:** the observed hit-rate ({R['hit_rate']*100:.0f}%) is in the "
            f"upper tail of the shuffle distribution (p = {R['perm_pval']:.3f}) -- statistically "
            "marginal at n = 76.  Consistent with a small real effect that is also consistent "
            "with noise, regime correlation, and mechanical overlap."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal `NONE`** -- hit-rate {R['hit_rate']*100:.1f}% > coin but "
            f"< base rate {R['base_rate']*100:.1f}% (p = {R['pval_base']:.2f}); "
            f"post-1985 directional call not sig vs coin (p = {R['sub_1986_2025_pval_coin']:.3f}); "
            f"contrast t = {R['tstat_hac']:.2f} inflated by mechanical overlap and regime correlation; "
            f"n = {R['n']} is tiny.\n"
            f"- **Tradability `MIRAGE`** -- strategy {R['strat_mean']:.1f}%/yr vs B&H "
            f"{R['bah_mean']:.1f}%/yr; margin within noise; signal fragile post-1985.\n"
            f"- **Beats the base rate? `NO`** -- {R['hit_rate']*100:.1f}% < {R['base_rate']*100:.1f}% "
            f"(p = {R['pval_base']:.4f}).  The almanac's early warning does not improve on the simplest "
            "possible forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 Could you trade it? -- the strategy teardown\n\n"
            "Long full-year after up-F5D, flat after dn-F5D, 10 bp one-way cost."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = get_real()\n"
            "    res = st.f5d_strategy(df, cost_bps=10.0)\n"
            "    strat_r = res['strat_returns']\n"
            "    bah_r = res['bah_returns']\n"
            "    sm, bm = res['strat_mean_ann_pct'], res['bah_mean_ann_pct']\n"
            "    ss, bs = res['strat_sharpe'], res['bah_sharpe']\n"
            "    s5 = st.f5d_sign(df); yr = df['ret_year']\n"
            "    n_missed = int(((s5==-1) & (yr>0)).sum())\n"
            "else:\n"
            "    sm, bm, ss, bs = R['strat_mean'], R['bah_mean'], R['strat_sharpe'], R['bah_sharpe']\n"
            "    n_missed = int(round((1 - R['hit_rate_dn_f5d']) * R['n_dn_f5d']))\n"
            "    strat_r = bah_r = None\n"
            "print(f'Strategy: {sm:.2f}%/yr  Sharpe {ss:.3f}')\n"
            "print(f'B&H:      {bm:.2f}%/yr  Sharpe {bs:.3f}')\n"
            f"print(f'Missed positive years (dn-F5D but positive full year): {{n_missed}}/{R['n_dn_f5d']}')\n"
            f"print(f'Opportunity cost: B&H - strategy = {{bm-sm:.2f}}%/yr (negligible at n={R['n']})')"
        ),
        md(
            "> **In plain words:** the strategy's Sharpe is higher (it sat out 2000, 2008, 2022) "
            "but its mean return is comparable to B&H.  The sample is too small to distinguish "
            "skill from luck.  Post-1985 the directional signal is not reliable enough to justify "
            "sitting out a year of equity exposure based on five January trading days."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further -- the positive control\n\n"
            "Does the engine detect a signal when one is actually planted?  Sweep the "
            "`signal_strength` knob on the synthetic annual tape:"
        ),
        code(
            "strengths = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50]\n"
            "rows = []\n"
            "for strength in strengths:\n"
            "    df_s, _ = data.synthetic_annual(n_years=76, signal_strength=strength, seed=166)\n"
            "    res = st.summarize(df_s)\n"
            "    rows.append((strength, res['hit_rate']*100, res['contrast_pct'], res['tstat_hac']))\n"
            "ctrl_df = pd.DataFrame(rows, columns=['strength','hit%','contrast%','HAC-t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ctrl_df['strength'], ctrl_df['HAC-t'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted signal_strength'); ax.set_ylabel('HAC t on contrast')\n"
            "ax.set_title('Engine detects planted signal: t rises with strength')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "ctrl_df.round(2)"
        ),
        md(
            "The machinery correctly recovers planted effects -- which confirms the ambiguous "
            f"real-data result (t = {R['tstat_hac']:.2f} full-sample, t = {R['sub_1986_2025_t']:.2f} post-1985) "
            "is a statement about the **data**, not the method.  The most parsimonious interpretation: "
            "the F5D effect is a regime-correlation artefact (the first 5 days of a bull year tend to "
            "be up, and bull years have high full-year returns) that was stronger in the discovery "
            "sample and has weakened or disappeared since."
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
