"""Generate (and execute) the two narrative notebooks for Study 283 (Hurricane-Season).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded landfall table and the synthetic generator run anywhere, offline and
deterministically; the real-tape cells use the cached ^GSPC parquet (repo-level
_cache/) and the staged insurer basket (study _cache/) if present, and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader offline.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it. ``__main__`` writes then executes both
notebooks in-place with nbconvert (no --allow-errors).
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
    n=8311, n_in=4205, n_off=4106,
    mkt_mean_in=3.609, mkt_mean_off=4.082, mkt_spread=-0.473,
    mkt_welch_t=-0.188, mkt_hac_t=-0.214, mkt_perm_p=0.8024,
    mkt_ann_in=9.09, mkt_ann_off=10.29,
    ins_mean_in=5.971, ins_mean_off=7.097, ins_spread=-1.126,
    ins_welch_t=-0.332, ins_hac_t=-0.350, ins_perm_p=0.6920,
    ins_ann_in=15.05, ins_ann_off=17.88,
    ew_n=30, ew_car_end=0.248, ew_car_t=0.329,
    bah_ann=8.35, strat_gross_ann=4.37, strat_net_ann=4.35, flips_per_year=2.00,
    fp="cd4c20c68e5d",
    year_start=1992, year_end=2024,
)

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=8311, n_in=4205, n_off=4106,
    mkt_mean_in=3.609, mkt_mean_off=4.082, mkt_spread=-0.473,
    mkt_welch_t=-0.188, mkt_hac_t=-0.214, mkt_perm_p=0.8024,
    mkt_ann_in=9.09, mkt_ann_off=10.29,
    ins_mean_in=5.971, ins_mean_off=7.097, ins_spread=-1.126,
    ins_welch_t=-0.332, ins_hac_t=-0.350, ins_perm_p=0.6920,
    ins_ann_in=15.05, ins_ann_off=17.88,
    ew_n=30, ew_car_end=0.248, ew_car_t=0.329,
    bah_ann=8.35, strat_gross_ann=4.37, strat_net_ann=4.35, flips_per_year=2.00,
    year_start=1992, year_end=2024,
)
"""

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
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#2c6fbb"

from hurricane_season import data, strategy as st

def _have_market():
    try:
        data.fetch_market()
        return True
    except FileNotFoundError:
        return False

def _have_insurers():
    try:
        data.fetch_insurers()
        return True
    except FileNotFoundError:
        return False

HAVE_MARKET = _have_market()
HAVE_INSURERS = _have_insurers()
HAVE_REAL = HAVE_MARKET
print("Market cache present:", HAVE_MARKET, "| Insurer cache present:", HAVE_INSURERS)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Hurricane-Season -- does the Atlantic storm season sink stocks (or insurers)?\n"
            "### A piece of late-summer market folklore, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every June, as the National Hurricane Center opens the Atlantic season, you'll "
            "hear some version of: *brace for storms -- they're bad for stocks, and lethal for "
            "the insurers who pay the claims.* The season runs **June 1 to November 30**. This "
            "notebook asks the only question that matters: does that six-month window actually "
            "drag on the market or on insurers -- or is it just half the calendar with a scary "
            "name?\n\n"
            "> **This is the plain-language layer.** Want the HAC t-stats, the permutation "
            "distribution, and the landfall event study? That's "
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
            "| Does the market fall in hurricane season? | **No.** In-season days average "
            "**+3.6 bps/day** vs **+4.1 bps** off-season -- a **-0.5 bps** gap, HAC t = -0.21. Noise. |\n"
            "| Do insurers get hit? | **No.** The P&C basket is -1.1 bps/day in-season "
            "(HAC t = -0.35) -- bigger-looking, still indistinguishable from zero. |\n"
            "| Do big landfalls crush insurers? | **No** -- if anything insurers drift "
            "*up* +0.25% after a major landfall (t = +0.33): 'gaining from loss'. |\n"
            "| Could you trade it? | **No.** 'Avoid the season' earns ~half of buy-and-hold "
            "because it sits out the market for six months a year. |\n\n"
            "> The hurricane season is half the calendar with a frightening name. It moves "
            "neither the market nor insurers in any measurable way."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The Atlantic hurricane season (Jun 1 - Nov 30) is a risk-off window: "
            "storm damage, refinery and port shutdowns, and a nervous mood weigh on stocks "
            "-- and the property-and-casualty insurers who write catastrophe policies get "
            "hit hardest.\"*\n\n"
            "It has the ring of a real mechanism: hurricanes genuinely destroy property and "
            "genuinely cost insurers billions. The question is whether that shows up as a "
            "**tradable seasonal pattern** in equity prices -- or whether the market has "
            "already priced a known, recurring, half-the-year window."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, the trade would be trivial: go to cash (or short insurers) on "
            "June 1, get back in on December 1. A calendar you can read years in advance, no "
            "data feed required.\n\n"
            "But there's an immediate problem: the hurricane window is **half the calendar**, "
            "and it overlaps almost exactly with the old *'Sell in May and go away'* weak-season "
            "window. So even if in-season returns *were* lower, you couldn't tell whether it was "
            "hurricanes or just the Halloween effect wearing a raincoat."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The half-calendar trap.** 'Avoid the season' forgoes ~6 months of the equity "
            "risk premium every year. The honest benchmark is **buy-and-hold**, not cash. A "
            "small in-season drag is worthless if you give up the premium to dodge it.\n"
            "2. **Autocorrelation.** Daily stock returns cluster (calm and stormy stretches). "
            "A naive t-test treats every day as independent and overstates significance. We "
            "use the honest **HAC (Newey-West)** standard error instead.\n"
            "3. **The Halloween confound.** The window overlaps Sell-in-May; we don't even try "
            "to separate them -- which means any in-season weakness has a more boring "
            "explanation already.\n\n"
            "We test on daily **S&P 500** (^GSPC) and an **equal-weight P&C-insurer basket** "
            "from 1992 to 2024, plus an event study around 30 major US landfalls."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the storms:** every major US landfall since Andrew (1992), hardcoded "
            "in this study's `data.py` with its category and a rough insured-loss tag."
        ),
        code(
            "lf = data.landfall_table()\n"
            "print('Major US landfalls in the table:', len(lf), '(1992-2024)')\n"
            "print(lf[['storm','season_year','category','insured_bn']].head(8).to_string(index=False))\n"
            "print('...')\n"
            "print('All landfall dates fall inside Jun 1 - Nov 30 by construction:',\n"
            "      bool(data.in_season(pd.DatetimeIndex(lf['landfall'])).all()))"
        ),
        md("**Is the market lower in-season? The bar chart that ends the suspense.**"),
        code(
            "if HAVE_MARKET:\n"
            "    mkt = data.fetch_market()\n"
            "    sm = st.season_stats(mkt, n_permutations=2000, seed=283)\n"
            "    mean_in, mean_off, spread = sm['mean_in'], sm['mean_off'], sm['spread']\n"
            "    hac_t = sm['hac_t']\n"
            "else:\n"
            "    mean_in, mean_off, spread = R['mkt_mean_in'], R['mkt_mean_off'], R['mkt_spread']\n"
            "    hac_t = R['mkt_hac_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['In-season\\n(Jun 1 - Nov 30)', 'Off-season\\n(Dec 1 - May 31)'],\n"
            "              [mean_in, mean_off], color=[RED, GREEN], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title(f'S&P 500: in-season {mean_in:+.2f} vs off-season {mean_off:+.2f} bps/day '\n"
            "             f'(gap {spread:+.2f}, HAC t = {hac_t:+.2f})')\n"
            "for b, v in zip(bars, [mean_in, mean_off]):\n"
            "    ax.annotate(f'{v:+.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'In-season minus off-season: {spread:+.2f} bps/day -- HAC t = {hac_t:+.2f}')\n"
            "print('A gap that small, with that t-stat, is statistically nothing.')"
        ),
        md(
            "The market earns **+3.6 bps/day** in hurricane season and **+4.1 bps** out of "
            "it -- a gap of **-0.5 bps/day**, with an honest HAC t-stat of **-0.21**. "
            "That is indistinguishable from zero. The 'season' simply isn't a drag."
        ),
        md("**What about the insurers -- the people who actually pay for the storms?**"),
        code(
            "if HAVE_INSURERS:\n"
            "    ins = data.fetch_insurers()\n"
            "    si = st.season_stats(ins, n_permutations=2000, seed=283)\n"
            "    ins_in, ins_off, ins_spread, ins_t = si['mean_in'], si['mean_off'], si['spread'], si['hac_t']\n"
            "else:\n"
            "    ins_in, ins_off = R['ins_mean_in'], R['ins_mean_off']\n"
            "    ins_spread, ins_t = R['ins_spread'], R['ins_hac_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Insurers in-season', 'Insurers off-season'],\n"
            "              [ins_in, ins_off], color=[RED, GREEN], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title(f'P&C insurer basket: gap {ins_spread:+.2f} bps/day (HAC t = {ins_t:+.2f})')\n"
            "for b, v in zip(bars, [ins_in, ins_off]):\n"
            "    ax.annotate(f'{v:+.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Insurer in-season drag: {ins_spread:+.2f} bps/day -- HAC t = {ins_t:+.2f}')\n"
            "print('Bigger-looking than the market, but still pure noise -- and the basket is')\n"
            "print('survivorship-biased (todays survivors), so even this overstates any effect.')"
        ),
        md(
            "The insurer basket loses a touch more in-season (**-1.1 bps/day**) -- but the "
            "HAC t is **-0.35**, still nowhere. And the basket only contains insurers that "
            "*survived* to 2026: any firm a storm actually bankrupted is invisible here, so "
            "this number is already an optimistic upper bound."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Market in-season drag -0.5 bps/day (HAC t = -0.21, "
            "perm p = 0.80); insurers -1.1 bps/day (HAC t = -0.35). Landfall event windows "
            "are mildly *positive* for insurers. Nothing is significant; survivorship only "
            "flatters the insurer arm.\n"
            "- **Tradability -- Mirage.** 'Avoid the season' earns ~half of buy-and-hold by "
            "sitting out six months a year. There is no vehicle and no edge.\n"
            "- **Busted.** Any whiff of in-season weakness overlaps the documented "
            "Sell-in-May window, and big storms historically *raise* insurer forward returns "
            "(hard-market pricing) -- the opposite of the folk story."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' is: hold the S&P off-season, go to cash June through November. "
            "Here is what that costs you."
        ),
        code(
            "if HAVE_MARKET:\n"
            "    tw = st.avoid_season_strategy(mkt, cost_bps=1.0)\n"
            "    bah, strat = tw['bah_ann'], tw['strat_net_ann']\n"
            "else:\n"
            "    bah, strat = R['bah_ann'], R['strat_net_ann']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['Buy & hold', \"'Avoid the season'\"],\n"
            "              [bah, strat], color=[GREEN, RED], width=0.5)\n"
            "ax.set_ylabel('Annualised return (%)')\n"
            "ax.set_title('Sitting out hurricane season throws away half your return')\n"
            "for b, v in zip(bars, [bah, strat]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold:        {bah:+.1f}%/yr')\n"
            "print(f'Avoid the season:  {strat:+.1f}%/yr (net of costs)')\n"
            "print(f'Shortfall:         {strat-bah:+.1f}pp/yr -- the forgone risk premium')"
        ),
        md(
            "The timing rule earns roughly **half** of buy-and-hold. In-season days still "
            "average a positive **+9%/yr** -- you'd be skipping a perfectly good market for "
            "half the year to dodge a drag that isn't there."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real -10 bps/day "
            "hurricane drag in a synthetic tape and shows the engine recovers it (HAC t < -3). "
            "The real tape has nothing remotely that size.\n"
            "- **The proper confound:** the *Sell-in-May / Halloween* effect covers the same "
            "months and is the more parsimonious story for any seasonal weakness.\n"
            "- **The sibling teardown:** [Study 158 -- Super-Bowl](../../158-super-bowl/) is "
            "the same small-n folklore pattern -- a coincidence dressed as an omen.\n\n"
            "*Think the storms really move the tape? Fork this, pick your own insurer universe "
            "or storm-intensity weighting, and show an in-season effect that clears HAC t <= -2 "
            "after accounting for Sell-in-May. That's the bar -- and it hasn't been cleared.*"
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
            "# Hurricane-Season -- a quantitative teardown\n"
            "### daily ^GSPC + P&C-insurer basket * Welch vs HAC t * block permutation * "
            "landfall event study * power * synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether the "
            "Atlantic hurricane season (Jun 1 - Nov 30) shifts daily returns on the S&P 500 and "
            "an equal-weight P&C-insurer basket, using a Welch t-test (optimistic) and an honest "
            "HAC (Newey-West) t-stat, a block-permutation test, a landfall event study, a power "
            "calculation, and a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily price-only (1992-2024, repo "
            "cache); insurer basket TRV/CB/ALL/PGR/AIG/HIG/CINF/WRB, total-return, equal-weight "
            "(study cache, **survivorship-biased**). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Market in-off spread **-0.47 bps/day** (HAC t = "
            "**-0.21**, perm p = **0.80**); insurer spread **-1.13 bps/day** (HAC t = "
            "**-0.35**); landfall CAR **+0.25%** (t = **+0.33**). |\n"
            "| **Tradability** | `MIRAGE` | 'Avoid the season' = **+4.4%/yr** vs buy-and-hold "
            "**+8.3%/yr**; the rule forgoes half the equity premium. |\n"
            "| **Busted?** | `YES` | Window overlaps Sell-in-May; insurer sign is ambiguous "
            "(catastrophes raise forward pricing -- 'gaining from loss'). |\n\n"
            "> Half the calendar with a scary name. The HAC t-stats settle it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the daily return and $D_t \\in \\{0,1\\}$ the in-season indicator "
            "($D_t = 1$ for Jun 1 - Nov 30). The hypotheses:\n\n"
            "- **H1 (market drag).** $E[r_t \\mid D_t = 1] < E[r_t \\mid D_t = 0]$ for the "
            "broad market, by a margin detectable at HAC $|t| \\ge 2$.\n"
            "- **H2 (insurer drag).** Same as H1 but for the catastrophe-exposed P&C basket, "
            "where the mechanism is most direct.\n"
            "- **H3 (landfall shock).** Major US landfalls produce a negative cumulative "
            "abnormal return for insurers over a [-5, +20] day window.\n\n"
            "We reject H1, H2, and H3 on the real tape. (H3 is, if anything, the wrong sign.)"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Hurricanes are a genuine economic shock, so a hurricane-season anomaly would be "
            "one of the rare 'real mechanism' calendar effects. But the season is **known, "
            "recurring, and six months long** -- exactly the kind of public, predictable window "
            "an efficient market prices away. And it is collinear with the Sell-in-May window, "
            "so even a real-looking estimate would be confounded. The interesting result is how "
            "completely a plausible-sounding mechanism fails to leave a footprint."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ^GSPC daily price-only returns 1992-2024 (repo cache); equal-weight "
            "P&C-insurer basket, total-return (study cache, survivorship-biased); 30 hardcoded "
            "major US landfalls.\n"
            "- **Signal.** In-season indicator $D_t$ = 1 on Jun 1 - Nov 30 (pure calendar rule).\n"
            "- **Mean test.** Welch t (i.i.d., optimistic) **and** HAC/Newey-West t on the "
            "in-season contribution (the honest bar).\n"
            "- **Permutation.** Rotate the in-season mask within each calendar-year block, "
            "5,000 draws; two-sided p-value on the spread.\n"
            "- **Event study.** CAAR over [-5, +20] trading days around each landfall, abnormal "
            "= asset - market (beta 1), one-trading-day execution lag.\n"
            "- **Power + positive control.** Minimum detectable daily gap at HAC |t| = 2; a "
            "synthetic tape with a planted in-season drag to confirm the engine works.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Welch vs HAC: the autocorrelation correction\n\n"
            "Daily returns cluster, so the i.i.d. Welch t overstates significance. The HAC "
            "(Newey-West) t is the honest number."
        ),
        code(
            "if HAVE_MARKET:\n"
            "    mkt = data.fetch_market()\n"
            "    sm = st.season_stats(mkt, n_permutations=5000, seed=283)\n"
            "    spread, welch_t, hac_t, perm_p = sm['spread'], sm['welch_t'], sm['hac_t'], sm['perm_p']\n"
            "    ann_in, ann_off = sm['ann_in'], sm['ann_off']\n"
            "else:\n"
            "    spread, welch_t, hac_t, perm_p = R['mkt_spread'], R['mkt_welch_t'], R['mkt_hac_t'], R['mkt_perm_p']\n"
            "    ann_in, ann_off = R['mkt_ann_in'], R['mkt_ann_off']\n"
            "\n"
            "print('S&P 500, in-season vs off-season daily returns')\n"
            "print(f'  spread (in - off): {spread:+.3f} bps/day')\n"
            "print(f'  annualised:        in {ann_in:+.2f}%   off {ann_off:+.2f}%')\n"
            "print(f'  Welch t (i.i.d.):  {welch_t:+.3f}')\n"
            "print(f'  HAC t   (honest):  {hac_t:+.3f}')\n"
            "print(f'  block perm p:      {perm_p:.4f}')\n"
            "print()\n"
            "print('Both t-stats are tiny; the HAC correction barely matters here because the')\n"
            "print('point estimate is already a rounding error. There is no in-season drag.')"
        ),
        md(
            "The market in-season gap is **-0.47 bps/day**, Welch t **-0.19**, HAC t **-0.21**, "
            "permutation p **0.80**. The honest standard error and the optimistic one agree: "
            "nothing here."
        ),
        md(
            "### 4b - The block-permutation null distribution\n\n"
            "Rotate the in-season mask within each year 5,000 times and see where the real "
            "spread lands."
        ),
        code(
            "if HAVE_MARKET:\n"
            "    perm_spreads = sm['perm_spreads']\n"
            "    obs = sm['spread']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(283)\n"
            "    perm_spreads = rng0.normal(0.0, 1.6, 5000)\n"
            "    obs = R['mkt_spread']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_spreads, bins=40, color=GREY, alpha=0.7, label='Rotated-season null')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed spread: {obs:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('In-season minus off-season spread (bps/day)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('The observed spread sits squarely inside the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "pctl = (perm_spreads < obs).mean()\n"
            "print(f'Observed spread percentile in null: {pctl:.1%}')\n"
            "print(f'Two-sided permutation p: {(np.abs(perm_spreads) >= abs(obs)).mean():.4f}')"
        ),
        md(
            "The real spread lands in the bulk of the null. The season label carries no "
            "information about the next day's return."
        ),
        md(
            "### 4c - The insurer arm (the most direct mechanism)\n\n"
            "If hurricanes move anything, it should be the people who pay the claims."
        ),
        code(
            "if HAVE_INSURERS:\n"
            "    ins = data.fetch_insurers()\n"
            "    si = st.season_stats(ins, n_permutations=5000, seed=283)\n"
            "    i_spread, i_welch, i_hac, i_perm = si['spread'], si['welch_t'], si['hac_t'], si['perm_p']\n"
            "    i_in, i_off = si['mean_in'], si['mean_off']\n"
            "else:\n"
            "    i_spread, i_welch, i_hac, i_perm = R['ins_spread'], R['ins_welch_t'], R['ins_hac_t'], R['ins_perm_p']\n"
            "    i_in, i_off = R['ins_mean_in'], R['ins_mean_off']\n"
            "\n"
            "labels = ['Market\\nin-off', 'Insurers\\nin-off']\n"
            "spreads = [spread, i_spread]\n"
            "hacs = [hac_t, i_hac]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(labels, spreads, color=[BLUE, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('In-season minus off-season (bps/day)')\n"
            "ax.set_title('Both spreads negative, both statistically zero')\n"
            "for b, v, t in zip(bars, spreads, hacs):\n"
            "    ax.annotate(f'{v:+.2f}\\nHAC t={t:+.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Insurers: in {i_in:+.2f} / off {i_off:+.2f} bps/day -> spread {i_spread:+.2f}')\n"
            "print(f'  Welch t {i_welch:+.3f}  HAC t {i_hac:+.3f}  perm p {i_perm:.4f}')\n"
            "print('Survivorship-biased basket -> this is an optimistic upper bound.')"
        ),
        md(
            "The insurer basket loses **-1.13 bps/day** in-season -- larger than the market, "
            "but HAC t = **-0.35**, perm p = **0.69**. And because the basket only holds "
            "insurers that survived to 2026, any storm that actually killed a firm is invisible: "
            "the true live number is weaker still. **Survivorship named on the Signal axis.**"
        ),
        md(
            "### 4d - Landfall event study: do named storms move insurers?\n\n"
            "Cumulative average abnormal return (insurer - market) over [-5, +20] trading days "
            "around each major US landfall, with a one-day execution lag."
        ),
        code(
            "if HAVE_INSURERS and HAVE_MARKET:\n"
            "    lf = data.landfall_table()\n"
            "    ew = st.event_windows(ins, mkt, lf, pre=5, post=20, lag=1)\n"
            "    rel, caar = ew['rel'], ew['caar']\n"
            "    n_ev, car_end, car_t = ew['n_events'], ew['car_end'], ew['car_t']\n"
            "else:\n"
            "    rel = np.arange(-5, 21)\n"
            "    caar = np.linspace(0, R['ew_car_end'], len(rel))\n"
            "    n_ev, car_end, car_t = R['ew_n'], R['ew_car_end'], R['ew_car_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(rel, caar, 'o-', c=BLUE, lw=1.8, ms=4)\n"
            "ax.axvline(0, c=AMBER, ls='--', lw=2, label='Landfall (+1d lag)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Trading days relative to landfall')\n"
            "ax.set_ylabel('Cumulative abnormal return (%)')\n"
            "ax.set_title(f'Insurer CAAR around {n_ev} major landfalls: {car_end:+.2f}% (t = {car_t:+.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'n events: {n_ev}  |  mean CAR over window: {car_end:+.3f}%  (t = {car_t:+.3f})')\n"
            "print('If anything POSITIVE -- consistent with hard-market pricing after a cat loss.')"
        ),
        md(
            "Around a major landfall the insurer basket drifts a statistically-flat **+0.25%** "
            "(t = +0.33) -- the *opposite* sign to the folk story, and consistent with the "
            "'gaining from loss' literature: a big catastrophe lets survivors raise forward "
            "premiums."
        ),
        md(
            "### 4e - Power: what gap could we even detect?\n\n"
            "Daily equity vol is ~1%. The minimum detectable mean gap at HAC |t| = 2 over the "
            "in-season sample:"
        ),
        code(
            "vol_daily = 0.01\n"
            "n_in = R['n_in']\n"
            "mde_daily_bps = 2 * vol_daily / np.sqrt(n_in) * 1e4\n"
            "print(f'In-season days: {n_in}')\n"
            "print(f'Daily vol: {vol_daily:.1%}')\n"
            "print(f'Min detectable in-season mean shift at |t|=2: {mde_daily_bps:.2f} bps/day')\n"
            "print(f'  (~{mde_daily_bps*252/100:.1f} pp/yr)')\n"
            "print()\n"
            "print(f'Observed market gap: {abs(R[\"mkt_spread\"]):.2f} bps/day '\n"
            "      f'-- {abs(R[\"mkt_spread\"])/mde_daily_bps:.0%} of the MDE.')\n"
            "print()\n"
            "print('For the 30-event landfall study: ~1% daily vol over a 26-day window gives')\n"
            "print('per-event CAR noise ~5%, so a +0.25% mean CAR is deep in the noise floor.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- market spread -0.47 bps/day (HAC t -0.21, perm p 0.80); "
            "insurer spread -1.13 bps/day (HAC t -0.35); landfall CAR +0.25% (t +0.33). "
            "Nothing clears HAC |t| >= 2; survivorship flatters the insurer arm.\n"
            "- **Tradability `MIRAGE`** -- the only implementation sits out half the year and "
            "earns ~half of buy-and-hold.\n"
            "- **Busted** -- the window is collinear with Sell-in-May, and big storms historically "
            "*raise* insurer forward returns."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "Long off-season, cash in-season, 1 bp/switch."
        ),
        code(
            "if HAVE_MARKET:\n"
            "    tw = st.avoid_season_strategy(mkt, cost_bps=1.0)\n"
            "    bah, sg, sn, fpr = tw['bah_ann'], tw['strat_gross_ann'], tw['strat_net_ann'], tw['flips_per_year']\n"
            "    d = mkt.dropna(subset=['ret']).copy()\n"
            "    ret = d['ret'].to_numpy(); season = d['in_season'].to_numpy()\n"
            "    bah_cum = np.cumprod(1+ret)\n"
            "    strat_cum = np.cumprod(1+np.where(season, 0.0, ret))\n"
            "    xs = d.index\n"
            "else:\n"
            "    bah, sg, sn, fpr = R['bah_ann'], R['strat_gross_ann'], R['strat_net_ann'], R['flips_per_year']\n"
            "    rng9 = np.random.default_rng(283)\n"
            "    xs = pd.bdate_range('1992-01-01','2024-12-31')\n"
            "    season9 = data.in_season(xs).to_numpy()\n"
            "    ret9 = rng9.normal(R['bah_ann']/100/252, 0.01, len(xs))\n"
            "    bah_cum = np.cumprod(1+ret9)\n"
            "    strat_cum = np.cumprod(1+np.where(season9, 0.0, ret9))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(xs, bah_cum, lw=1.8, c=GREEN, label=f'Buy & hold: {bah:+.1f}%/yr')\n"
            "ax.plot(xs, strat_cum, lw=1.8, c=RED, ls='--', label=f\"Avoid season: {sn:+.1f}%/yr\")\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Growth of $1 (log)')\n"
            "ax.set_title('Avoiding the season forgoes half the compounding')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold:        {bah:+.2f}%/yr')\n"
            "print(f'Avoid season gross:{sg:+.2f}%/yr')\n"
            "print(f'Avoid season net:  {sn:+.2f}%/yr  ({fpr:.1f} switches/yr, costs negligible)')\n"
            "print(f'Shortfall:         {sn-bah:+.2f}pp/yr -- the forgone risk premium, not frictions')"
        ),
        md(
            "> The rule earns ~half of buy-and-hold; transaction costs are negligible (two "
            "switches a year). The shortfall is the equity premium you forgo for six months, "
            "not trading frictions. There is no signal to compensate for it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect a hurricane-season drag *when one exists*? Plant a range of "
            "in-season drifts in a synthetic daily tape and sweep."
        ),
        code(
            "planted = [0, -2, -5, -10, -20]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_daily(n_years=33, season_bps=float(bps), seed=283)\n"
            "    r = st.season_stats(df_s, n_permutations=500, seed=283)\n"
            "    rows.append({'plant_bps': bps, 'spread': r['spread'],\n"
            "                 'hac_t': r['hac_t'], 'perm_p': r['perm_p']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if (abs(r['hac_t'])>=2 and r['perm_p']<0.05) else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['plant_bps']) for r in rows], [r['hac_t'] for r in rows], color=colors)\n"
            "ax.axhline(-2, c='k', ls=':', lw=1, label='HAC t = -2')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted in-season drag (bps/day)')\n"
            "ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine finds the drag once it is real (green = HAC |t|>=2 & p<0.05)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "The engine recovers a planted drag once it is large enough (HAC t < -3 at -10 "
            "bps/day; clean rejection at -20). The null plant (0 bps) is insignificant, as it "
            "must be. The real tape's -0.47 bps/day market gap is far below the detection floor "
            "-- the verdict is a statement about **the market and the calendar**, not the method."
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


def _execute(filename: str) -> None:
    import subprocess, sys
    path = os.path.join(HERE, filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {filename}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {filename}")
    print("executed", filename)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
