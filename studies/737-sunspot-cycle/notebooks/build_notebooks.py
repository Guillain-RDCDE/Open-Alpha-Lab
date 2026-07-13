"""Generate the two narrative notebooks for Study 737 (Sunspot-Cycle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^GSPC
price-only close under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^GSPC price-only
# monthly close 1927-12-31 -> 2026-06-30; solar cycles 16-25 hardcoded from SILSO/NOAA).
R = dict(
    asof="2026-06-30", cyc_lo=16, cyc_hi=25, n_cycles=10,
    daily_rows=24740, monthly_rows=1183, span_lo="1927-12-31", span_hi="2026-06-30",
    n_max=10, n_min=9,
    # PRIMARY — forward-12-month price-only return after turning points
    max_mean=5.63, max_t=0.736, min_mean=18.91, min_t=3.327,
    diff_mean=-13.28, welch_t=-1.394, uncond_fwd=8.17,
    min_pl_mean=8.15, min_pl_sd=6.67, min_pl_p=0.054, max_pl_p=0.674,
    min_hit=7, min_hit_n=9, min_wilson=(45.3, 93.7),
    max_hit=6, max_hit_n=10, max_wilson=(31.3, 83.2),
    # forward returns per event, for the tour (max then min)
    max_fwd={"1928-04": 31.3, "1937-04": -40.5, "1947-05": 15.5, "1958-03": 31.7,
             "1968-11": -13.4, "1979-12": 25.8, "1989-11": -6.9, "2001-11": -17.8,
             "2014-04": 10.7, "2024-10": 19.9},
    min_fwd={"1933-09": -6.6, "1944-02": 21.0, "1954-04": 34.3, "1964-10": 8.9,
             "1976-03": -4.2, "1986-09": 39.1, "1996-08": 38.0, "2008-12": 23.5,
             "2019-12": 16.3},
    # REGIME split — high vs low activity months
    hi_mean_mo=0.441, lo_mean_mo=1.002, regime_spread_annbps=-673.2,
    regime_p=0.179, regime_ci_mo=(-1.378, 0.271), n_hi=387, n_lo=387,
    regime_pl_mean=-1.9, regime_pl_sd=379.3, regime_pl_p=0.095,
    # PHASE regression — the 11-year sinusoid
    t_cos=1.041, t_sin=1.830, r2=0.00360, amp_annbps=554.3,
    # TIMER — cost -> (timer_cagr%, excess%/yr, t_diff)
    timer={0: (3.10, -3.13, -2.654), 5: (3.09, -3.14, -2.660), 10: (3.07, -3.15, -2.666)},
    bh_cagr=6.23, timer_expo=38.2, timer_switches=20, timer_sharpe=0.366, bh_sharpe=0.420,
    # SYNTHETIC control
    syn_null_p=0.471, syn_null_fire=1, syn_null_r2=0.00175,
    syn_planted_spread=-3970.4, syn_planted_r2=0.114, syn_planted_tcos=14.47,
    fp_daily="74635b6681a5", fp_monthly="a26d426a3204",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Jevons cycle%3F: Busted](https://img.shields.io/badge/Jevons_cycle%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"
GOLD = "#e8a33d"

from sunspot_cycle import data, strategy as st

CYCLES = data.solar_cycles()
TPS = data.turning_points()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    CLOSE = data.load_real()
    M = data.monthly_close(CLOSE)
    RET = st.monthly_returns(M)
    AR = st.abnormal_returns(RET)
    PROX = data.sunspot_proxy(M.index)
else:
    CLOSE = M = RET = AR = PROX = None
print("real cache present:", HAVE_REAL, "| solar cycles:", len(CYCLES),
      "| turning points:", len(TPS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does an 11-year clock on the Sun move the stock market? ☀️📈\n"
            "### The \"sunspot cycle → stock returns\" curio — a Victorian idea that "
            "still floats around, tested on a century of the S&P\n\n"
            + BADGES +
            "In the 1870s the economist **William Stanley Jevons** noticed that the Sun's "
            "roughly **11-year sunspot cycle** seemed to line up with waves of boom and "
            "bust — his theory ran *sunspots → weather → harvests → trade cycles → "
            "markets*. The harvest chain has long since been abandoned, but the headline "
            "keeps being reincarnated: *the market runs on an 11-year solar clock.* When "
            "the Sun is active, the story goes, stocks do well; when it's quiet, they "
            "don't.\n\n"
            "It's the perfect curio to put on the desk's bench, because we have an "
            "unusually long tape (the S&P back to **1927**, almost **ten** full solar "
            "cycles) and the cycle calendar is *public, free, and known centuries in "
            "advance*. If it worked even a little, it would be the easiest market-timing "
            "signal ever printed. Let's see.\n\n"
            "> 📓 **Plain-language layer.** Want the regression, the block-bootstrap and "
            "the placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the 11-year solar cycle explain stock returns? | **No.** Fit a clean "
            f"11-year wave to monthly returns and it explains **{R['r2']*100:.2f}%** of "
            "the variation — essentially nothing, and not statistically significant. |\n"
            "| Do stocks do better when the Sun is active? | **No — if anything the "
            f"reverse.** High-activity months average **{R['hi_mean_mo']:.2f}%** vs "
            f"**{R['lo_mean_mo']:.2f}%** for quiet months, and even that gap is "
            "statistically indistinguishable from a coincidence. |\n"
            "| Is there *any* solar date that looks special? | **One, and it's a trap.** "
            f"The year after a solar **minimum** the S&P rose **+{R['min_mean']:.0f}%** on "
            f"average (*t* = {R['min_t']:.1f}) — but that's the *opposite* of the claim, "
            "it's just \"the market drifts up,\" and against a random calendar it barely "
            f"registers (p = {R['min_pl_p']:.2f}). |\n"
            "| Could you trade a solar clock? | **No.** A \"long in the active half\" "
            f"timer earns **{R['timer'][5][0]:.1f}%/yr** vs **{R['bh_cagr']:.1f}%/yr** for "
            "just buying and holding — it loses, significantly, before costs even bite. |\n\n"
            "> A 150-year-old curio, given a century of data and every benefit of the "
            "doubt — and the 11-year clock simply isn't in the tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Sun runs on an ~11-year cycle of sunspots, and so, roughly, does the "
            "economy. Jevons showed it lines up with commercial crises; active-Sun years "
            "are boom years and quiet-Sun years are lean ones. The cycle is astronomical "
            "— utterly exogenous to markets — so it's a rare *clean* predictor: the "
            "calendar is set by the Sun, not by anything traders do.\"*\n\n"
            "This is genuine intellectual history, not a strawman: Jevons (1878) really "
            "did present a sunspot theory of the trade cycle to the British Association, "
            "and \"solar cycle investing\" newsletters have recycled it ever since. The "
            "steelman is exactly the thing that makes it testable — the cycle is "
            "**exogenous and known in advance**, so there's no reverse-causation or "
            "data-mining-the-dates escape hatch."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a free, exogenous, 11-year astronomical calendar could time the equity "
            "market, it would be extraordinary in two directions at once. Practically, "
            "you'd have a market-timing signal printed *decades* ahead with zero data "
            "cost. Scientifically, it would say something is transmitting solar activity "
            "into human financial behaviour on an 11-year beat — the market as a very "
            "expensive sunspot detector.\n\n"
            "So we ask it three ways: does a fitted 11-year wave explain returns, do "
            "active-Sun months beat quiet-Sun months, and could a solar-clock timer beat "
            "just holding the index?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The clock.** Solar cycles **{R['cyc_lo']}–{R['cyc_hi']}** "
            f"({R['n_cycles']} of them) — each cycle's minimum and maximum date, keyed "
            "from the official SILSO / NOAA sunspot record. From those we reconstruct "
            "*where in the 11-year cycle* each month sits (a labelled proxy, not the raw "
            "sunspot file).\n"
            f"- **The tape.** The S&P 500 **price index** back to {R['span_lo'][:4]} — "
            f"almost a century, {R['n_cycles']} solar cycles. (Price-only, no dividends — "
            "labelled as such; an 11-year wiggle can't hide in dividends anyway.)\n"
            "- **The three tests.** (1) fit a clean 11-year wave to monthly returns and "
            "see how much it explains; (2) compare active-Sun months to quiet-Sun months; "
            "(3) look at the year *after* each solar peak and each solar trough.\n"
            "- **The luck check.** Slide the whole solar calendar to a random start "
            "thousands of times — how often does a *random* 11-year clock look as good?\n"
            "- **The trade check.** Hold the S&P through the active half of each cycle, "
            "sit in cash through the quiet half, pay costs, compare to buy-and-hold."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the picture that launched a thousand newsletters.** Put the S&P "
            "(log scale, so a century fits) next to the sunspot cycle and eyeball it — do "
            "they march together?"
        ),
        code(
            "fig, ax1 = plt.subplots(figsize=(10.2, 4.8))\n"
            "if HAVE_REAL:\n"
            "    ax1.semilogy(M.index, M.values, color='k', lw=1.1, label='S&P 500 (price, log)')\n"
            "    ax2 = ax1.twinx()\n"
            "    ax2.fill_between(PROX.index, PROX['ssn_proxy'].values, color=GOLD, alpha=.35,\n"
            "                     label='sunspot proxy')\n"
            "    ax2.set_ylabel('sunspot number (SILSO proxy)', color=GOLD)\n"
            "    ax2.set_ylim(0, 340)\n"
            "    for _, r in CYCLES.iterrows():\n"
            "        ax1.axvline(r['max_date'], color=GOLD, ls='--', lw=.8, alpha=.7)\n"
            "else:\n"
            "    ax1.text(.5, .5, 'cache miss', ha='center')\n"
            "ax1.set_ylabel('S&P 500 price index (log)')\n"
            "ax1.set_title('A century of the S&P vs the 11-year sunspot cycle — spot the clock?')\n"
            "ax1.legend(loc='upper left')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('dashed gold lines = solar maxima; the S&P climbs through active AND quiet suns alike')"
        ),
        md(
            "The market goes up through active Suns and quiet Suns alike; the crashes "
            "(1929, 1974, 2000, 2008, 2020) land at all phases of the cycle. Eyeballing "
            "won't settle it though — let's put numbers on the two halves of the cycle.\n\n"
            "**Active-Sun months vs quiet-Sun months.** Split every month by how busy the "
            "Sun was and compare average returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.regime_split(RET, PROX)\n"
            "    hi_b, lo_b = rs['hi_mean'] * 100, rs['lo_mean'] * 100\n"
            "else:\n"
            "    hi_b, lo_b = R['hi_mean_mo'], R['lo_mean_mo']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.4))\n"
            "ax.bar(['active Sun\\n(high sunspots)', 'quiet Sun\\n(low sunspots)'],\n"
            "       [hi_b, lo_b], color=[GOLD, GREY], width=.55)\n"
            "for i, v in enumerate([hi_b, lo_b]):\n"
            "    ax.annotate(f'{v:.2f}%/mo', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average monthly S&P return')\n"
            "ax.set_title('If anything, the market does slightly WORSE when the Sun is active')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'active-Sun {hi_b:.3f}%/mo vs quiet-Sun {lo_b:.3f}%/mo -> the sign is backwards for Jevons')"
        ),
        md(
            "The bars lean the *wrong way* for the claim — quiet-Sun months edge out "
            "active-Sun months — and (the quants notebook shows) the gap is well inside "
            "what a random 11-year clock throws up. So much for \"active Sun, good "
            "market.\"\n\n"
            "**The one solar date that looks special — and why it's a trap.** Look at the "
            "12 months *after* each solar peak and each solar trough:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp = st.turning_point_stats(M, TPS, horizon=12)\n"
            "    mx_b, mn_b = tp['max_mean'] * 100, tp['min_mean'] * 100\n"
            "else:\n"
            "    mx_b, mn_b = R['max_mean'], R['min_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['after a solar MAX\\n(claim: boom)', 'after a solar MIN\\n(claim: bust)',\n"
            "        'any random year'], [mx_b, mn_b, R['uncond_fwd']],\n"
            "       color=[GOLD, GREY, '#bbbbbb'], width=.6)\n"
            "for i, v in enumerate([mx_b, mn_b, R['uncond_fwd']]):\n"
            "    ax.annotate(f'+{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average next-12-month S&P return')\n"
            "ax.set_title('The best year follows solar MINIMA — the exact opposite of the claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'after MAX +{mx_b:.1f}%  |  after MIN +{mn_b:.1f}%  |  random year +{R[\"uncond_fwd\"]:.1f}%')"
        ),
        md(
            f"Here's the trap. The year after a solar **minimum** the S&P rose "
            f"**+{R['min_mean']:.0f}%** on average — a big-looking number. But it points "
            "the **wrong way** (the claim says peaks are the good times), it rests on just "
            f"**{R['min_hit_n']} events**, and — crucially — *any* random year already "
            f"returns **+{R['uncond_fwd']:.1f}%** because the market drifts up. Measured "
            "against a random calendar of the same size, that shiny minimum number is only "
            f"a **p = {R['min_pl_p']:.2f}** curiosity (the quants notebook nails this "
            "down). A few solar minima happened to sit near market bottoms (1954, 1986, "
            "2008); with nine of them, that's coincidence, not a solar law.\n\n"
            "**Finally, the trade.** Could a solar-clock timer beat just holding?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm = st.solar_timer(M, PROX, smooth_lag=6, cost_bps=5.0)\n"
            "    t_cagr, bh_cagr = tm['timer_cagr'] * 100, tm['bh_cagr'] * 100\n"
            "else:\n"
            "    t_cagr, bh_cagr = R['timer'][5][0], R['bh_cagr']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.4))\n"
            "ax.bar(['solar-clock timer\\n(long the active half)', 'just buy & hold'],\n"
            "       [t_cagr, bh_cagr], color=[GOLD, GREEN], width=.55)\n"
            "for i, v in enumerate([t_cagr, bh_cagr]):\n"
            "    ax.annotate(f'{v:.2f}%/yr', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualised return')\n"
            "ax.set_title('Timing the market on the Sun just keeps you out of a rising market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'solar timer {t_cagr:.2f}%/yr vs buy-and-hold {bh_cagr:.2f}%/yr')"
        ),
        md(
            f"The timer earns **{R['timer'][5][0]:.1f}%/yr** against buy-and-hold's "
            f"**{R['bh_cagr']:.1f}%/yr** — it *halves* your return, because \"sit in cash "
            "for the quiet half of the cycle\" mostly means sitting out of a market that "
            "drifts up regardless of the Sun. There's no edge to charge costs against; "
            "the costs are almost a rounding error next to the damage done by being out "
            "of the market."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** A fitted 11-year wave explains **{R['r2']*100:.2f}%** "
            "of monthly returns; active-Sun months don't beat quiet-Sun months (the sign "
            "is even backwards); the one *t* > 2 in the whole study points the wrong way, "
            "rests on 9 events, and evaporates against a random calendar.\n"
            "- **Tradability — Mirage.** A solar-clock timer loses to buy-and-hold by "
            f"~**{-R['timer'][5][1]:.1f} percentage points a year** — before costs.\n"
            "- **\"Jevons's sunspot cycle in the market?\" — Busted.** Given a century of "
            "data and every benefit of the doubt, the 11-year clock isn't there."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This isn't a knock on solar physics** — the sunspot cycle is beautifully "
            "real. It's a knock on the *market* half of the claim: there's no 11-year beat "
            "in equity returns for the solar cycle to sync with.\n"
            "- **The minima curiosity is the fun thread to pull.** It's the wrong sign for "
            "Jevons and dies against a random calendar — but if you *wanted* to steelman "
            "\"buy the quiet Sun,\" the honest next step is a pre-registered out-of-sample "
            "test on other countries' long indices, not more slicing of this one.\n"
            "- **Sibling curios on this desk:** "
            "[279-geomagnetic-storms](../../279-geomagnetic-storms/) (a *physiological* "
            "solar-activity → mood channel — a different mechanism entirely), "
            "[280-solar-eclipse](../../280-solar-eclipse/), "
            "[278-sunshine-effect](../../278-sunshine-effect/) and "
            "[150-sad-effect](../../150-sad-effect/) (daylight/mood) — the whole "
            "\"sky moves markets\" family, each tested with the same rails.\n\n"
            "*Think a longer or non-US tape hides the 11-year clock? Fork it, run the same "
            "three tests out-of-sample, and show the wave — then we'll talk.*"
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
            "# The Sunspot-Cycle — a quantitative teardown 🔬\n"
            "### An 11-year phase regression (HAC) · a high/low activity regime split with "
            "a circular block bootstrap · an independent-turning-point forward-return event "
            "study and its random-calendar placebo · a lag-honest solar-clock timer · a "
            "20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — Jevons's (1878) sunspot theory of the "
            "trade cycle, in its surviving \"11-year solar clock on the market\" form — is "
            "a genuine piece of intellectual history and, better still, an *exogenous, "
            "known-in-advance* calendar with no reverse-causation escape hatch. The job "
            "here is to measure it honestly on the longest equity tape available, then ask "
            "the only question that pays: *is any of it real, and if so, tradable?*\n\n"
            "> ⚠️ **Data note.** ^GSPC **price-only** monthly close (1927→2026), yfinance, "
            "cached; **solar cycles 16–25 hardcoded** from the SILSO/NOAA record, with a "
            "**labelled cosine proxy** of the smoothed monthly sunspot number (not the raw "
            "SILSO file). Price-only is labelled everywhere — there is no total-return "
            "equity series of this length, and an 11-year cyclicality cannot live in "
            "dividends. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_daily"] +
            "` daily / `" + R["fp_monthly"] + "` monthly).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 11-year phase regression R² = **{R['r2']*100:.2f}%**, "
            f"HAC *t* = **{R['t_cos']:.2f}** (cos) / **{R['t_sin']:.2f}** (sin); regime "
            f"spread **{R['regime_spread_annbps']:.0f} bps/yr** (wrong sign), placebo "
            f"**p = {R['regime_pl_p']:.3f}** |\n"
            f"| **Tradability** | `MIRAGE` | solar-clock timer "
            f"**{R['timer'][5][0]:.1f}%/yr** vs buy-and-hold **{R['bh_cagr']:.1f}%/yr**, "
            f"excess *t* = **{R['timer'][5][2]:.2f}** — significantly worse, pre-cost |\n"
            f"| **Jevons's cycle in the tape?** | `BUSTED` | the one \\|*t*\\| > 2 (forward "
            f"return after solar minima, *t* = **{R['min_t']:.2f}**) is wrong-signed, n = "
            f"{R['min_hit_n']}, and only **p = {R['min_pl_p']:.2f}** vs a random calendar |\n\n"
            "> 💡 In plain words: the exogenous, benefit-of-the-doubt version of the test "
            "comes back empty on every axis, and the single shiny number is a look-elsewhere "
            "artifact pointing the wrong way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the S&P's monthly price-only return, $a_t = r_t - \\bar r$ its "
            "abnormal return (constant-mean model, Brown & Warner 1985), and $\\varphi_t "
            "\\in [0, 2\\pi)$ the solar phase (0 at a minimum, $\\pi$ at the following "
            "maximum), reconstructed from the SILSO turning points. The claim's testable "
            "forms:\n\n"
            "- **H₁ (11-year wave).** $a_t = \\beta_c \\cos\\varphi_t + \\beta_s "
            "\\sin\\varphi_t + \\varepsilon_t$ has a jointly significant $(\\beta_c, "
            "\\beta_s)$ and a non-trivial $R^2$.\n"
            "- **H₂ (activity regime).** Mean return in high-activity months exceeds "
            "low-activity months: $E[r \\mid \\text{active}] > E[r \\mid \\text{quiet}]$.\n"
            "- **H₃ (turning points).** The 12-month forward return is higher after solar "
            "**maxima** than after **minima** (boom follows the active peak).\n"
            "- **H₄ (capture).** A long-in-the-active-half timer beats buy-and-hold net of "
            "costs.\n\n"
            f"We find **H₁ not supported** (R² = {R['r2']*100:.2f}%, both HAC *t* < 2), "
            f"**H₂ rejected on sign** (spread {R['regime_spread_annbps']:.0f} bps/yr, "
            f"active < quiet), **H₃ rejected on sign** (Welch *t* = {R['welch_t']:.2f}, "
            "min > max), **H₄ not supported** (timer loses, excess *t* = "
            f"{R['timer'][5][2]:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two units of analysis, matched to two kinds of test, on purpose:\n\n"
            "- **Turning points are independent events.** Solar maxima and minima are "
            "≈ 5–6 years apart, so a 12-month forward-return window around one never "
            "overlaps the next — the right test is a **one-sample t across the "
            f"{R['n_max']} + {R['n_min']} events**, not a HAC regression on an "
            "autocorrelated panel. Each turning-point group's forward return is also "
            "placed against a **random-calendar placebo** of the same size, because the "
            "market's up-drift makes *every* set of dates look positive.\n"
            "- **The monthly panel is autocorrelated.** For the phase regression and the "
            "regime split the unit is the month, and consecutive months (and the highly "
            "persistent regime label) are correlated — so the phase regression uses "
            "**Newey-West (HAC, 12 lags)** *t*'s and the regime spread a **circular block "
            "bootstrap** (12-month blocks), never an i.i.d. monthly *t*. The falsification "
            "is a **phase-shift placebo**: roll the entire solar calendar to a random "
            "start, preserving its exact 11-year shape but breaking its real alignment."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Clock.** Solar cycles {R['cyc_lo']}–{R['cyc_hi']} ({R['n_cycles']}), "
            "SILSO/NOAA turning points hardcoded; a labelled cosine proxy of the smoothed "
            "monthly sunspot number (phase, activity, rising-flag).\n"
            f"- **Tape.** ^GSPC price-only monthly close, {R['span_lo']} → {R['asof']} "
            f"(as-of, last complete month), {R['monthly_rows']} months.\n"
            "- **H₁.** OLS $a_t$ on $(1, \\cos\\varphi_t, \\sin\\varphi_t)$; HAC *t*'s, R², "
            "sinusoid amplitude in annualised bps.\n"
            "- **H₂.** High vs low activity tercile; block-bootstrap 95% CI + two-sided p "
            "of the spread; phase-shift placebo.\n"
            f"- **H₃.** 12-month forward return after each of {R['n_max']} maxima and "
            f"{R['n_min']} minima; one-sample & Welch *t*; random-calendar placebo per "
            "group; Wilson hit rates.\n"
            "- **H₄ (timer).** Long the index when the *lagged* phase is rising "
            "(min→max), else cash; **one documented lag** — the SILSO smoothing lag "
            "(6 months: you only *know* the phase once the smoothed number is published); "
            "one-way cost × NAV per switch; long-or-flat; vs buy-and-hold.\n"
            "- **Control.** Synthetic monthly tape with a TUNABLE planted solar-cycle "
            "return term; the null (amp=0) must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · H₁ — the 11-year wave (phase regression, HAC)\n\n"
            "Regress monthly abnormal returns on $\\cos\\varphi$ and $\\sin\\varphi$. A "
            "real 11-year cycle is a jointly significant $(\\cos,\\sin)$ pair and a "
            "non-trivial R²."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pr = st.phase_regression(AR, PROX)\n"
            "    r2, tc, ts_, amp = pr['r2'], pr['t_cos'], pr['t_sin'], pr['amp_ann_bps']\n"
            "    df = pd.DataFrame({'ar': AR}).join(PROX[['phase']]).dropna()\n"
            "    ph = df['phase'].to_numpy(); y = df['ar'].to_numpy()\n"
            "    order = np.argsort(ph)\n"
            "    fitted = pr['beta_cos'] * np.cos(ph) + pr['beta_sin'] * np.sin(ph)\n"
            "else:\n"
            "    r2, tc, ts_, amp = R['r2'], R['t_cos'], R['t_sin'], R['amp_annbps']\n"
            "    ph = np.linspace(0, 2*np.pi, 400); order = np.argsort(ph)\n"
            "    y = None; fitted = 0.006*np.cos(ph)\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "if y is not None:\n"
            "    ax.scatter(ph, y*100, s=6, color=GREY, alpha=.35, label='monthly abnormal return')\n"
            "ax.plot(ph[order], fitted[order]*100, color=RED, lw=2.5,\n"
            "        label=f'fitted 11-yr wave (R2={r2*100:.2f}%)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])\n"
            "ax.set_xticklabels(['min', 'rising', 'MAX', 'falling', 'min'])\n"
            "ax.set_xlabel('solar phase'); ax.set_ylabel('monthly abnormal return (%)')\n"
            "ax.set_title(f'A flat line with a whisker of a wave on it: R2 = {r2*100:.2f}%')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'cos t = {tc:+.2f}, sin t = {ts_:+.2f}, R2 = {r2*100:.3f}%, amplitude = {amp:.0f} bps/yr')"
        ),
        md(
            f"> 💡 In plain words: the best-fit 11-year wave explains "
            f"**{R['r2']*100:.2f}%** of monthly returns — for scale, that's about "
            "**1 part in 280**. Neither harmonic clears the desk bar (cos *t* = "
            f"{R['t_cos']:.2f}, sin *t* = {R['t_sin']:.2f}), and even taking the "
            f"point estimate at face value the whole swing top-to-bottom is "
            f"~{R['amp_annbps']:.0f} bps/yr of *predicted* return — swamped by the "
            "market's own monthly noise. H₁ is not supported."
        ),
        md(
            "### 4b · H₂ — active vs quiet Sun (regime split + block bootstrap + placebo)\n\n"
            "High-activity (top tercile) vs low-activity (bottom tercile) months; the "
            "spread gets a 12-month circular-block-bootstrap CI, and a phase-shift placebo "
            "(roll the whole solar calendar to a random start)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.regime_split(RET, PROX)\n"
            "    pl = st.placebo_regime_spread(RET, PROX, n_draws=3000, seed=737)\n"
            "    spread_ann = rs['spread_ann_bps']; p_boot = rs['p_boot']\n"
            "    pl_ann = pl * 12 * 1e4\n"
            "    obs = rs['spread']\n"
            "    p_pl = st.placebo_pvalue(obs, pl, tail='two')\n"
            "else:\n"
            "    spread_ann, p_boot, p_pl = R['regime_spread_annbps'], R['regime_p'], R['regime_pl_p']\n"
            "    rng = np.random.default_rng(737)\n"
            "    pl_ann = rng.normal(R['regime_pl_mean'], R['regime_pl_sd'], 3000)\n"
            "    obs = spread_ann / (12*1e4)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(pl_ann, bins=45, color=GREY, alpha=.85,\n"
            "        label='null: random 11-year clocks (phase-shift placebo)')\n"
            "ax.axvline(spread_ann, c=RED, lw=2.5,\n"
            "           label=f'observed high-low spread {spread_ann:+.0f} bps/yr')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('high-activity minus low-activity spread (annualised bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud, and on the wrong side of zero (placebo p = {p_pl:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed spread {spread_ann:+.0f} bps/yr | block-boot p = {p_boot:.3f} | placebo p = {p_pl:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the active−quiet spread is "
            f"**{R['regime_spread_annbps']:.0f} bps/yr** — *negative*, i.e. the market "
            "does slightly worse when the Sun is busy, the opposite of H₂. And it's not "
            f"even a real negative: the block-bootstrap two-sided p is **{R['regime_p']:.3f}** "
            f"and the phase-shift placebo p is **{R['regime_pl_p']:.3f}** — a random "
            "11-year clock produces a spread this big about one time in ten. H₂ is "
            "rejected on sign and unsupported on magnitude."
        ),
        md(
            "### 4c · H₃ — turning points, and the one shiny number in the study\n\n"
            "12-month forward return after each solar maximum and each solar minimum "
            "(independent events). Each group's mean is placed against a random-calendar "
            "placebo of the same size — because the market's up-drift makes *any* set of "
            "dates look positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp = st.turning_point_stats(M, TPS, horizon=12)\n"
            "    plmin = st.forward_placebo(M, tp['n_min'], 12, n_draws=20000, seed=737)\n"
            "    plmax = st.forward_placebo(M, tp['n_max'], 12, n_draws=20000, seed=737)\n"
            "    mx_m, mn_m = tp['max_mean'], tp['min_mean']\n"
            "    mx_t, mn_t, wt = tp['max_t'], tp['min_t'], tp['welch_t']\n"
            "    p_min = float((plmin >= mn_m).mean()); p_max = float((plmax >= mx_m).mean())\n"
            "else:\n"
            "    mx_m, mn_m = R['max_mean']/100, R['min_mean']/100\n"
            "    mx_t, mn_t, wt = R['max_t'], R['min_t'], R['welch_t']\n"
            "    p_min, p_max = R['min_pl_p'], R['max_pl_p']\n"
            "    rng = np.random.default_rng(737)\n"
            "    plmin = rng.normal(R['min_pl_mean']/100, R['min_pl_sd']/100, 20000)\n"
            "    plmax = rng.normal(R['uncond_fwd']/100, R['min_pl_sd']/100, 20000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.hist(plmin*100, bins=50, color=GREY, alpha=.85,\n"
            "        label='random 9-date calendars')\n"
            "a1.axvline(mn_m*100, c=RED, lw=2.5, label=f'after solar minima {mn_m*100:+.1f}%')\n"
            "a1.set_title(f'Solar minima: shiny t={mn_t:.1f}, but placebo p={p_min:.3f}')\n"
            "a1.set_xlabel('mean next-12m return (%)'); a1.legend(fontsize=8)\n"
            "a2.hist(plmax*100, bins=50, color=GREY, alpha=.85, label='random 10-date calendars')\n"
            "a2.axvline(mx_m*100, c=GOLD, lw=2.5, label=f'after solar maxima {mx_m*100:+.1f}%')\n"
            "a2.set_title(f'Solar maxima (the claim): t={mx_t:.1f}, placebo p={p_max:.2f}')\n"
            "a2.set_xlabel('mean next-12m return (%)'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'MAX +{mx_m*100:.1f}% (t={mx_t:.2f}, placebo p={p_max:.3f}) | '\n"
            "      f'MIN +{mn_m*100:.1f}% (t={mn_t:.2f}, placebo p={p_min:.3f}) | Welch(max-min) t={wt:.2f}')"
        ),
        md(
            f"> 💡 In plain words: after solar **maxima** — the *predicted* boom — the S&P "
            f"returns **+{R['max_mean']:.1f}%** over the next year (*t* = {R['max_t']:.2f}, "
            f"placebo p = {R['max_pl_p']:.2f}): nothing, and if anything below the "
            f"**+{R['uncond_fwd']:.1f}%** you'd get on a random year. The **minima** number "
            f"is the study's one *t* > 2 (**+{R['min_mean']:.1f}%**, *t* = {R['min_t']:.2f}) "
            "— but it is (i) the **wrong direction** for the claim, (ii) tested against a "
            f"random calendar only **p = {R['min_pl_p']:.2f}** (that fat one-sample *t* is "
            "just re-discovering that markets drift up over any 9 dates), and (iii) built "
            f"on {R['min_hit_n']} events, a few of which (1954, 1986, 2008) happen to sit "
            f"near market bottoms. The Welch max−min contrast is *t* = {R['welch_t']:.2f}, "
            "wrong-signed. H₃ is rejected."
        ),
        md(
            "### 4d · H₄ — the solar-clock timer, lag-honest and costed\n\n"
            "Long the index when the *lagged* phase is rising (min→max), else cash. The "
            "phase is only known once the smoothed sunspot number is published, so the "
            "signal is lagged **6 months** (one documented lag, applied once). Costs "
            "one-way × NAV per switch; long-or-flat; vs buy-and-hold on the identical "
            "window."
        ),
        code(
            "rows = []\n"
            "for c in (0.0, 5.0, 10.0):\n"
            "    if HAVE_REAL:\n"
            "        tm = st.solar_timer(M, PROX, smooth_lag=6, cost_bps=c)\n"
            "        rows.append((c, tm['timer_cagr']*100, tm['bh_cagr']*100,\n"
            "                     tm['excess_cagr']*100, tm['t_diff'], tm['timer_sharpe'],\n"
            "                     tm['bh_sharpe'], tm['exposure']))\n"
            "    else:\n"
            "        tc_, ex_, td_ = R['timer'][int(c)]\n"
            "        rows.append((c, tc_, R['bh_cagr'], ex_, td_, R['timer_sharpe'],\n"
            "                     R['bh_sharpe'], R['timer_expo']/100))\n"
            "labels = [f'{int(c)} bps' for c, *_ in rows]\n"
            "t_cagrs = [r[1] for r in rows]; bh = rows[0][2]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.5))\n"
            "x = np.arange(len(rows))\n"
            "ax.bar(x, t_cagrs, width=.5, color=GOLD, label='solar-clock timer')\n"
            "ax.axhline(bh, c=GREEN, lw=2, ls='--', label=f'buy & hold {bh:.2f}%/yr')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_xlabel('one-way cost per switch'); ax.set_ylabel('annualised return')\n"
            "ax.set_title('The timer loses to buy-and-hold at every cost — the costs barely matter')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for c, tcg, bhc, ex, td, sht, shb, expo in rows:\n"
            "    print(f'{int(c):>2d} bps: timer {tcg:.2f}%/yr  excess {ex:+.2f}%/yr  '\n"
            "          f'excess t={td:+.2f}  Sharpe {sht:.2f} vs {shb:.2f}  exposure {expo*100:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: the timer earns ~**{R['timer'][5][0]:.1f}%/yr** against "
            f"buy-and-hold's **{R['bh_cagr']:.1f}%/yr** — an excess of "
            f"**{R['timer'][5][1]:.1f} pp/yr** at *t* = **{R['timer'][5][2]:.2f}** "
            "(significantly *worse*), and its Sharpe is lower too "
            f"({R['timer_sharpe']:.2f} vs {R['bh_sharpe']:.2f}). It's only invested "
            f"~{R['timer_expo']:.0f}% of the time, so it mostly loses by sitting out of a "
            "rising market — the costs (a handful of switches over 98 years) are almost "
            "irrelevant. H₄ is not supported; there is no edge to charge costs against."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic monthly tape phased to the *real* solar calendar, with a TUNABLE "
            "planted solar-cycle return term. The null (amp=0) is checked over **20 "
            "seeds** — never a single stream — and a planted cycle must light up."
        ),
        code(
            "null_p = []\n"
            "for s_ in range(20):\n"
            "    c, px = data.synthetic_world(amp=0.0, seed=737 + s_)\n"
            "    null_p.append(st.synthetic_detect(c, px)['regime_p'])\n"
            "null_p = np.asarray(null_p)\n"
            "c, px = data.synthetic_world(amp=0.02, seed=737)\n"
            "planted = st.synthetic_detect(c, px)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_p, color=GREY, s=40,\n"
            "           label='null worlds (amp=0), 20 seeds')\n"
            "ax.scatter([1], [planted['regime_p']], color=RED, s=90, zorder=5,\n"
            "           label='planted solar cycle (amp=2%)')\n"
            "ax.axhline(0.05, ls='--', c=RED, lw=1, label='p = 0.05')\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('regime-split p-value'); ax.set_ylim(-0.03, 1.03)\n"
            "ax.set_title('Control: nulls scatter across p; a planted cycle is detected at p<0.001')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean p = {null_p.mean():.2f}, p<0.05 in {(null_p<0.05).sum()}/20 seeds  |  '\n"
            "      f'planted: regime p = {planted[\"regime_p\"]:.3f}, phase R2 = {planted[\"phase_r2\"]*100:.1f}%, '\n"
            "      f'cos t = {planted[\"t_cos\"]:.1f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the regime test averages "
            f"p = {R['syn_null_p']:.2f} and falsely fires in only "
            f"{R['syn_null_fire']}/20 seeds (≈ the 5% you'd expect), while a planted 2% "
            f"solar cycle is caught at p < 0.001 (phase R² = {R['syn_planted_r2']*100:.0f}%, "
            f"cos *t* = {R['syn_planted_tcos']:.1f}). The machinery detects a solar cycle "
            "when there *is* one — so the empty real-tape result is a genuine reading, not "
            "a sleepy detector. *(A faithful-engine / power check only — never cited in "
            "support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the 11-year phase regression explains R² = "
            f"**{R['r2']*100:.2f}%** of monthly returns with both HAC *t*'s below 2 "
            f"(cos {R['t_cos']:.2f}, sin {R['t_sin']:.2f}); the active−quiet regime spread "
            f"is **{R['regime_spread_annbps']:.0f} bps/yr**, wrong-signed, block-boot "
            f"p = {R['regime_p']:.3f}, placebo p = {R['regime_pl_p']:.3f}; the max−min "
            f"turning-point contrast is Welch *t* = {R['welch_t']:.2f}, also wrong-signed. "
            f"The lone \\|*t*\\| > 2 (forward return after minima, *t* = {R['min_t']:.2f}) is "
            f"a wrong-signed, n={R['min_hit_n']} look-elsewhere artifact that is only "
            f"p = {R['min_pl_p']:.2f} against a random calendar.\n"
            f"- **Tradability `MIRAGE`** — a lag-honest solar-clock timer earns "
            f"**{R['timer'][5][0]:.1f}%/yr** vs buy-and-hold's **{R['bh_cagr']:.1f}%/yr** "
            f"(excess *t* = {R['timer'][5][2]:.2f}), lower Sharpe too, before costs bite. "
            "No edge exists to charge costs against.\n"
            "- **\"Jevons's sunspot cycle in the market?\" `BUSTED`** — on ten solar "
            "cycles of the longest S&P tape available, with an exogenous known-in-advance "
            "calendar and every benefit of the doubt, there is no 11-year clock in equity "
            "returns. (This says nothing about the sunspot cycle itself, which is real; "
            "only that the *market* does not run on it.)"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The exogenous calendar is the point.** Unlike most anomalies, there is no "
            "reverse-causation or date-mining escape hatch here — the Sun's schedule is "
            "fixed and public. That makes the null unusually clean: it isn't \"we couldn't "
            "find the trade,\" it's \"the 11-year period isn't in the returns.\"\n"
            "- **The minima curiosity** is the honest loose end: wrong-signed for Jevons, "
            "p = 0.05 against a random calendar, and economically a post-crisis-recovery "
            "coincidence in 9 events. A pre-registered out-of-sample test on other long "
            "national indices (UK, France) is the disciplined way to kill or confirm it — "
            "not more cuts of this tape.\n"
            "- **Dedup map:** [279-geomagnetic-storms](../../279-geomagnetic-storms/) "
            "tests a *different* solar-activity channel (geomagnetic disturbance → human "
            "mood → returns, a physiological hypothesis), "
            "[280-solar-eclipse](../../280-solar-eclipse/) an event-day superstition, and "
            "[278-sunshine-effect](../../278-sunshine-effect/) / "
            "[150-sad-effect](../../150-sad-effect/) the daylight/mood family — this study "
            "owns the specific **11-year sunspot-cycle → returns** (Jevons) axis, and "
            "nothing else on the desk tests it.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
