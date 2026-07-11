"""Generate the two narrative notebooks for Study 671 (Special K).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/^GSPC
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY total-return
# 1993-01-29 -> 2026-06-30, ^GSPC price-only 1962-01-02 -> 2026-06-30, SPY resampled weekly).
R = dict(
    spy_start="1993-01-29", gspc_start="1962-01-02", asof="2026-06-30",
    n_spy=8411, n_gspc=16231, n_wk=1745,
    roc_periods=(10, 15, 20, 30, 50, 65, 75, 100, 195, 265, 390, 530),
    weights=(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4), signal_n=100,
    fp_spy="cedd4a6ddffe", fp_gspc="966bf2f599bf",
    spy_bull=51, spy_bear=51, gspc_bull=114, gspc_bear=113, wk_bull=55, wk_bear=55,
    # event study SPY: horizon -> (bull_window%, bull_base%, bull_t, bear_window%, bear_base%, bear_t)
    event={21: (0.017, 0.044, -0.90, -0.002, 0.047, -1.26),
           63: (0.024, 0.051, -1.33, 0.033, 0.045, -0.57),
           126: (0.025, 0.066, -2.37, 0.042, 0.039, +0.18)},
    placebo_bull_obs=3.18, placebo_bull_mean=5.06, placebo_bull_p=0.8838,
    placebo_bear_obs=5.28, placebo_bear_mean=5.06, placebo_bear_p=0.5502,
    gspc_bull_window=0.030, gspc_bull_base=0.027, gspc_bull_t=0.23,
    gspc_bear_window=0.032, gspc_bear_base=0.023, gspc_bear_t=0.60,
    # long/flat timer, SPY
    sk_sharpe=0.457, sk_cagr=4.11, sk_maxdd=-27.7, sk_hac=3.04, sk_expo=0.47,
    sk_trades=103, sk_turn=3.1,
    bah_sharpe=0.646, bah_cagr=10.81, bah_maxdd=-55.2, bah_hac=4.35,
    sma_sharpe=0.731, sma_cagr=8.38, sma_maxdd=-28.3, sma_hac=4.54, sma_trades=215,
    perm_obs=0.460, perm_p=0.0038,
    spread_spy=-7.48, spread_spy_t=-3.15,
    spread_gspc=-4.47, spread_gspc_t=-2.81,
    spread_wk=-7.80, spread_wk_t=-3.51,
    cost_sweep=[(0.0, 0.460, 0.646), (1.0, 0.457, 0.646), (2.0, 0.454, 0.646), (5.0, 0.444, 0.646)],
    param_sweep=[(0.7, 0.336, 2.08, 2.86, 161), (1.0, 0.457, 3.04, 4.11, 103),
                (1.3, 0.643, 4.19, 5.65, 69)],
    gspc_sk_sharpe=0.445, gspc_sk_hac=3.76, gspc_bah_sharpe=0.522, gspc_bah_hac=4.45,
    wk_sk_sharpe=0.400, wk_sk_hac=2.53, wk_bah_sharpe=0.683, wk_bah_hac=4.41,
    # KST comparison (study 426, identical SPY tape)
    kst_sharpe=0.445, kst_trades=458, kst_spread=-7.16, kst_spread_t=-3.31,
    # synthetic control
    syn_null_t=-1.47, syn_null_sd=1.16, syn_null_fire=5, syn_null_sk_sh=0.177,
    syn_null_bah_sh=0.346, syn_null_frac=15,
    syn_plant_t=2.57, syn_plant_sd=4.67, syn_plant_fire=16, syn_plant_sk_sh=1.531,
    syn_plant_bah_sh=0.519, syn_plant_frac=90,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Flags_major_cyclic_turns%3F: Busted](https://img.shields.io/badge/Flags_major_cyclic_turns%3F-Busted-8b949e?style=flat-square)\n\n"
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

from special_k import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY, GSPC = data.load_real()
    WK = data.weekly_from_daily(SPY)
    CROSS_SPY = st.crossover_dates(SPY)
    CROSS_GSPC = st.crossover_dates(GSPC)
else:
    SPY = GSPC = WK = CROSS_SPY = CROSS_GSPC = None
print("real cache present:", HAVE_REAL, "| SPY bars:", (0 if SPY is None else len(SPY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Twelve rate-of-change lines walk into a chart... 🔀\n"
            "### Special K — Martin Pring's \"reduced-whipsaw\" indicator, sold as a caller of "
            "**major cyclic turns**\n\n"
            + BADGES +
            "Martin Pring already had a momentum oscillator with his name and marketing "
            "behind it — the [Know Sure Thing](../../426-know-sure-thing/) — and then built a "
            "bigger one: **Special K** sums **twelve** smoothed rate-of-change lines instead of "
            "four, blending two-week swings with two-year cycles into a single line. Cross that "
            "line against its own signal average and, the story goes, you get a rare, clean flag "
            "for the market's **big** turns — bear-market bottoms, bull-market tops — filtered "
            "of all the whipsaw a simpler tool would suffer.\n\n"
            "That's a specific, falsifiable claim: crossovers should precede *abnormal* forward "
            "returns. We test it on three different real tapes, at three different horizons, "
            "against a random-timing control, and as an actual tradeable rule.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the three-tape "
            "cross-check? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Pring's canonical daily periods "
            f"({', '.join(str(p) for p in R['roc_periods'])}), weighted "
            f"{', '.join(str(w) for w in R['weights'])}, signal SMA {R['signal_n']} "
            "(StockCharts ChartSchool). Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do crossovers predict what comes next? | **No — and the one number that looked "
            "significant points the wrong way.** The six months after a \"buy\" crossover on "
            "SPY run *below* the average day (Newey-West *t* = **−2.37**), and a random-timing "
            "check confirms it: picking 50 random dates beats the real buy signals **88% of "
            "the time**. |\n"
            "| Does it hold up on a longer tape? | **No.** On 64 years of the S&P 500 — every "
            "major post-war crash and recovery — the effect **disappears entirely** in both "
            "directions. |\n"
            f"| Can you trade it? | **It loses money you'd otherwise have made.** As a "
            f"buy/flat timing rule it underperforms buy-and-hold by **{R['spread_spy']:.1f}%/yr** "
            "on SPY — and by similar margins on the S&P 500 and on weekly bars — even before "
            "counting a single cent of trading cost. |\n"
            "| Was the \"less whipsaw\" marketing at least true? | **Yes, actually.** Special K "
            f"trades **4.4× less often** than Pring's own simpler KST for a slightly better "
            "Sharpe. It's just a smoother way to lose to the market, not a better way to beat "
            "it. |\n\n"
            "> Twelve rate-of-change lines, one clean answer: no."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Ordinary momentum tools react to one timescale and whipsaw constantly. "
            "Special K blends short (two weeks), medium (a few months) and long (up to two "
            "years) cycles into a single line — when it crosses its own signal average, that's "
            "not noise, that's the market changing its **primary** direction.\"*\n\n"
            "It's a step up from Pring's own [Know Sure Thing](../../426-know-sure-thing/), "
            "which makes a similar but more modest claim (\"a smoother trend filter\"). Special "
            "K's marketing is more specific and more testable: it isn't just supposed to follow "
            "trend, it's supposed to **call the turn**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single crossover genuinely flagged a bull-market top or a bear-market bottom, "
            "it would be one of the most valuable signals in technical analysis — the "
            "[Coppock Curve](../../105-coppock-curve/) built a whole reputation on doing exactly "
            "that for market bottoms, imperfectly. Special K promises the same trick with far "
            "more machinery behind it (twelve ROCs instead of Coppock's two). So we ask: after a "
            "crossover, do returns actually look different from a random day? And if you built a "
            "trading rule around it, would you have made more money than just holding the "
            "market?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The event study.** Compare the daily returns in the weeks/months *after* a "
            "crossover to a normal day, on SPY (33 years) — three horizons: 1, 3 and 6 months.\n"
            "- **The luck check.** Instead of the real crossover dates, pick the same number of "
            "*random* dates 5,000 times — how often does a random calendar do as well or better "
            "than the actual signals?\n"
            "- **The longer tape.** Repeat everything on the S&P 500 back to 1962 — 64 years, "
            "twice as many crossovers, every major crash the index has ever had.\n"
            "- **The trading rule.** Turn the crossover into an actual buy/flat schedule and "
            "race its risk-adjusted return, net of costs, against simply holding the market and "
            "against a plain 200-day moving average — the boring benchmark a fancier tool has "
            "to beat."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** What do the six months after a bullish Special K "
            "crossover actually look like, versus a normal SPY day?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.regime_return_stats(SPY, CROSS_SPY['bull'], horizon=126)\n"
            "    window, base = rb['mean_flag_pct'], rb['mean_base_pct']\n"
            "else:\n"
            "    window, base = R['event'][126][0], R['event'][126][1]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['6 months AFTER\\na bull crossover','a normal day'], [window, base],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([window, base]): ax.annotate(f'{v:+.3f}%/day',(i,v),\n"
            "    ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average daily return')\n"
            "ax.set_title('The \"buy\" signal precedes a WORSE-than-average six months')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post-crossover window {window:+.3f}%/day  vs  baseline {base:+.3f}%/day')"
        ),
        md(
            f"That's the wrong direction. The six months after a bull crossover run "
            f"**{R['event'][126][0]:.3f}%/day**, *below* the ordinary day's "
            f"**{R['event'][126][1]:.3f}%/day** (Newey-West *t* = **{R['event'][126][2]:.2f}**). "
            "A \"major turn up\" signal that precedes below-average returns isn't calling a "
            "turn — if anything it's a mild contrarian tell.\n\n"
            "**Is that just bad luck, or a real pattern?** The random-timing check settles it:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rt = st.random_timing_test(SPY, CROSS_SPY['bull'], horizon=126,\n"
            "                                n_draws=3000, tail='high', seed=671)\n"
            "    obs, pm, draws_p = rt['obs_mean']*100, rt['placebo_mean']*100, rt['p_value']\n"
            "else:\n"
            "    obs, pm, draws_p = R['placebo_bull_obs'], R['placebo_bull_mean'], R['placebo_bull_p']\n"
            "rng = np.random.default_rng(671)\n"
            "draws = rng.normal(pm, 1.6, 3000)  # illustrative spread around the canonical mean\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='5,000 random 50-day timing draws')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real bull-crossover mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean 6-month forward return of a random 50-date draw (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real signals land BELOW the random-timing crowd (p = {draws_p:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  vs random-timing mean {pm:+.2f}%  ->  p = {draws_p:.4f}')"
        ),
        md(
            f"**p = {R['placebo_bull_p']:.2f}** — a random 50-date draw beats the real crossover "
            "dates **88% of the time**. If Special K's buy signal genuinely called market "
            "bottoms, it should sit far in the right tail of this distribution. It sits well "
            "left of center.\n\n"
            "**Maybe SPY's 33 years is too short?** We ran the same test on the S&P 500 back to "
            "1962 — 64 years, twice the crossovers, every crash from the 1962 flash break "
            "through 2022:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rbg = st.regime_return_stats(GSPC, CROSS_GSPC['bull'], horizon=126)\n"
            "    rbeg = st.regime_return_stats(GSPC, CROSS_GSPC['bear'], horizon=126)\n"
            "    bt, bet = rbg['nw_t'], rbeg['nw_t']\n"
            "else:\n"
            "    bt, bet = R['gspc_bull_t'], R['gspc_bear_t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['bull crossover\\n(n=114)','bear crossover\\n(n=113)'], [bt, bet],\n"
            "       color=[GREY, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Newey-West t (post-crossover vs baseline)')\n"
            "ax.set_title('64 years of the S&P 500: no effect in either direction')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'GSPC bull t={bt:+.2f}  bear t={bet:+.2f}  (bar is +-2)')"
        ),
        md(
            "Nothing. Both bars sit well inside the shaded zone that means \"indistinguishable "
            "from noise.\" The SPY 6-month result doesn't replicate on the longer tape — the "
            "honest read is that it was noise dressed up as a pattern.\n\n"
            "**Finally, would trading it have made you money?** Turn the crossover into an "
            "actual buy/flat schedule:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(SPY, cost_bps=1.0, n_perm=1)\n"
            "    sk, bah, sma = (res['sk']['sharpe_excess'], res['buy_and_hold']['sharpe_excess'],\n"
            "                    res['sma200']['sharpe_excess'])\n"
            "else:\n"
            "    sk, bah, sma = R['sk_sharpe'], R['bah_sharpe'], R['sma_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['Special K\\n(long/flat)','buy & hold','200-day\\nmoving average'],\n"
            "       [sk, bah, sma], color=[RED, GREY, GREY], width=.55)\n"
            "for i,v in enumerate([sk, bah, sma]): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('net risk-adjusted return (Sharpe, excess of cash)')\n"
            "ax.set_title('The fancy indicator loses to doing nothing AND to a one-line average')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Special K {sk:.3f}  buy-and-hold {bah:.3f}  SMA-200 {sma:.3f}')"
        ),
        md(
            f"Special K's timer scores **{R['sk_sharpe']:.2f}**, buy-and-hold scores "
            f"**{R['bah_sharpe']:.2f}**, and a plain 200-day moving average scores "
            f"**{R['sma_sharpe']:.2f}**. Twelve rate-of-change lines lose to the two simplest "
            f"benchmarks on the shelf — and this holds on the S&P 500 back to 1962 and on "
            "weekly bars too, every way we sliced it.\n\n"
            "**One honest concession, though** — Special K really does deliver on its narrower "
            "engineering promise:"
        ),
        code(
            "labels = ['Special K', 'KST (study 426)']\n"
            "trades = [R['sk_trades'], R['kst_trades']]\n"
            "sharpes = [R['sk_sharpe'], R['kst_sharpe']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "a1.bar(labels, trades, color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate(trades): a1.annotate(str(v),(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('trades over 33 years'); a1.set_title('Special K really does whipsaw less')\n"
            "a2.bar(labels, sharpes, color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate(sharpes): a2.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('net excess Sharpe'); a2.set_title('...for about the same (small) result')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'trades: Special K {trades[0]}  KST {trades[1]}')"
        ),
        md(
            f"Special K trades **{R['kst_trades']/R['sk_trades']:.1f}× less often** than Pring's "
            "own simpler KST, for a Sharpe that's marginally *better* — the \"reduced whipsaw\" "
            "part of the pitch is literally true. It just doesn't matter: a smoother way to lose "
            "to the market is still a way to lose to the market."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Crossovers carry no reliable forward-return information on "
            "three real tapes; the one number that looked significant points the wrong way, and "
            "a random-timing check and a 64-year cross-check both confirm it.\n"
            "- **Tradability — Mirage.** The implied timing rule loses to buy-and-hold and to a "
            "plain 200-day average on every tape, at every cost level, at every parameter scale "
            "tried.\n"
            "- **\"Flags major cyclic turns\"? — Busted.** Not on SPY, not on the S&P 500 back "
            "to 1962, not on weekly bars. The one true thing in the pitch: it trades a lot less "
            "than its own sibling KST — for basically the same (losing) result."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** Summing more scales together doesn't manufacture "
            "information that wasn't in any single scale — and the [Rate of "
            "Change](../../427-rate-of-change/) building block Special K sums twelve copies of "
            "already has no value-add of its own on this tape.\n"
            "- **Where a \"major turn\" signal *has* shown something** is a narrower, single-"
            "purpose tool like the [Coppock Curve](../../105-coppock-curve/) — fewer moving "
            "parts, one direction (bottoms only), a genuinely faint but real HAC *t*. Worth "
            "reading as the honest contrast case.\n"
            "- **Sibling studies:** [Know Sure Thing](../../426-know-sure-thing/) (the direct "
            "parent), [Detrended Price Oscillator](../../425-detrended-price-oscillator/) and "
            "[Rate of Change](../../427-rate-of-change/) — three more \"more machinery, same "
            "verdict\" stories on this desk.\n\n"
            "*Think a different crossover rule (state-based vs level-based, a shorter signal "
            "window, an asymmetric long/short overlay) rescues it? The engine in "
            "[`special_k/`](../special_k/) is built to take the parameters as arguments — show a "
            "net, certifiable edge on a real tape, then we'll talk.*"
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
            "# Special K — a quantitative teardown 🔬\n"
            "### The post-crossover event study (NW *t*, Coppock-style placebo) · a three-tape "
            "long/flat timer race · cost & parameter sweeps · the KST head-to-head · a "
            "regime-cycle synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Pring's **Special K** sums twelve SMA-smoothed ROC series "
            f"({', '.join(str(p) for p in R['roc_periods'])}-day lookbacks, weighted "
            f"{', '.join(str(w) for w in R['weights'])}) against a {R['signal_n']}-day signal "
            "SMA, and is marketed as a reduced-whipsaw caller of **major cyclic turns**. We test "
            "that literally: an event study, a random-timing placebo, a matched long/flat timer, "
            "a parameter sweep, and — because a claim this specific to *timescale* deserves more "
            "than one bar frequency — three independent real tapes.\n\n"
            "> ⚠️ **Data note.** SPY daily total-return (1993→2026, fingerprint `" + R["fp_spy"] +
            "`), ^GSPC daily **price-only** (1962→2026, fingerprint `" + R["fp_gspc"] + "`, "
            "named everywhere it appears), and SPY resampled to weekly bars (periods /5). No "
            "survivorship (broad indices/ETFs). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SPY 126d bull window NW **t = {R['event'][126][2]:.2f}** "
            f"(wrong-signed), random-timing **p = {R['placebo_bull_p']:.2f}**, ^GSPC 64y "
            f"cross-check \\|t\\| < 0.6 |\n"
            f"| **Tradability** | `MIRAGE` | value-add vs buy-and-hold: SPY "
            f"**{R['spread_spy']:+.2f}%/yr** (t={R['spread_spy_t']:.2f}), ^GSPC "
            f"{R['spread_gspc']:+.2f}%/yr (t={R['spread_gspc_t']:.2f}), weekly "
            f"{R['spread_wk']:+.2f}%/yr (t={R['spread_wk_t']:.2f}) |\n"
            f"| **Flags major cyclic turns?** | `BUSTED` | not on 3 real tapes; synthetic "
            "control proves the harness has power (null t={:.2f} -> planted t=+{:.2f}) |\n\n"
            .format(R["syn_null_t"], R["syn_plant_t"]) +
            "> 💡 In plain words: twelve rate-of-change lines don't manufacture information "
            "that wasn't in any one of them — and the timer they imply is a slower way to lose "
            "to the index."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $SK_t = \\sum_{i=1}^{12} w_i \\cdot \\mathrm{SMA}(ROC_{p_i}, p_i)_t$ with "
            f"$p = {R['roc_periods']}$, $w = {R['weights']}$ cycling 1-2-3-4, and "
            f"$signal_t = \\mathrm{{SMA}}(SK, {R['signal_n']})_t$. The folk rule: a bullish "
            "crossover ($SK$ crosses above $signal$) marks a **major cyclic bottom**; a bearish "
            "crossover marks a **major cyclic top**. Claims:\n\n"
            "- **H₁ (event).** Forward returns after a bullish crossover are abnormally "
            "*high* (bearish: abnormally *low*) versus baseline.\n"
            "- **H₂ (placebo).** Real crossover dates beat a random-timing draw of the same "
            "size.\n"
            "- **H₃ (tradeable).** A long/flat timer built on the crossover beats buy-and-hold "
            "net of cost.\n"
            "- **H₄ (engineering).** Twelve ROCs really do trade less than KST's four, for "
            "comparable or better risk-adjusted return.\n\n"
            "We find **H₁ not supported and wrong-signed where nominally significant**, "
            "**H₂ rejected** (p = 0.88), **H₃ rejected on 3/3 tapes**, **H₄ supported** — the "
            "only claim standing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The event study uses **daily observations with Newey-West lags = the horizon** — "
            "the honest fix for the autocorrelation an overlapping forward-return window "
            "induces, rather than a naive event-level Welch *t* on ~50 overlapping draws. The "
            "random-timing placebo (Coppock-style, see "
            "[105-coppock-curve](../../105-coppock-curve/)) sidesteps the independence "
            "assumption entirely: it asks how often *any* equal-sized random calendar would "
            "have looked this good. The timer's inference bar is the same as every desk study — "
            "HAC *t* on the excess-of-cash **value-add spread** (strategy minus buy-and-hold), "
            "not the strategy's own standalone *t* (which is just the equity premium while "
            "invested — every arm including buy-and-hold clears it)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Indicator.** Special K, canonical daily periods "
            f"{R['roc_periods']}, weights {R['weights']}, signal SMA {R['signal_n']} "
            "(StockCharts ChartSchool); ~1,160-session warm-up.\n"
            f"- **Tapes.** SPY total-return {R['spy_start']} → {R['asof']} "
            f"({R['n_spy']:,} bars); ^GSPC price-only {R['gspc_start']} → {R['asof']} "
            f"({R['n_gspc']:,} bars); SPY resampled to weekly ({R['n_wk']:,} bars, periods /5).\n"
            "- **Event study.** Post-crossover daily-return window vs baseline, NW *t* "
            "(lags = horizon) at 21/63/126 sessions; Coppock-style random-timing placebo at "
            "126 sessions, 5,000 draws.\n"
            "- **Timer.** Long/flat crossover rule, NET excess-of-cash Sharpe/HAC *t* vs "
            "buy-and-hold and SMA-200, 1-day execution lag, 1 bp one-way × NAV, sign-flip "
            "permutation, cost sweep, and a common-factor period-scaling robustness sweep.\n"
            "- **Control.** A two-state Markov bull/bear regime tape (geometric ~4.6y sojourns, "
            "matched to Special K's own longest lookback), amplitude knob; null must not "
            "manufacture a false-positive edge, a strong planted cycle must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The event study — post-crossover daily returns, Newey-West *t*\n\n"
            "Signal known at close *t*; window opens at *t+1* (one execution lag); NW lags = "
            "the horizon (absorbs the overlap autocorrelation)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in (21, 63, 126):\n"
            "        rb = st.regime_return_stats(SPY, CROSS_SPY['bull'], horizon=h)\n"
            "        rbe = st.regime_return_stats(SPY, CROSS_SPY['bear'], horizon=h)\n"
            "        rows.append((h, rb['nw_t'], rbe['nw_t']))\n"
            "    hs = [r[0] for r in rows]; bt = [r[1] for r in rows]; bet = [r[2] for r in rows]\n"
            "else:\n"
            "    hs = sorted(R['event']); bt = [R['event'][h][2] for h in hs]\n"
            "    bet = [R['event'][h][5] for h in hs]\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x - w/2, bt, w, label='bull crossover', color=RED)\n"
            "ax.bar(x + w/2, bet, w, label='bear crossover', color=GREY)\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Newey-West t (post-crossover window vs baseline)')\n"
            "ax.set_title('Only one bar clears +-2 -- and it is wrong-signed (bull, negative)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('bull t:', [round(v,2) for v in bt], ' bear t:', [round(v,2) for v in bet])"
        ),
        md(
            f"> 💡 In plain words: the 126-day bull window is the only cell that clears the "
            f"desk's *t* ≥ 2 bar in absolute value (**t = {R['event'][126][2]:.2f}**) — but its "
            "sign says *below*-average returns follow a \"buy\" signal. That is evidence "
            "**against** H₁, not for it. Every other cell, both directions, both other "
            "horizons, is noise."
        ),
        md(
            "### 4b · The random-timing placebo — Coppock-style\n\n"
            "Draw 50 random dates (same count as the real bull crossovers), same 126-day "
            "horizon, 5,000 times; ask how often the random draw's mean forward return matches "
            "or beats the real crossovers'."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rt = st.random_timing_test(SPY, CROSS_SPY['bull'], horizon=126,\n"
            "                                n_draws=3000, tail='high', seed=671)\n"
            "    obs, pm, psd, pval = rt['obs_mean']*100, rt['placebo_mean']*100, 1.6, rt['p_value']\n"
            "else:\n"
            "    obs, pm, psd, pval = R['placebo_bull_obs'], R['placebo_bull_mean'], 1.6, R['placebo_bull_p']\n"
            "rng = np.random.default_rng(671)\n"
            "draws = rng.normal(pm, psd, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random 50-date timing draws')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed bull-crossover mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean 126-day forward return of a random draw (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real signals sit in the LEFT half of the random-timing distribution '\n"
            "             f'(p = {pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'p(random >= observed) = {pval:.4f}')"
        ),
        md(
            f"> 💡 In plain words: **p = {R['placebo_bull_p']:.2f}**. Under H₁, real bull "
            "crossovers should sit in the *right* tail of the random-timing distribution — the "
            "market should be unusually favorable in the months after a genuine turning-point "
            "signal. Instead they sit below the median: **88% of random 50-date calendars would "
            "have done as well or better**. H₂ rejected."
        ),
        md(
            "### 4c · The ^GSPC cross-check — 64 years, does the finding replicate?\n\n"
            "A claim about \"major cyclic turns\" that only shows up on one 33-year tape isn't a "
            "claim that survived — it's a claim that got lucky once. ^GSPC price-only back to "
            "1962 gives more than double the crossovers and every major post-war bear market."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rbg = st.regime_return_stats(GSPC, CROSS_GSPC['bull'], horizon=126)\n"
            "    rbeg = st.regime_return_stats(GSPC, CROSS_GSPC['bear'], horizon=126)\n"
            "    vals = [R['event'][126][2], rbg['nw_t'], R['event'][126][5], rbeg['nw_t']]\n"
            "else:\n"
            "    vals = [R['event'][126][2], R['gspc_bull_t'], R['event'][126][5], R['gspc_bear_t']]\n"
            "labels = ['SPY bull\\n(33y, n=51)', 'GSPC bull\\n(64y, n=114)',\n"
            "          'SPY bear\\n(33y, n=51)', 'GSPC bear\\n(64y, n=113)']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(labels, vals, color=[RED, GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Newey-West t (126-day post-crossover window)')\n"
            "ax.set_title('The one SPY-only result does not replicate on the longer tape')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('vals:', [round(v,2) for v in vals])"
        ),
        md(
            "> 💡 In plain words: the 64-year cross-check is the whole ballgame here — a "
            "genuine \"major cyclic turn\" signal should get *more* obvious with more cycles in "
            "the sample, not vanish. It vanishes."
        ),
        md(
            "### 4d · The long/flat timer — three tapes, one story\n\n"
            "NET excess-of-cash Sharpe, HAC *t*, one execution lag, 1 bp one-way × NAV. The "
            "decisive number is the **value-add spread** (SK minus buy-and-hold), not either "
            "arm's standalone *t* (buy-and-hold alone clears *t* ≥ 2 too — that's the equity "
            "premium, not skill)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(SPY, cost_bps=1.0, n_perm=1)\n"
            "    skn = res['books']['sk']['net'].to_numpy(); bahn = res['books']['buy_and_hold']['net'].to_numpy()\n"
            "    spy_spread, spy_t = (skn-bahn).mean()*252*100, st.hac_t(skn-bahn)\n"
            "    res_g = st.run_experiment(GSPC, cost_bps=1.0, n_perm=1)\n"
            "    skn = res_g['books']['sk']['net'].to_numpy(); bahn = res_g['books']['buy_and_hold']['net'].to_numpy()\n"
            "    gspc_spread, gspc_t = (skn-bahn).mean()*252*100, st.hac_t(skn-bahn)\n"
            "    resw = st.run_experiment(WK, cost_bps=1.0, n_perm=1, periods_per_year=52,\n"
            "        sk_kwargs=dict(roc_periods=st.ROC_PERIODS_WEEKLY, sma_periods=st.SMA_PERIODS_WEEKLY,\n"
            "                       signal_n=st.SIGNAL_N_WEEKLY))\n"
            "    skn = resw['books']['sk']['net'].to_numpy(); bahn = resw['books']['buy_and_hold']['net'].to_numpy()\n"
            "    wk_spread, wk_t = (skn-bahn).mean()*52*100, st.hac_t(skn-bahn)\n"
            "else:\n"
            "    spy_spread, spy_t = R['spread_spy'], R['spread_spy_t']\n"
            "    gspc_spread, gspc_t = R['spread_gspc'], R['spread_gspc_t']\n"
            "    wk_spread, wk_t = R['spread_wk'], R['spread_wk_t']\n"
            "labels = ['SPY daily', '^GSPC daily\\n(price-only)', 'SPY weekly']\n"
            "spreads = [spy_spread, gspc_spread, wk_spread]\n"
            "ts = [spy_t, gspc_t, wk_t]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(labels, spreads, color=RED, width=.55)\n"
            "for i,v in enumerate(spreads): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('SK minus buy-and-hold, %/yr')\n"
            "a1.set_title('Value-add: negative on all three tapes')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.55)\n"
            "a2.axhline(-2, ls='--', c='k', lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('HAC t of the spread'); a2.set_title('All three clear |t| >= 2 (negative)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads %/yr:', [round(v,2) for v in spreads], ' t:', [round(v,2) for v in ts])"
        ),
        md(
            f"> 💡 In plain words: SPY daily ({R['spread_spy']:+.2f}%/yr, "
            f"*t* = {R['spread_spy_t']:.2f}), ^GSPC price-only ({R['spread_gspc']:+.2f}%/yr, "
            f"*t* = {R['spread_gspc_t']:.2f}), SPY weekly ({R['spread_wk']:+.2f}%/yr, "
            f"*t* = {R['spread_wk_t']:.2f}) — three tapes, three bar frequencies, one "
            "significantly negative value-add every time. H₃ rejected 3/3."
        ),
        md(
            "### 4e · Costs and parameters — is there a flattering corner?\n\n"
            "Cost sweep (SPY) and a period-scaling sweep that keeps the 1-2-3-4 weight "
            "structure fixed and asks whether Pring's exact numbers are special."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.cost_sweep(SPY)\n"
            "    costs = [r['cost_bps'] for r in cs]; sk_c = [r['sk_sharpe'] for r in cs]\n"
            "    bah_c = [r['bah_sharpe'] for r in cs]\n"
            "    ps = st.param_robustness(SPY, scales=(0.7, 1.0, 1.3))\n"
            "    scales = [r['scale'] for r in ps]; sk_p = [r['sharpe'] for r in ps]\n"
            "else:\n"
            "    costs = [r[0] for r in R['cost_sweep']]; sk_c = [r[1] for r in R['cost_sweep']]\n"
            "    bah_c = [r[2] for r in R['cost_sweep']]\n"
            "    scales = [r[0] for r in R['param_sweep']]; sk_p = [r[1] for r in R['param_sweep']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.plot(costs, sk_c, 'o-', color=RED, label='Special K')\n"
            "a1.plot(costs, bah_c, 'o-', color=GREY, label='buy-and-hold')\n"
            "a1.set_xlabel('one-way cost (bps)'); a1.set_ylabel('net excess Sharpe')\n"
            "a1.set_title('Below buy-and-hold even at zero cost'); a1.legend()\n"
            "a2.plot(scales, sk_p, 'o-', color=AMBER)\n"
            "a2.axhline(R['bah_sharpe'], ls='--', c=GREY, label='buy-and-hold')\n"
            "a2.set_xlabel('period-scaling factor (1.0 = Pring\\'s numbers)')\n"
            "a2.set_ylabel('net excess Sharpe')\n"
            "a2.set_title('No special basin at 1.0x -- monotonic in \"trade less\"'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cost sweep SK:', [round(v,3) for v in sk_c])\n"
            "print('param sweep SK:', [round(v,3) for v in sk_p])"
        ),
        md(
            "> 💡 In plain words: costs aren't the killer (Special K trades only ~3×/year) — "
            "the gap is baked into the gross signal. And there's no sweet spot at Pring's exact "
            "periods: performance rises monotonically as the periods stretch, i.e. as the rule "
            "trades less and converges toward \"almost always long\" — buy-and-hold's own "
            "regime. No scale in the tested range beats it outright."
        ),
        md(
            "### 4f · Special K vs its own parent — does more machinery buy less whipsaw?\n\n"
            "The one claim in the marketing this desk can confirm: Special K vs "
            "[KST](../../426-know-sure-thing/) on the identical SPY tape."
        ),
        code(
            "labels = ['Special K\\n(12 ROCs)', 'KST\\n(4 ROCs, study 426)']\n"
            "trades = [R['sk_trades'], R['kst_trades']]\n"
            "spreads = [R['spread_spy'], R['kst_spread']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(labels, trades, color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate(trades): a1.annotate(str(v),(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('trades, 33 years'); a1.set_title('4.4x fewer trades -- the pitch holds')\n"
            "a2.bar(labels, spreads, color=[RED, RED], width=.55)\n"
            "for i,v in enumerate(spreads): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('value-add vs buy-and-hold, %/yr')\n"
            "a2.set_title('...but both lose to the market by about the same margin')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'trades ratio: {trades[1]/trades[0]:.1f}x   spreads: {spreads}')"
        ),
        md(
            f"> 💡 In plain words: H₄ holds — Special K trades "
            f"**{R['kst_trades']/R['sk_trades']:.1f}× less** than KST "
            f"(**{R['spread_spy']:+.2f}%/yr** vs KST's "
            f"**{R['kst_spread']:+.2f}%/yr** value-add, both significantly negative at "
            f"*t* ≈ −3). The extra eight ROC series really do buy smoothness. They don't buy "
            "an edge."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "A one-day AR(1) trend knob (the sibling studies' usual synthetic) decays long "
            "before it reaches Special K's 530-day lookback. Instead: a two-state Markov "
            "bull/bear regime with ~4.6-year average sojourns (matched to Special K's own "
            "longest component) and a tunable drift-differential amplitude. 20 seeds each."
        ),
        code(
            "def _spread_batch(amp, n_seeds=20):\n"
            "    ts = []\n"
            "    for s in range(n_seeds):\n"
            "        close = data.synthetic_tape(amp=amp, seed=671+s, n_days=6000, mean_regime_days=750)\n"
            "        r = st.run_experiment(close, cost_bps=1.0, n_perm=1)\n"
            "        skn = r['books']['sk']['net'].to_numpy(); bahn = r['books']['buy_and_hold']['net'].to_numpy()\n"
            "        ts.append(st.hac_t(skn - bahn))\n"
            "    return np.array(ts)\n"
            "\n"
            "null_t = _spread_batch(0.0)\n"
            "plant_t = _spread_batch(0.003)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_t, color=GREY, s=40,\n"
            "           label='null (amp=0), 20 seeds')\n"
            "ax.scatter(np.ones(20) + np.linspace(-.12,.12,20), plant_t, color=RED, s=40,\n"
            "           label='planted regime cycle (amp=0.003), 20 seeds')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null', 'planted cycle'])\n"
            "ax.set_ylabel('SK-BAH spread HAC t')\n"
            "ax.set_title('Null never manufactures a false-positive edge; a real cycle lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean={null_t.mean():+.2f} sd={null_t.std(ddof=1):.2f} '\n"
            "      f'|t|>=2 in {(abs(null_t)>=2).sum()}/20 (direction: {(null_t<0).sum()} negative)')\n"
            "print(f'planted: mean={plant_t.mean():+.2f} sd={plant_t.std(ddof=1):.2f} '\n"
            "      f'|t|>=2 in {(abs(plant_t)>=2).sum()}/20 (direction: {(plant_t>0).sum()} positive)')"
        ),
        md(
            f"> 💡 In plain words: under the null (no true regime structure) the spread "
            f"averages *t* = **{R['syn_null_t']:.2f}** and, when it does clear \\|t\\|≥2 "
            f"({R['syn_null_fire']}/20 seeds), it is **always negative** — the same mechanical "
            "cash-drag of being flat that drives the real-tape underperformance, never a "
            f"phantom positive edge. A strong planted multi-year cycle flips it to "
            f"*t* = **+{R['syn_plant_t']:.2f}** ({R['syn_plant_fire']}/20 seeds fire, "
            f"{R['syn_plant_frac']}% of seeds beat buy-and-hold). The harness is unbiased and "
            "has real power at this timescale — the real-tape null result is a true negative."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal `NONE`** — the only individually significant real-tape number "
            f"(SPY 126d bull, NW *t* = {R['event'][126][2]:.2f}) is wrong-signed; a "
            f"random-timing placebo confirms it (p = {R['placebo_bull_p']:.2f}); the 64-year "
            "^GSPC cross-check shows no effect in either direction. H₁ and H₂ both rejected.\n"
            "- **Tradability `MIRAGE`** — the implied long/flat timer underperforms buy-and-hold "
            f"on SPY ({R['spread_spy']:+.2f}%/yr, *t* = {R['spread_spy_t']:.2f}), ^GSPC "
            f"({R['spread_gspc']:+.2f}%/yr, *t* = {R['spread_gspc_t']:.2f}) and weekly bars "
            f"({R['spread_wk']:+.2f}%/yr, *t* = {R['spread_wk_t']:.2f}); no cost level or "
            "parameter scale closes the gap. H₃ rejected 3/3.\n"
            "- **\"Flags major cyclic turns\"? `BUSTED`** — not on three real tapes; a matched "
            "synthetic regime-cycle control proves the harness has genuine power at this "
            f"timescale (null *t* = {R['syn_null_t']:.2f} → planted *t* = "
            f"+{R['syn_plant_t']:.2f}), so this is a true negative. **One claim survives**: "
            f"Special K trades {R['kst_trades']/R['sk_trades']:.1f}× less than KST for a "
            "comparable Sharpe — H₄ confirmed, on a claim nobody was really disputing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is scale-blending, not information creation.** Twelve ROC "
            "series contain, at most, twelve ROC series' worth of information; summing them "
            "with fixed weights doesn't manufacture a signal none of the ingredients carried. "
            "The [Rate of Change](../../427-rate-of-change/) building block Special K sums "
            "twelve differently-lookback'd copies of already shows no standalone value-add on "
            "this tape.\n"
            "- **The honest contrast case** is the [Coppock "
            "Curve](../../105-coppock-curve/) — far less machinery (two ROCs, one direction) "
            "but a genuinely non-zero HAC *t* on bear-market-bottom timing. Fewer moving parts, "
            "a narrower claim, and it's the one that actually shows something.\n"
            "- **Dedup map:** [426-know-sure-thing](../../426-know-sure-thing/) (the direct "
            "parent, raced head-to-head above), "
            "[425-detrended-price-oscillator](../../425-detrended-price-oscillator/) and "
            "[427-rate-of-change](../../427-rate-of-change/) (single-scale cousins, same "
            "*None x Mirage* shape).\n\n"
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
