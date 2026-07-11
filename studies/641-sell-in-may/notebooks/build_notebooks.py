"""Generate the two narrative notebooks for Study 641 (Sell in May).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^GSPC/SPY/
^SP500TR/^IRX tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^GSPC 1950-01-03 ->
# 2026-06-30 price-only, SPY 1993-02-01 ->, ^SP500TR 1988-01-05 ->, ^IRX 1960-01-04 ->).
R = dict(
    as_of="2026-06-30",
    fp_gspc="f17f1fc81bc9", fp_spy="8324508d6a89", fp_tr="02c7f8f5728e", fp_irx="20827eb9627f",
    # headline split: tape -> (n_winter, n_summer, winter_pct, summer_pct, gap_pct,
    #                          winter_half_pct, summer_half_pct, welch_t, nw_t)
    gspc=dict(n_w=459, n_s=458, winter=1.048, summer=0.278, gap=0.770,
              winter_half=6.49, summer_half=1.68, welch=2.78, nw=2.96),
    sp500tr=dict(n_w=231, n_s=230, winter=1.168, summer=0.643, gap=0.526,
                 winter_half=7.26, summer_half=3.93, welch=1.33, nw=1.42),
    spy=dict(n_w=200, n_s=200, winter=1.092, summer=0.619, gap=0.473,
             winter_half=6.77, summer_half=3.78, welch=1.10, nw=1.18),
    # year-block pairing: tape -> (n_years, k_wins, hit_pct, wilson_lo, wilson_hi, sign_p,
    #                              boot_mean_pct, boot_t, boot_lo, boot_hi, boot_p_le0)
    yb_gspc=dict(n=76, k=47, hit=61.8, wlo=50.6, whi=71.9, signp=0.0505,
                 mean=4.64, t=3.16, lo=1.78, hi=7.56, ple0=0.0006),
    yb_tr=dict(n=38, k=23, hit=60.5, wlo=44.7, whi=74.4, signp=0.2559,
               mean=3.26, t=1.61, lo=-0.69, hi=7.17, ple0=0.0542),
    yb_spy=dict(n=33, k=19, hit=57.6, wlo=40.8, whi=72.8, signp=0.4869,
                mean=3.00, t=1.41, lo=-1.08, hi=7.08, ple0=0.0765),
    # by-calendar-month, ^GSPC 1950-2026, mean %
    by_month=dict(Jan=0.96, Feb=-0.10, Mar=0.89, Apr=1.45, May=0.37, Jun=0.12,
                  Jul=1.22, Aug=-0.06, Sep=-0.73, Oct=0.74, Nov=1.76, Dec=1.35),
    # bad-autumns decomposition
    bad_full_gap=0.770, bad_full_t=2.78, bad_trim_gap=0.496, bad_trim_t=1.88,
    bad_n_dropped=10, bad_n_total=917,
    bad_dates=["1974-09", "1978-10", "1979-10", "1986-09", "1987-10", "2002-09",
               "2008-09", "2008-10", "2018-10", "2022-09"],
    # Halloween timer: tape -> {cost_bps: (cagr, bh_cagr, sharpe, bh_sharpe, maxdd, bh_maxdd)}
    timer_tr={5: (8.06, 10.23, 0.52, 0.54, -31.7, -53.2), 10: (7.85, 10.23, 0.50, 0.54, -31.9, -53.2)},
    timer_spy={5: (7.28, 9.54, 0.49, 0.52, -31.5, -53.0), 10: (7.07, 9.54, 0.47, 0.52, -31.7, -53.0)},
    timer_gspc={5: (7.62, 6.31, 0.35, 0.19, -33.2, -56.9), 10: (7.41, 6.31, 0.33, 0.19, -33.4, -56.9)},
    timer_years_tr=38.4, timer_years_spy=33.3, timer_years_gspc=66.5,
    reverse_cagr=4.64, reverse_bh_cagr=10.23, reverse_sharpe=0.20,
    # synthetic control
    syn_null_mean=-0.01, syn_null_sd=1.04, syn_null_fire=1, syn_planted_t=2.38,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Bad autumns?: Confirmed](https://img.shields.io/badge/Bad_autumns%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from sell_in_may import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    MM = data.load_monthly()
else:
    MM = None
print("real cache present:", HAVE_REAL, "| monthly series:",
      (None if MM is None else {k: len(v) for k, v in MM.items()}))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Should you really sell your stocks every May? 🍂\n"
            "### The Halloween indicator — half folklore, half true, and a cash-in-summer "
            "timer that loses you money either way\n\n"
            + BADGES +
            "*\"Sell in May and go away\"* is maybe the most-repeated line in market folklore: "
            "hold stocks November through April, sit in cash May through October, and you'll "
            "beat the market with less risk. It even has an academic paper behind it — Bouman "
            "& Jacobsen (2002) found the winter half beats the summer half in 36 of 37 countries "
            "they studied.\n\n"
            "So is it true? Yes — mostly. Should you trade it? No. This is the story of how both "
            "of those can be true at once.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** ^GSPC 1950→2026 (price-only, matching the deep-history "
            "literature), SPY 1993→2026 and ^SP500TR 1988→2026 (both dividend-inclusive), "
            "yfinance, cached. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does winter really beat summer? | **Yes, on average, everywhere we look.** Over "
            f"76 years of deep history the gap is **+{R['gspc']['gap']:.2f} points a month** "
            "(winter ~6.5% per half-year vs summer ~1.7%). |\n"
            "| Is that a *statistical fact* or *could be luck*? | **Depends which tape.** The "
            "76-year price-only tape clears our significance bar comfortably. The 33-38 year "
            "*dividend-inclusive* tapes — the ones that describe what you'd actually earn — "
            "don't quite get there. Direction agrees; certainty doesn't. |\n"
            f"| Is \"summer is dead money\" true? | **No — that's the catch.** Summer's average "
            f"return is **positive** on every tape we tested (+{R['spy']['summer']:.2f}%/mo on "
            "SPY, roughly +3.8% every six months). It's just smaller than winter's, not zero. |\n"
            "| So can you trade it? | **Not profitably.** A timer that holds cash all summer "
            "*loses* to buy-and-hold, in both raw return and risk-adjusted terms, because it's "
            "giving up a season that pays positive money. |\n\n"
            "> The seasonal is real-ish. The trade isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"November through April is where the stock market does its work. May through "
            "October you're better off in cash — the risk isn't worth the (near-zero) reward.\"*\n\n"
            "This isn't a fringe idea. Bouman & Jacobsen published it in the *American Economic "
            "Review* in 2002, and follow-up work has traced the pattern back to 1693 in UK data. "
            "It's one of the most studied, most repeated calendar effects in finance — which "
            "makes it a perfect test case: is the famous version of the story still true once you "
            "hold it to the desk's bar?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If \"sell in May\" is real *and* tradable, it's one of the simplest edges imaginable: "
            "two trades a year, no forecasting, no data mining — just a calendar. Every discount "
            "brokerage account could run it. That's exactly why it's worth checking carefully: an "
            "edge this famous and this cheap to implement should have been arbitraged away years "
            "ago if it were genuinely free money. So we ask three things: is the pattern real, how "
            "much of it is a few scary autumns doing all the work, and does the obvious trade "
            "actually pay you after you account for the fact that summer isn't actually negative?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The split.** Every calendar month since 1950 sorted into \"winter\" (Nov–Apr) or "
            "\"summer\" (May–Oct) — a fixed, public rule; no fitting.\n"
            "- **Three tapes, not one.** The deep ^GSPC history (price-only, since dividends "
            "aren't tracked that far back) *and* two modern dividend-inclusive tapes (SPY, "
            "^SP500TR) — if they disagree, that itself is the finding.\n"
            "- **The pairing check.** One (summer, winter) score per year for ~76 years, "
            "resampled at the *year* level so we never accidentally split a season in half.\n"
            "- **The autopsy.** How much of the gap survives if we remove the ten worst "
            "individual Septembers/Octobers?\n"
            "- **The trade check.** Buy stocks Nov 1, sell May 1 into cash (a real T-bill rate), "
            "buy back Nov 1 — versus just holding stocks the whole time, both compared on the "
            "*same* risk-adjusted basis."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average monthly return, winter vs summer, on three "
            "different tapes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tags = {'gspc': '^GSPC (price-only, 1950-)', 'sp500tr': '^SP500TR (total-return, 1988-)',\n"
            "            'spy': 'SPY (div-adj, 1993-)'}\n"
            "    stats = {k: st.headline_split(MM[k]) for k in tags if k in MM}\n"
            "    labels = list(tags[k] for k in stats)\n"
            "    winters = [stats[k]['winter_pct'] for k in stats]\n"
            "    summers = [stats[k]['summer_pct'] for k in stats]\n"
            "else:\n"
            "    labels = ['^GSPC (price-only, 1950-)', '^SP500TR (total-return, 1988-)', 'SPY (div-adj, 1993-)']\n"
            "    winters = [R['gspc']['winter'], R['sp500tr']['winter'], R['spy']['winter']]\n"
            "    summers = [R['gspc']['summer'], R['sp500tr']['summer'], R['spy']['summer']]\n"
            "x = np.arange(len(labels)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(x - w/2, winters, w, color=RED, label='winter (Nov-Apr)')\n"
            "ax.bar(x + w/2, summers, w, color=GREY, label='summer (May-Oct)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean monthly return (%)')\n"
            "ax.set_title('Winter beats summer everywhere - but summer is never negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(list(zip(labels, winters, summers)))"
        ),
        md(
            f"Winter beats summer on **every** tape — but look at the grey bars: summer's mean is "
            f"**positive** every time (+{R['spy']['summer']:.2f}%/mo on the modern SPY tape). "
            "\"Sell in May\" implicitly assumes summer is dead money. It isn't — it's just the "
            "*smaller* half.\n\n"
            "**Next — does the gap actually hold up statistically?** This is where the three "
            "tapes disagree:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = [stats[k]['welch_t'] for k in stats]\n"
            "else:\n"
            "    ts = [R['gspc']['welch'], R['sp500tr']['welch'], R['spy']['welch']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "cols = [GREEN if t >= 2 else AMBER for t in ts]\n"
            "ax.bar(labels, ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1.4, label='desk certification bar (t=2)')\n"
            "for i, v in enumerate(ts): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Welch t (winter vs summer)')\n"
            "ax.set_title('Only the 76-year price-only tape clears the bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Welch t:', dict(zip(labels, ts)))"
        ),
        md(
            f"The deep 76-year tape (**t = +{R['gspc']['welch']:.2f}**) clears our bar. The two "
            "modern tapes that actually include dividends — the ones that describe what you'd "
            f"really earn — sit at **t = +{R['sp500tr']['welch']:.2f}** and "
            f"**t = +{R['spy']['welch']:.2f}**: positive, same direction, but **not certified**. "
            "That's an honest \"maybe, mostly.\"\n\n"
            "**So what's driving even the strong result?** A few famous autumns:"
        ),
        code(
            "months = list(R['by_month'].keys())\n"
            "vals = list(R['by_month'].values())\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "cols = [RED if m in ('Sep', 'Oct') else (GREEN if v > 0 and m not in ('May','Jun','Jul','Aug','Sep','Oct') else GREY) for m, v in zip(months, vals)]\n"
            "ax.bar(months, vals, color=cols, width=.62)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean monthly return, 1950-2026 (%)')\n"
            "ax.set_title('September is the only negative month on the calendar')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('gap before trimming: {R['bad_full_gap']:+.2f}%/mo (t={R['bad_full_t']:+.2f})')\n"
            f"print('gap after dropping the 10 worst Sep/Oct months: {R['bad_trim_gap']:+.2f}%/mo (t={R['bad_trim_t']:+.2f})')"
        ),
        md(
            f"September is the *only* month with a negative 76-year average. Pull out just the "
            f"**{R['bad_n_dropped']} worst Septembers and Octobers** ({R['bad_n_dropped']}/"
            f"{R['bad_n_total']} months — about **1%** of the entire sample: 1987's crash, 2008's "
            "crash, a handful of quieter bad years) and the winter-summer gap shrinks by "
            f"**35%**, and the significance **disappears** (t = {R['bad_full_t']:.2f} → "
            f"{R['bad_trim_t']:.2f}). A meaningful chunk of \"sell in May\" is really \"avoid a "
            "short list of famous crashes,\" dressed up as a smooth six-month pattern.\n\n"
            "**Finally — does the actual trade pay you?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.halloween_timer(MM['sp500tr'], MM['irx_pct'], cost_bps=5.0)\n"
            "    cagr, bh_cagr = r['cagr_pct'], r['bh_cagr_pct']\n"
            "    sh, bh_sh = r['sharpe_excess'], r['bh_sharpe_excess']\n"
            "else:\n"
            "    cagr, bh_cagr = R['timer_tr'][5][0], R['timer_tr'][5][1]\n"
            "    sh, bh_sh = R['timer_tr'][5][2], R['timer_tr'][5][3]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['timer\\n(Nov-Apr only)', 'buy & hold'], [cagr, bh_cagr], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([cagr, bh_cagr]): a1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('CAGR (%)'); a1.set_title('Total return (^SP500TR, 5bps cost)')\n"
            "a2.bar(['timer', 'buy & hold'], [sh, bh_sh], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([sh, bh_sh]): a2.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('Sharpe, excess of cash'); a2.set_title('Risk-adjusted - still a loss')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer CAGR {cagr:+.2f}% vs buy&hold {bh_cagr:+.2f}%  |  Sharpe {sh:.2f} vs {bh_sh:.2f}')"
        ),
        md(
            "The timer **loses** — on plain return *and* on a risk-adjusted basis. Why? Because "
            "summer isn't a dangerous, break-even season to be avoided — it's a **smaller but "
            "positive** one. Sitting in cash for six months a year gives up real money, and the "
            "lower volatility you get in exchange isn't enough to make up for it once you compare "
            "like-for-like (excess of the same cash rate on both sides).\n\n"
            "> ⚠️ One trap we found and avoided: run the same timer on the **price-only** ^GSPC "
            "tape (no dividends) and it looks like a *winner*. That's an illusion — buy-and-hold "
            "there is missing dividends for all twelve months, while the timer only misses six "
            "(it swaps the other six for a real cash yield). Put dividends back in (SPY, "
            "^SP500TR) and the advantage reverses. Always compare like-for-like."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Real over 76 years of deep, price-only history "
            f"(*t* = +{R['gspc']['welch']:.2f}) but **not certified** on the modern "
            "dividend-inclusive tape that actually matters for a portfolio "
            f"(*t* = +{R['spy']['welch']:.2f}). Direction agrees everywhere; certainty doesn't.\n"
            "- **Tradability — Mirage.** A cash-in-summer timer loses to buy-and-hold on both "
            "raw return and risk-adjusted return, because summer pays positive money.\n"
            "- **\"Driven by a handful of bad autumns\"? — Confirmed.** About 1% of the months "
            "(ten crash-adjacent Septembers/Octobers) explain over a third of the gap and flip "
            "the deep-history significance off entirely when removed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson: a durable pattern isn't automatically a trade.** \"Sell in "
            "May\" is one of the most replicated calendar effects in finance, and it *still* "
            "loses to buy-and-hold once you charge it against a positive alternative. Statistical "
            "significance and economic value are two different questions.\n"
            "- **Where this could get interesting**: testing the seasonal on markets/decades with "
            "genuinely negative or flat summers (some emerging markets, some pre-1950 eras) where "
            "the trade-off might tip the other way.\n"
            "- **Sibling study:** [55-summer-lull](../../55-summer-lull/) reaches a compatible "
            "verdict on a single blended tape; this study shows *why* the tapes disagree and "
            "*how much* of the effect is a few famous autumns.\n\n"
            "*Think the timer can be saved? Show a version that beats buy-and-hold on "
            "**excess-of-cash Sharpe**, after real costs, on the dividend-inclusive tape — "
            "then we'll talk.*"
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
            "# Sell in May — a quantitative teardown 🔬\n"
            "### Winter/summer Welch-HAC splits on three tapes · year-block bootstrap & sign "
            "test · the crash-autumn decomposition · an honest excess-of-cash Halloween-timer "
            "backtest · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "\"Sell in May\" (Bouman & Jacobsen 2002) is one of the best-documented calendar "
            "anomalies in the literature. The job here is to measure it honestly on the tape we "
            "actually have, decompose what's driving it, and then ask the only question that "
            "pays: *is any of it tradable, after dividends and costs?*\n\n"
            "> ⚠️ **Data note.** ^GSPC daily Close (price-only, 1950→2026, matching the brief's "
            "deep-history instruction), SPY daily Close (dividend-adjusted, 1993→2026), "
            "^SP500TR daily Close (genuine total return, 1988→2026), ^IRX daily Close (13-week "
            "T-bill discount yield, cash proxy, 1960→2026), yfinance, cached. No survivorship "
            "(indices/index-trackers). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_gspc"] + "` / `" +
            R["fp_spy"] + "` / `" + R["fp_tr"] + "` / `" + R["fp_irx"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | ^GSPC (price-only, 1950-): Welch **t = +{R['gspc']['welch']:.2f}**, "
            f"NW(3) **t = +{R['gspc']['nw']:.2f}**, year-block bootstrap **t = +{R['yb_gspc']['t']:.2f}**. "
            f"^SP500TR (1988-): **t = +{R['sp500tr']['welch']:.2f}**; SPY (1993-): "
            f"**t = +{R['spy']['welch']:.2f}** — neither dividend-inclusive tape clears 2 |\n"
            f"| **Tradability** | `MIRAGE` | Halloween timer CAGR {R['timer_tr'][5][0]:+.2f}% vs "
            f"buy&hold {R['timer_tr'][5][1]:+.2f}% (^SP500TR, 5bps); excess-Sharpe "
            f"{R['timer_tr'][5][2]:.2f} vs {R['timer_tr'][5][3]:.2f} — loses both ways |\n"
            f"| **Bad autumns?** | `CONFIRMED` | dropping {R['bad_n_dropped']}/{R['bad_n_total']} "
            f"months (1.1%) cuts the gap 35% and the Welch t from "
            f"+{R['bad_full_t']:.2f} to +{R['bad_trim_t']:.2f} |\n\n"
            "> 💡 In plain words: the seasonal is real on paper over the long run, largely a "
            "crash-autumn story under the hood, uncertified on the modern tradable tape, and "
            "not worth trading even where it's real."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the monthly log return of the tape and $W_t \\in \\{0,1\\}$ the "
            "winter-month flag (Nov–Apr = 1, May–Oct = 0), a fixed calendar partition known "
            "years in advance — no fitting, no look-ahead in either leg. The claims:\n\n"
            "- **H₁ (the gap).** $E[r_t \\mid W_t=1] \\gg E[r_t \\mid W_t=0]$ — large, systematic, "
            "not a sampling coincidence.\n"
            "- **H₂ (dead summer).** $E[r_t \\mid W_t=0] \\approx 0$ — the implicit premise "
            "behind \"go away.\"\n"
            "- **H₃ (broad-based).** The gap is a smooth six-month tailwind, not a concentrated "
            "tail event.\n"
            "- **H₄ (bankable).** A timer that holds cash May–Oct beats buy-and-hold, risk-"
            "adjusted, after costs.\n\n"
            "We find **H₁ supported on deep history, not certified on the dividend-inclusive "
            "tape**, **H₂ rejected** (summer is positive on every tape), **H₃ rejected** (a "
            "1.1%-of-sample trim kills the deep-tape significance), **H₄ rejected** (the timer "
            "loses on CAGR *and* excess-Sharpe)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Winter and summer are **non-overlapping six-month blocks**, so the planned primary "
            "is a **Welch *t*** on the monthly group split, cross-checked by a **Newey-West "
            "(3-lag) *t*** on the winter-dummy regression (monthly data has far less serial "
            "correlation than daily, but a HAC check costs nothing and catches anything the "
            "raw split misses). Because a group split can still be inflated by within-year "
            "correlation between adjacent months, we add a **year-block bootstrap** (10,000 "
            "draws, resampling whole Halloween *years*, never individual months) and an exact "
            "**sign test** with a Wilson interval on the paired per-year gap. Three tapes are "
            "run **separately, never pooled** — price-only ^GSPC (deep history, matching the "
            "brief), dividend-adjusted SPY and genuine total-return ^SP500TR — because pooling "
            "them would hide exactly the disagreement that turns out to be the finding."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Calendar.** Nov→Apr = winter, May→Oct = summer — fixed, public, no fitting.\n"
            "- **Tapes.** ^GSPC (price-only, 1950→2026), SPY (div-adj, 1993→2026), ^SP500TR "
            "(total-return, 1988→2026); ^IRX (13-week T-bill yield) for the cash leg. As-of "
            f"{R['as_of']} (last complete month).\n"
            "- **Headline.** Welch *t* + NW(3) *t* on each tape separately.\n"
            "- **Robustness.** Year-block bootstrap (10,000 draws) + exact sign test, paired "
            "per Halloween year.\n"
            "- **Decomposition.** By-calendar-month means; gap recomputed after dropping the "
            "worst 5 Septembers + 5 Octobers (a diagnostic, reported once, not re-run as the "
            "headline).\n"
            "- **Execution (third axis).** The calendar rule is public years ahead — hold the "
            "full named month, zero look-ahead. Timer: 2 × one-way cost × NAV per switch, 2 "
            "switches/yr; Sharpe **excess of cash on both legs**.\n"
            "- **Control.** Synthetic seeded monthly world, planted winter-premium knob; the "
            "null must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split — three tapes, one honest disagreement\n\n"
            "Welch *t* and Newey-West(3) *t* on the winter-dummy regression, tape by tape."
        ),
        code(
            "rows = []\n"
            "for key, label in (('gspc', '^GSPC price-only 1950-'), ('sp500tr', '^SP500TR 1988-'), ('spy', 'SPY 1993-')):\n"
            "    if HAVE_REAL and key in MM:\n"
            "        s = st.headline_split(MM[key])\n"
            "        rows.append((label, s['winter_pct'], s['summer_pct'], s['gap_pct'], s['welch_t'], s['nw_t']))\n"
            "    else:\n"
            "        r = R[key]\n"
            "        rows.append((label, r['winter'], r['summer'], r['gap'], r['welch'], r['nw']))\n"
            "for label, w_, s_, g_, t_, nw_ in rows:\n"
            "    print(f'{label:26s} winter {w_:+.3f}%  summer {s_:+.3f}%  gap {g_:+.3f}%  '\n"
            "          f'Welch t={t_:+.2f}  NW t={nw_:+.2f}')\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "labels = [r[0] for r in rows]; ts = [r[4] for r in rows]\n"
            "cols = [GREEN if t >= 2 else AMBER for t in ts]\n"
            "ax.bar(labels, ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1.4, label='t = 2 certification bar')\n"
            "for i, v in enumerate(ts): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Welch t'); ax.set_title('Deep price-only history clears the bar; dividend-inclusive tapes do not')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the effect is real and large over 76 years of price-only "
            f"history (t = +{R['gspc']['welch']:.2f} / NW +{R['gspc']['nw']:.2f}) but the two "
            f"tapes that actually price dividends — the economically correct comparison — sit "
            f"at t = +{R['sp500tr']['welch']:.2f} and +{R['spy']['welch']:.2f}. Per house "
            "precedent ([89-turn-of-the-month](../../89-turn-of-the-month/)), a deep tape "
            "clearing the bar while the modern sample can't certify it reads **WEAK**, not REAL."
        ),
        md(
            "### 4b · Year-block bootstrap and sign test — the paired-year robustness check\n\n"
            "One (summer, winter) score per Halloween year; resampling **whole years** (10,000 "
            "draws) instead of individual months respects the within-year correlation and never "
            "lets a draw straddle a season boundary."
        ),
        code(
            "rows = []\n"
            "for key, ykey, label in (('gspc', 'yb_gspc', '^GSPC'), ('sp500tr', 'yb_tr', '^SP500TR'), ('spy', 'yb_spy', 'SPY')):\n"
            "    if HAVE_REAL and key in MM:\n"
            "        pairs = st.halloween_year_pairs(MM[key])\n"
            "        sgn = st.sign_test_stats(pairs); bs = st.year_block_bootstrap(pairs)\n"
            "        rows.append((label, sgn['n'], sgn['k_winter_wins'], sgn['hit_rate']*100,\n"
            "                     sgn['p_value'], bs['mean_gap_pct'], bs['t_analytic'],\n"
            "                     bs['boot_lo_pct'], bs['boot_hi_pct'], bs['boot_p_le0']))\n"
            "    else:\n"
            "        y = R[ykey]\n"
            "        rows.append((label, y['n'], y['k'], y['hit'], y['signp'], y['mean'], y['t'], y['lo'], y['hi'], y['ple0']))\n"
            "for label, n, k, hit, p, mean, t, lo, hi, ple0 in rows:\n"
            "    print(f'{label:10s} n={n:3d}  winter wins {k}/{n} = {hit:.1f}%  sign-p={p:.4f}  '\n"
            "          f'boot mean {mean:+.2f}%  t={t:+.2f}  CI[{lo:+.2f}%, {hi:+.2f}%]  P(<=0)={ple0:.4f}')\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "labels = [r[0] for r in rows]; means = [r[5] for r in rows]\n"
            "los = [r[5]-r[7] for r in rows]; his = [r[8]-r[5] for r in rows]\n"
            "ax.bar(labels, means, yerr=[los, his], color=AMBER, capsize=6, width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('bootstrap mean paired gap (%, 95% CI)')\n"
            "ax.set_title(\"Only the deep tape's CI clears zero\")\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: on the deep tape, winter beat summer in "
            f"{R['yb_gspc']['k']}/{R['yb_gspc']['n']} years ({R['yb_gspc']['hit']:.1f}%) and the "
            f"bootstrap CI [{R['yb_gspc']['lo']:+.2f}%, {R['yb_gspc']['hi']:+.2f}%] sits entirely "
            "above zero — a genuinely robust result on that sample. On the two shorter, "
            "dividend-inclusive tapes the sign test (*p* = "
            f"{R['yb_tr']['signp']:.2f} / {R['yb_spy']['signp']:.2f}) and the bootstrap CIs "
            "(both straddling zero) tell the same uncertified story as 4a — two independent "
            "methods, same conclusion."
        ),
        md(
            "### 4c · The crash-autumn decomposition — how broad-based is this, really?\n\n"
            "Mean return by calendar month, then the gap recomputed after removing the five "
            "worst individual Septembers and five worst Octobers — a diagnostic reported once, "
            "never re-run as the headline number."
        ),
        code(
            "months = list(R['by_month'].keys()); vals = list(R['by_month'].values())\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = [RED if v < 0 else (GREEN if m in ('Nov','Dec','Jan','Feb','Mar','Apr') else AMBER) for m, v in zip(months, vals)]\n"
            "ax.bar(months, vals, color=cols, width=.62)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean monthly return, ^GSPC 1950-2026 (%)')\n"
            "ax.set_title('September: the one reliably negative month')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gap: full {R['bad_full_gap']:+.3f}%/mo (t={R['bad_full_t']:+.2f})  ->  \"\n"
            "      f\"after dropping {R['bad_n_dropped']}/{R['bad_n_total']} crash months: \"\n"
            "      f\"{R['bad_trim_gap']:+.3f}%/mo (t={R['bad_trim_t']:+.2f})\")\n"
            "print('dropped:', ', '.join(R['bad_dates']))"
        ),
        md(
            f"> 💡 In plain words: removing just **{R['bad_n_dropped']} months out of "
            f"{R['bad_n_total']}** (1.1% of the sample — the worst individual Septembers and "
            f"Octobers, including 1987 and 2008) shrinks the gap by 35% and flips the deep-tape "
            f"Welch *t* from +{R['bad_full_t']:.2f} to +{R['bad_trim_t']:.2f} — **below the "
            "certification bar**. H₃ (broad-based) is rejected: a meaningful share of the "
            "\"sell in May\" story is really \"a short list of famous autumn crashes,\" not a "
            "smooth six-month tailwind."
        ),
        md(
            "### 4d · The third axis — the honest Halloween-timer backtest\n\n"
            "Long the winter leg, cash (^IRX) the summer leg, vs buy-and-hold; 2 × one-way cost "
            "× NAV per switch, 2 switches/yr; Sharpe **excess of cash on both legs** (a "
            "part-time-in-cash strategy must not race a raw buy-and-hold Sharpe)."
        ),
        code(
            "rows = []\n"
            "for key, tkey, label in (('sp500tr', 'timer_tr', '^SP500TR'), ('spy', 'timer_spy', 'SPY')):\n"
            "    for cb in (5.0, 10.0):\n"
            "        if HAVE_REAL and key in MM:\n"
            "            r = st.halloween_timer(MM[key], MM['irx_pct'], cost_bps=cb)\n"
            "            rows.append((label, cb, r['cagr_pct'], r['bh_cagr_pct'], r['sharpe_excess'], r['bh_sharpe_excess']))\n"
            "        else:\n"
            "            c, bc, s_, bs_, dd, bdd = R[tkey][int(cb)]\n"
            "            rows.append((label, cb, c, bc, s_, bs_))\n"
            "for label, cb, c, bc, s_, bs_ in rows:\n"
            "    print(f'{label:8s} @ {cb:>4.1f}bps: timer CAGR {c:+.2f}% vs b&h {bc:+.2f}%   '\n"
            "          f'excess-Sharpe {s_:.2f} vs {bs_:.2f}')\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "sub = [r for r in rows if r[1] == 5.0]\n"
            "labels = [r[0] for r in sub]\n"
            "a1.bar(np.arange(len(sub)) - .18, [r[2] for r in sub], .32, color=AMBER, label='timer')\n"
            "a1.bar(np.arange(len(sub)) + .18, [r[3] for r in sub], .32, color=GREY, label='buy & hold')\n"
            "a1.set_xticks(range(len(sub))); a1.set_xticklabels(labels); a1.set_ylabel('CAGR (%)')\n"
            "a1.set_title('Return: timer loses'); a1.legend()\n"
            "a2.bar(np.arange(len(sub)) - .18, [r[4] for r in sub], .32, color=AMBER, label='timer')\n"
            "a2.bar(np.arange(len(sub)) + .18, [r[5] for r in sub], .32, color=GREY, label='buy & hold')\n"
            "a2.set_xticks(range(len(sub))); a2.set_xticklabels(labels); a2.set_ylabel('Sharpe, excess of cash')\n"
            "a2.set_title('Risk-adjusted: timer loses too')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: on both dividend-inclusive tapes the timer trails buy-and-hold "
            f"on CAGR by **~2.2–2.3 points a year** and loses on excess-of-cash Sharpe too "
            f"(+{R['timer_tr'][5][2]:.2f} vs +{R['timer_tr'][5][3]:.2f} on ^SP500TR). H₄ "
            "(bankable) is rejected. **Named artefact:** run the same timer on price-only ^GSPC "
            f"(1960-2026, {R['timer_years_gspc']:.0f} years, cash from ^IRX) and it *looks* like "
            f"a winner (CAGR {R['timer_gspc'][5][0]:+.2f}% vs {R['timer_gspc'][5][1]:+.2f}%, "
            f"Sharpe {R['timer_gspc'][5][2]:.2f} vs {R['timer_gspc'][5][3]:.2f}) — but that's "
            "because the no-dividend buy-and-hold benchmark is missing *twelve* months of "
            "dividends while the timer only misses *six* (the other six are a real cash yield). "
            "Put dividends back in and the advantage reverses; the reverse straw man (long "
            f"summer, cash winter) does even worse ({R['reverse_cagr']:+.2f}% CAGR, "
            f"{R['reverse_sharpe']:.2f} excess-Sharpe vs buy-and-hold's "
            f"{R['reverse_bh_cagr']:+.2f}% / {R['timer_tr'][5][3]:.2f}), confirming the calendar "
            "asymmetry is genuine — it just isn't worth harvesting."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic seeded monthly log-return series, TUNABLE planted winter premium. The "
            "null (premium = 0) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    ret = data.synthetic_world(premium_bp=0.0, seed=641 + s_)\n"
            "    null_ts.append(st.synthetic_detect(ret)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "ret = data.synthetic_world(premium_bp=40.0, seed=641)\n"
            "planted_t = st.synthetic_detect(ret)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (premium=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted premium = +40bps/mo')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (winter vs summer)')\n"
            "ax.set_title('Control: the null rarely fires; a planted premium lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires (\\|t\\| ≥ 2) in "
            f"{R['syn_null_fire']}/20 seeds — about the nominal 5% false-positive rate at a 2σ "
            f"cutoff, i.e. correctly calibrated, not biased toward finding a seasonal that isn't "
            f"there. A planted +40bps/mo winter premium reads t = +{R['syn_planted_t']:.2f}. "
            "The machinery is unbiased; the real-tape numbers above are the genuine article. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — deep price-only ^GSPC (1950-2026): Welch "
            f"**t = +{R['gspc']['welch']:.2f}**, NW(3) **t = +{R['gspc']['nw']:.2f}**, "
            f"year-block bootstrap **t = +{R['yb_gspc']['t']:.2f}** (95% CI clear of zero). "
            f"Dividend-inclusive ^SP500TR/SPY: **t = +{R['sp500tr']['welch']:.2f}** / "
            f"**+{R['spy']['welch']:.2f}** — neither certified, sign tests *p* = "
            f"{R['yb_tr']['signp']:.2f} / {R['yb_spy']['signp']:.2f}. A 1.1%-of-sample trim of "
            f"crash autumns drops the deep-tape *t* below 2. Real over long history, uncertified "
            "on the modern tradable tape.\n"
            f"- **Tradability `MIRAGE`** — the Halloween timer trails buy-and-hold on CAGR "
            f"(-2.2 to -2.3 pts/yr) *and* excess-of-cash Sharpe on every dividend-inclusive "
            "tape; a price-only race flatters the timer via a named dividend-accounting "
            "artefact, not a real edge.\n"
            "- **Bad autumns? `CONFIRMED`** — 10/917 months (1.1%) explain 35% of the gap and "
            "flip the deep-history certification off when removed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson.** Even one of the most-replicated calendar anomalies in "
            "finance — decades of literature, dozens of countries — fails to certify on a modern "
            "sample and loses money once traded against a positive alternative. Statistical "
            "durability and economic value are separate questions, and this study answers both "
            "honestly rather than stopping at the first \"yes.\"\n"
            "- **The natural sequel** is testing whether the *magnitude* of the winter premium "
            "co-moves with a macro state variable (rate cycle, recession risk) rather than being "
            "a pure calendar constant — Bouman & Jacobsen themselves flagged this as unresolved.\n"
            "- **Dedup map:** [55-summer-lull](../../55-summer-lull/) (the same claim, a single "
            "blended tape, a compatible WEAK/MIRAGE verdict this study deepens), "
            "[89-turn-of-the-month](../../89-turn-of-the-month/) (the methodological precedent "
            "for the deep-history-vs-modern-sample split), "
            "[290-september-effect](../../290-september-effect/) (the single-month question "
            "this study's by-calendar-month table complements), "
            "[136-mark-twain](../../136-mark-twain/) (the October-specific myth this study's "
            "crash-autumn decomposition independently touches).\n\n"
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
