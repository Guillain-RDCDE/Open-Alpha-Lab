"""Generate the two narrative notebooks for Study 615 (Yen Safe Haven).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tape
under ../_cache/ (built once from yfinance) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance JPY=X + SPY +
# ^IRX + ^TNX, 1996-10-30 -> 2026-06-30, 7,418 joint daily returns, 356 complete months).
R = dict(
    start="1996-10-30", end="2026-06-30", years=29.4, n_days=7418, n_months=356,
    fingerprint="6d34d723338d", as_of="2026-06-30",
    # piecewise daily regression, HAC/NW t (10 lags)
    beta_dn=-0.113, t_dn=-4.62, beta_up=-0.046, t_up=-1.94,
    beta_all=-0.080, t_all=-4.45, r2=0.020, nw_lags=10,
    # quintile ladder (Q1 = worst SPY days), bps/day
    q_means=[12.17, -1.35, -3.06, -4.20, -6.64], spy_q1_bps=-157.1,
    t_q1_q5=5.81, t_q1_mid=5.81,
    # monthly split
    dn_bps=-8.51, up_bps=-17.16, n_dn=128, n_up=228, t_month=0.26,
    dec_n=35, dec_spy=-7.93, dec_jpy=0.39, dec_hit=57,
    # robustness of the full-sample daily signal (skeptic pass)
    drop_all_ev_t=-4.27, drop_all_ev_welch=3.97, drop_all_ev_n=5924,
    drop100_t=-3.02, drop100_welch=4.68, boot_welch_mean=5.72, boot_frac_gt2=100,
    # non-stationarity: the signal is carried by the first half and dead in the last decade
    #   (era, n, beta_dn, HAC t, Q1 bps/day, Welch t Q1 vs mid)
    sub=[("1996-2010", 3529, -0.179, -6.18, 24.04, 5.57),
         ("2011-2026", 3889, -0.022, -0.89, -0.83, 1.03),
         ("2016-2026", 2633, -0.011, -0.38, -0.92, 0.16)],
    # event studies: (label, start, end, spy%, jpy%, hedged)
    events=[
        ("GFC deleveraging (Lehman -> year-end)", "2008-09-12", "2008-12-31", -27.43, 19.07, True),
        ("Aug-2015 (CNY deval + flash crash)", "2015-08-10", "2015-08-25", -11.07, 4.56, True),
        ("COVID risk-off leg (peak -> 03-09)", "2020-02-19", "2020-03-09", -18.95, 5.75, True),
        ("COVID dollar scramble (03-09 -> trough)", "2020-03-09", "2020-03-23", -18.22, -5.91, False),
        ("2022 bear market (top -> trough)", "2022-01-03", "2022-10-12", -24.50, -21.07, False),
        ("Aug-2024 carry unwind (BOJ hike -> crash)", "2024-07-31", "2024-08-05", -6.07, 4.87, True),
    ],
    # regime split A: carry-stock proxy (bill >= 2% vs ZIRP)
    hi_beta_dn=-0.097, hi_t_dn=-3.83, hi_n=3310, hi_share=45,
    lo_beta_dn=-0.123, lo_t_dn=-3.59, lo_n=4108,
    # regime split B: SPY down-months by 10y yield direction
    fell_bps=112.53, rose_bps=-150.06, n_fell=69, n_rose=59, t_gap=5.14,
    # tradability: the carry bill
    spot_ann=-2.20, excess_ann=-4.34, net_ann=-4.74, avg_bill=2.21, expense_bps=40,
    # 2022 focus
    spy_2022=-24.50, jpy_2022=-21.07, ratio_2022=86,
    # synthetic control: (planted hedge, beta_dn, HAC t, Q1 bps, welch t Q1 vs Q5)
    syn=[(0.00, 0.015, 1.32, -3.22, -1.59), (0.30, -0.284, -25.91, 45.74, 20.75)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![2022 failure?: Confirmed](https://img.shields.io/badge/2022_failure%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from yen_safe_haven import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    TAPE = data.load_real()                  # as-of'd daily tape (usdjpy, spy, irx, tnx)
    RETS = st.to_returns(TAPE)
    M = st.monthly_returns(TAPE)
else:
    TAPE = RETS = M = None
print("real tape cached:", HAVE_REAL,
      "| daily returns:", (0 if RETS is None else len(RETS)),
      "| months:", (0 if M is None else len(M)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the yen really rally when stocks crash? 🇯🇵\n"
            "### The \"free crash hedge\" legend, tested on thirty years of tape — in plain English\n\n"
            + BADGES +
            "Every trading floor knows the story. For decades, Japan had the lowest interest rates on "
            "Earth, so the world **borrowed yen** for almost nothing and bought anything that paid more — "
            "the famous **carry trade**. When markets panic, those trades get unwound in a hurry: everyone "
            "has to **buy back the yen they borrowed**, all at once. Result, says the legend: *when stocks "
            "crash, the yen rallies* — a crash hedge you get **for free**.\n\n"
            "It has a highlight reel: 2008, the August-2015 flash crash, the COVID panic, the August-2024 "
            "carry unwind. It also has one very loud counter-example: **2022**, when US stocks fell 25% "
            "and the yen... fell 21% at the same time.\n\n"
            "So which is it — free insurance, or a story that quietly bills you? "
            "*(Our sibling study [69-safe-haven](../../69-safe-haven/README.md) asks the same question of "
            "**gold** — different asset, different mechanism, and, it turns out, a different way of "
            "disappointing you.)*\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the regime splits and the carry math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it from the cached tape (yfinance `JPY=X` + "
            "`SPY`, 1996–2026). House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the yen jump on scary days? | **It used to.** Over the full 30 years, on the worst "
            "20% of S&P days it gained about **+12 bps a day** while everything else burned — a genuine "
            "signal, not luck (it survives dropping every crash and a bootstrap). But it has **faded**: "
            "almost all of it is 1996–2010, and since 2011 the daily jump is essentially **gone**. |\n"
            "| Does it hedge whole bear markets? | **No.** Over full *months*, down-months and up-months "
            "look the same for the yen. In the 35 worst S&P months it rose only **57%** of the time. |\n"
            "| Is the insurance free? | **Absolutely not.** Holding yen cost about **4.7% a year, for "
            "thirty years** — you pay the carry the legend's heroes were *earning*. |\n"
            "| And 2022? | The hedge **failed exactly when needed** — and for a knowable reason: that "
            "crash was caused by *rising rates*, which makes the yen weaker, not stronger. |"
        ),

        md(
            "## One picture of the whole legend\n\n"
            "Yen strength (the inverse of the USD/JPY rate — up = strong yen) against the S&P 500, with "
            "the famous panic episodes shaded. You can *see* the legend: yen spikes at 2008, 2015, "
            "early-2020, Aug-2024... and the huge 2022–2024 slide where it fell *with* stocks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(10.5, 5.2))\n"
            "    yen = 100.0 / TAPE['usdjpy'].dropna()          # yen strength, up = strong\n"
            "    yen = 100 * yen / yen.iloc[0]\n"
            "    spy = TAPE['spy'].dropna()\n"
            "    spy = 100 * spy / spy.iloc[0]\n"
            "    ax.plot(yen.index, yen.values, color=RED, lw=1.2, label='Yen strength vs USD (indexed)')\n"
            "    ax2 = ax.twinx()\n"
            "    ax2.plot(spy.index, spy.values, color=GREY, lw=1.0, alpha=.8, label='SPY (total return, indexed, log)')\n"
            "    ax2.set_yscale('log'); ax2.grid(False)\n"
            "    for lab, s, e in data.EVENTS:\n"
            "        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=AMBER, alpha=.35)\n"
            "    ax.set_title('The yen vs the S&P 500, 1996-2026 (shaded: the famous panics)')\n"
            "    ax.set_ylabel('yen strength (100 = 1996)'); ax2.set_ylabel('SPY (log)')\n"
            "    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()\n"
            "    ax.legend(h1 + h2, l1 + l2, loc='upper left')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('cache missing - see docs/results.md for the frozen numbers')"
        ),

        md(
            "## Scary days: the legend is true\n\n"
            "Sort every trading day since 1996 into five buckets by how the S&P did. On the **worst** "
            "bucket (average S&P day ≈ −1.6%), the yen *gained*. On every other bucket it drifted down. "
            "That staircase is exactly what a crash hedge should look like — and the quants' notebook "
            "confirms it's far too clean to be luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    Q = st.quintile_ladder(RETS)\n"
            "    means = Q['means_bps']\n"
            "else:\n"
            "    means = R['q_means']\n"
            "fig, ax = plt.subplots()\n"
            "labels = ['Q1\\nworst SPY days', 'Q2', 'Q3', 'Q4', 'Q5\\nbest SPY days']\n"
            "colors = [GREEN if v > 0 else RED for v in means]\n"
            "ax.bar(labels, means, color=colors)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_ylabel('yen return (bps/day)')\n"
            "ax.set_title('The yen by S&P-day bucket - it gains only on the bad days')\n"
            "for i, v in enumerate(means):\n"
            "    ax.text(i, v + (0.6 if v > 0 else -1.2), f'{v:+.1f}', ha='center')\n"
            "plt.show()\n"
            "print(f\"Q1 mean {means[0]:+.2f} bps/day on days when SPY averaged {R['spy_q1_bps']:+.0f} bps\")"
        ),

        md(
            "## The highlight reel — and the two nights the hero didn't show\n\n"
            "Six famous panics, side by side. Four times the yen delivered handsomely. Twice it failed — "
            "the March-2020 **dollar scramble** (when the whole world needed dollars *right now* and sold "
            "everything else, even yen), and the **2022 bear market**, the longest, slowest crash of the "
            "sample... where the \"hedge\" lost almost as much as the stocks it was supposed to protect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    EV = st.event_table(TAPE)\n"
            "    rows = [(e['label'], e['spy_pct'], e['jpy_pct']) for e in EV]\n"
            "else:\n"
            "    rows = [(lab, s_, j_) for lab, _a, _b, s_, j_, _h in R['events']]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.2))\n"
            "x = np.arange(len(rows)); w = 0.38\n"
            "ax.bar(x - w/2, [r[1] for r in rows], w, color=GREY, label='SPY')\n"
            "ax.bar(x + w/2, [r[2] for r in rows], w,\n"
            "       color=[GREEN if r[2] > 0 else RED for r in rows], label='Yen (vs USD)')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels(['2008\\nGFC', '2015\\nCNY deval', 'COVID\\nrisk-off', 'COVID\\n$ scramble',\n"
            "                    '2022\\nbear', '2024\\ncarry unwind'])\n"
            "ax.set_ylabel('cumulative return over the window (%)')\n"
            "ax.set_title('Six panics: four hedged, two failed')\n"
            "ax.legend()\n"
            "plt.show()"
        ),

        md(
            "## The bill nobody mentions\n\n"
            "To own this hedge you hold yen instead of dollars. Dollars earned interest (about "
            f"**{R['avg_bill']:.1f}%/yr** on average since 1996); yen earned roughly zero — *that gap is "
            "the same carry the legend's traders were harvesting, and as the hedge-owner you're on the "
            "**paying** side of it.* Add the yen's own slow slide and an ETF fee, and \"free insurance\" "
            f"cost about **{abs(R['net_ann']):.1f}% a year, every year, for {R['years']:.0f} years**. "
            "Here is what a dollar kept in the yen sleeve (after the interest you gave up) looked like:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bill = RETS['irx'].values / 100.0 / 252.0\n"
            "    ex = RETS['jpy'].values - bill\n"
            "    wealth = pd.Series(np.exp(np.cumsum(np.log1p(ex))), index=RETS.index)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(wealth.index, wealth.values, color=RED, lw=1.4)\n"
            "    ax.axhline(1.0, color='k', lw=.8, ls='--')\n"
            "    ax.set_title('One dollar held in the yen \"hedge\" sleeve, net of the T-bill given up')\n"
            "    ax.set_ylabel('wealth (start = 1.00)')\n"
            "    ax.annotate(f\"{R['excess_ann']:+.2f}%/yr excess, {R['years']:.1f} years\",\n"
            "                xy=(0.55, 0.85), xycoords='axes fraction', color=RED)\n"
            "    plt.show()\n"
            "    print(f'final wealth of $1: ${wealth.iloc[-1]:.2f}')\n"
            "else:\n"
            "    print(f\"excess {R['excess_ann']:+.2f}%/yr, net {R['net_ann']:+.2f}%/yr - see docs/results.md\")"
        ),

        md(
            "## Why 2022 wasn't bad luck\n\n"
            "Split all the S&P's *down months* by what interest rates did at the same time:\n\n"
            f"- Stocks down because of a **growth scare** (rates falling): the yen made **about "
            f"+{R['fell_bps']/100:.1f}% a month**. That's 2008, 2020's first leg, 2024.\n"
            f"- Stocks down because of a **rate shock** (rates rising): the yen **lost about "
            f"{R['rose_bps']/100:.1f}% a month**. That's 2022, in one line.\n\n"
            "When a crash is caused by *rising US rates*, borrowing yen to hold dollars gets **more** "
            "attractive every week — so the carry trade grows *during* the selloff instead of unwinding. "
            "The yen doesn't hedge crashes. It hedges **fear about growth**. 2022 was fear about rates.\n\n"
            "## The verdict\n\n"
            "| Question | Verdict |\n|---|---|\n"
            "| Yen jumps on crash days? | **REAL** — the cleanest daily safe-haven signature on this desk |\n"
            "| An ownable crash hedge? | **MIRAGE** — ~4.7%/yr of negative carry, no monthly hedge value, "
            "and it skips rate-driven bears |\n"
            "| \"2022: failed exactly when needed\"? | **CONFIRMED** — yen −21% while SPY −25%, and the "
            "mechanism says that was the *expected* outcome |\n\n"
            "> 🔬 **For the quants:** the HAC *t*-stats, both regime splits, the carry accounting and the "
            "planted-hedge synthetic control live in [02_for_the_quants.ipynb](02_for_the_quants.ipynb).\n\n"
            "*Fingerprinted headline run: [docs/results.md](../docs/results.md) (as-of "
            f"{R['as_of']}, fingerprint `{R['fingerprint']}`). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Yen Safe Haven — the quant teardown 🇯🇵\n\n"
            + BADGES +
            "**Claim.** Risk-off forces yen-funded carry books to buy back JPY: *when stocks crash, the "
            "yen rallies* — a free crash hedge. **Tape.** Daily yfinance `JPY=X` (USDJPY spot; the yen's "
            "return is **minus** the USDJPY change; price-only, labeled) + `SPY` (total-return) + `^IRX` "
            "(13w bill = the carry / foregone-rate proxy; JP short rates ~0 nearly all sample) + `^TNX`, "
            f"{R['start']} → {R['end']} (as-of {R['as_of']}, last complete month), "
            f"{R['n_days']:,} joint daily returns / {R['n_months']} months. No survivorship (one pair, "
            "one index). **Execution.** The hedge is a *static sleeve held in advance* — no timing "
            "signal, hence no lag to apply (the one documented execution convention); event and quintile "
            "returns use same-close daily returns. Fingerprint `" + R["fingerprint"] + "` "
            "([docs/results.md](../docs/results.md)). Sibling: [69-safe-haven](../../69-safe-haven/README.md) "
            "(gold — same question, store-of-value mechanism)."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · The conditional (downside) beta — HAC/Newey-West\n\n"
            "Piecewise daily regression `jpy ~ a + b_dn·min(spy,0) + b_up·max(spy,0)`; NW lags by the "
            "`4(n/100)^{2/9}` plug-in (10 on the full sample). The safe-haven claim is **b_dn < 0**.\n\n"
            "> 💡 **In plain words:** we let the yen respond *differently* to stock drops and stock "
            "rallies, and ask whether the drop-response is really there once we account for the fact "
            "that currency noise clusters."
        ),
        code(
            "if HAVE_REAL:\n"
            "    B = st.downside_beta(RETS)\n"
            "    print(f\"downside beta {B['beta_dn']:+.3f}  (HAC t = {B['t_dn']:+.2f})\")\n"
            "    print(f\"upside   beta {B['beta_up']:+.3f}  (HAC t = {B['t_up']:+.2f})\")\n"
            "    print(f\"plain    beta {B['beta_all']:+.3f}  (HAC t = {B['t_all']:+.2f})   \"\n"
            "          f\"R2={B['r2']:.3f}  n={B['n']:,}  NW lags={B['lags']}\")\n"
            "else:\n"
            "    print('cache missing - frozen:', R['beta_dn'], R['t_dn'])"
        ),
        md(
            f"**b_dn = {R['beta_dn']:+.3f}, HAC t = {R['t_dn']:+.2f}** — decisively past the desk's "
            f"t ≥ 2 bar, on 29.4 years of daily tape with nothing survivor-selected. The upside leg "
            f"({R['beta_up']:+.3f}, t = {R['t_up']:+.2f}) is less than half the size: the yen is a "
            "*bad-day* currency, exactly as the legend says. This full-sample number is **not** a "
            "two-episode mirage — it survives dropping all six named crash years at once "
            f"(n = {R['drop_all_ev_n']:,}, HAC t = **{R['drop_all_ev_t']:+.2f}**), dropping the 100 "
            f"worst SPY days (HAC t = **{R['drop100_t']:+.2f}**), and a 20-seed stationary bootstrap "
            f"(Welch t mean **+{R['boot_welch_mean']:.2f}**, {R['boot_frac_gt2']}% of seeds > 2)."
        ),

        md(
            "### 1b · …but the daily signal has decayed to zero\n\n"
            "The one thing the full-sample *t* hides is **when** the effect happened. Re-estimate the "
            "downside beta on calendar halves and on the recent decade — the whole signal is "
            "pre-2011, and it is statistically **absent** since.\n\n"
            "> 💡 **In plain words:** graded on the whole tape it clears the bar; graded on a live, "
            "forward basis it does not. The reflex faded as Japan's rate-differential regime changed. "
            "That is why the Signal axis is **Mixed**, not Real."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for lo, hi in [(1996, 2010), (2011, 2026), (2016, 2026)]:\n"
            "        sub = RETS[(RETS.index.year >= lo) & (RETS.index.year <= hi)]\n"
            "        b = st.downside_beta(sub); q = st.quintile_ladder(sub)\n"
            "        print(f'  {lo}-{hi}: n={b[\"n\"]:>5,}  beta_dn={b[\"beta_dn\"]:+.3f} '\n"
            "              f'(HAC t={b[\"t_dn\"]:+.2f})  Q1={q[\"q1_bps\"]:+6.2f} bps/day  '\n"
            "              f'Welch t(Q1 vs mid)={q[\"t_q1_vs_mid\"]:+.2f}')\n"
            "else:\n"
            "    for e in R['sub']:\n"
            "        print('  frozen:', e)"
        ),
        md(
            f"**{R['sub'][0][0]}** carries the whole thing (HAC t = **{R['sub'][0][3]:+.2f}**, Q1 = "
            f"**+{R['sub'][0][4]:.2f} bps/day**). **{R['sub'][1][0]}** — half the sample, 15 years — is "
            f"a flat zero (HAC t = **{R['sub'][1][3]:+.2f}**, Q1 = **{R['sub'][1][4]:+.2f}**, Welch t = "
            f"**+{R['sub'][1][5]:.2f}**), and **{R['sub'][2][0]}** is deader still (HAC t = "
            f"**{R['sub'][2][3]:+.2f}**). The flight-to-yen is real in the tape but **gone from the last "
            "decade** — a fading signal, not a live one."
        ),

        md(
            "## 2 · Quintile ladder + the monthly horizon\n\n"
            "Mean JPY return per SPY-day quintile (Welch t across buckets), then the same idea at the "
            "monthly horizon — where a *bear-market hedge* would have to live.\n\n"
            "> 💡 **In plain words:** the staircase below is the daily hedge. Then we zoom out to whole "
            "months and it's gone — the yen protects you for hours, not quarters."
        ),
        code(
            "if HAVE_REAL:\n"
            "    Q = st.quintile_ladder(RETS)\n"
            "    for k, (mu, n) in enumerate(zip(Q['means_bps'], Q['ns']), start=1):\n"
            "        print(f'  Q{k}: {mu:+6.2f} bps/day  (n={n:,})')\n"
            "    print(f\"  Welch t (Q1 vs Q5)  = {Q['t_q1_vs_q5']:+.2f}\")\n"
            "    print(f\"  Welch t (Q1 vs mid) = {Q['t_q1_vs_mid']:+.2f}\")\n"
            "    DM = st.down_month_split(M)\n"
            "    print(f\"\\n  monthly: down-months (n={DM['n_dn']}) {DM['dn_bps']:+.2f} bps/mo vs \"\n"
            "          f\"up-months (n={DM['n_up']}) {DM['up_bps']:+.2f} bps/mo  ->  Welch t = {DM['t_dn_vs_up']:+.2f}\")\n"
            "    print(f\"  worst-decile SPY months (n={DM['dec_n']}, SPY {DM['dec_spy_pct']:+.2f}%/mo): \"\n"
            "          f\"JPY {DM['dec_jpy_pct']:+.2f}%/mo, up {DM['dec_hit']*100:.0f}% of the time\")\n"
            "else:\n"
            "    print('cache missing - frozen ladder:', R['q_means'])"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))\n"
            "    means = Q['means_bps']\n"
            "    axes[0].bar(range(1, 6), means, color=[GREEN if v > 0 else RED for v in means])\n"
            "    axes[0].axhline(0, color='k', lw=.8)\n"
            "    axes[0].set_xticks(range(1, 6)); axes[0].set_xticklabels(['Q1\\nworst', 'Q2', 'Q3', 'Q4', 'Q5\\nbest'])\n"
            "    axes[0].set_title('daily: JPY by SPY quintile (bps/day)')\n"
            "    dn = M.loc[M['spy'] < 0, 'jpy'] * 1e4\n"
            "    up = M.loc[M['spy'] >= 0, 'jpy'] * 1e4\n"
            "    axes[1].bar(['SPY down-months', 'SPY up-months'], [dn.mean(), up.mean()], color=[GREY, GREY])\n"
            "    axes[1].axhline(0, color='k', lw=.8)\n"
            "    axes[1].set_title(f'monthly: the hedge evaporates (Welch t = {DM[\"t_dn_vs_up\"]:+.2f})')\n"
            "    axes[1].set_ylabel('JPY mean (bps/mo)')\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            f"The daily ladder is emphatic (Q1 = **+{R['q_means'][0]:.2f}** bps/day, Welch t vs Q5 "
            f"= **+{R['t_q1_q5']:.2f}**). The monthly contrast is **t = {R['t_month']:.2f}** — nothing. "
            f"Worst-decile months: **+{R['dec_jpy']:.2f}%/mo**, up only **{R['dec_hit']}%** of the time "
            "(gold's crash coin-flip, again). The negative drift eats the flight-to-quality within weeks."
        ),

        md(
            "## 3 · Event studies\n\n"
            "Window dates hardcoded in `data.EVENTS` (BIS-sourced framing in the comments); returns "
            "always computed from the tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for e in st.event_table(TAPE):\n"
            "        tick = 'HEDGED' if e['hedged'] else 'FAILED'\n"
            "        print(f\"  {e['label']:<44s} {e['start']} -> {e['end']}: \"\n"
            "              f\"SPY {e['spy_pct']:+7.2f}%  JPY {e['jpy_pct']:+7.2f}%  [{tick}]\")\n"
            "else:\n"
            "    for lab, s, e_, sp, jp, h in R['events']:\n"
            "        print(f'  {lab:<44s} {s} -> {e_}: SPY {sp:+7.2f}%  JPY {jp:+7.2f}%')"
        ),
        md(
            "Four clean hedges (2008 **+19.1%** against −27.4%; Aug-2015; COVID's first leg; Aug-2024) "
            "and two failures with a shared signature: the selloff being driven by the **dollar side** — "
            "the Mar-2020 funding scramble (BIS Bulletin 2) and the 2022 rate-shock bear.\n\n"
            "> 💡 **In plain words:** when the panic is about *growth*, money runs to the yen. When the "
            "panic is about *dollars* (everyone needs them, or they suddenly yield 4% more), money runs "
            "**away** from the yen — crash or no crash."
        ),

        md(
            "## 4 · Regime splits — what turns the hedge off\n\n"
            "**(A) The carry-stock proxy.** If the mechanism were *only* carry unwind, the downside beta "
            "should vanish when the USD-JPY differential is ~0 (US ZIRP: nothing to unwind).\n"
            "**(B) Yield direction.** Among SPY down-months: growth scares (10y fell) vs rate shocks "
            "(10y rose)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    RG = st.carry_regime_split(RETS, cut=2.0)\n"
            "    for tag, b, n in [(f\"bill >= 2% ({RG['share_hi']*100:.0f}% of days)\", RG['hi'], RG['n_hi']),\n"
            "                      ('bill <  2% (ZIRP)', RG['lo'], RG['n_lo'])]:\n"
            "        print(f\"  {tag:<26s} n={n:>5,}  beta_dn={b['beta_dn']:+.3f} (HAC t={b['t_dn']:+.2f})\")\n"
            "    YD = st.yield_direction_split(M)\n"
            "    print(f\"\\n  down-months, 10y FELL (growth scare, n={YD['n_fell']}): JPY {YD['fell_bps']:+8.2f} bps/mo\")\n"
            "    print(f\"  down-months, 10y ROSE (rate shock,   n={YD['n_rose']}): JPY {YD['rose_bps']:+8.2f} bps/mo\")\n"
            "    print(f\"  Welch t of the gap = {YD['t_gap']:+.2f}\")"
        ),
        md(
            f"**(A)** The hedge survives *both* regimes (fat differential: b_dn = {R['hi_beta_dn']:+.3f}, "
            f"t = {R['hi_t_dn']:+.2f}; ZIRP: {R['lo_beta_dn']:+.3f}, t = {R['lo_t_dn']:+.2f}) — so the "
            "daily reflex is general risk-off demand (Habib-Stracca's net-foreign-asset story), **not** "
            "purely the size of the carry book. The folk *mechanism* is at best half the story even when "
            "the folk *fact* is right.\n\n"
            f"**(B)** The operative switch is the yield direction: **+{R['fell_bps']:.1f} bps/mo** when "
            f"yields fell vs **{R['rose_bps']:.1f}** when they rose (Welch t = **+{R['t_gap']:.2f}**). "
            "A rate-shock bear *widens* the differential all the way down — carry gets more attractive "
            "during the crash. That is 2022, mechanically.\n\n"
            "> 💡 **In plain words:** the yen is insurance against \"the economy is breaking\", not "
            "against \"my stocks are falling\". In 2022 stocks fell *because* rates rose — the one crash "
            "flavor this insurance is guaranteed to skip."
        ),

        md(
            "## 5 · Tradability — the carry bill\n\n"
            "Excess-vs-excess: the static sleeve's return = JPY spot (price-only; JPY deposits ~0) "
            "**minus** the US bill the dollars would have earned, minus a 40 bps/yr FXY-style expense. "
            "No timing signal → no lag; one entry, <1 bp/yr amortised."
        ),
        code(
            "if HAVE_REAL:\n"
            "    HC = st.hedge_cost(RETS)\n"
            "    print(f\"  JPY spot (price-only): {HC['spot_ann_pct']:+.2f}%/yr over {HC['years']:.1f} years\")\n"
            "    print(f\"  minus avg bill {HC['avg_bill_pct']:.2f}%/yr  ->  excess {HC['excess_ann_pct']:+.2f}%/yr\")\n"
            "    print(f\"  minus {HC['expense_bps']:.0f} bps expense    ->  net    {HC['net_ann_pct']:+.2f}%/yr\")"
        ),
        md(
            f"**{R['net_ann']:+.2f}%/yr net, for {R['years']:.1f} years.** The insurance premium is the "
            "size of the equity risk premium itself — and section 2 showed the monthly hedge value is "
            "statistically zero. Access is not the issue (FXY, futures, spot FX all work at retail); the "
            "**negative carry is the position**. Stamp: **MIRAGE**."
        ),

        md(
            "## 6 · Synthetic control — machinery proof *(never market evidence)*\n\n"
            "Seeded world (`synthetic_world`, seed 615): the planted yen earns `hedge × |SPY|` on SPY "
            "down days only. The null (`hedge = 0`) must stay flat; the plant must light up."
        ),
        code(
            "for hedge in (0.0, 0.30):\n"
            "    world = data.synthetic_world(hedge=hedge, seed=615)\n"
            "    r = st.to_returns(world)\n"
            "    b = st.downside_beta(r)\n"
            "    q = st.quintile_ladder(r)\n"
            "    print(f'  hedge={hedge:.2f}: beta_dn={b[\"beta_dn\"]:+.3f} (HAC t={b[\"t_dn\"]:+.2f})  '\n"
            "          f'Q1={q[\"q1_bps\"]:+.2f} bps/day  Welch t(Q1 vs Q5)={q[\"t_q1_vs_q5\"]:+.2f}')"
        ),
        md(
            f"Null: t = {R['syn'][0][2]:+.2f} (flat). Plant: t = {R['syn'][1][2]:+.2f}. The detector "
            "cannot manufacture a safe haven from noise and cannot miss a real one — so the real-tape "
            f"t = {R['t_dn']:+.2f} is a property of the tape, not the harness.\n\n"
            "## Verdict\n\n"
            f"- **Signal — MIXED (real on the full tape, decayed and fast-twitch).** b_dn = "
            f"**{R['beta_dn']:+.3f}** (HAC t = **{R['t_dn']:+.2f}**), Q1 = **+{R['q_means'][0]:.2f} "
            f"bps/day** (Welch t = **+{R['t_q1_q5']:.2f}**); no survivorship, and robust to dropping "
            f"every episode and to bootstrap — a genuine signal, not a lucky seed. But **non-stationary**: "
            f"{R['sub'][0][0]} carries it (t = {R['sub'][0][3]:+.2f}) while {R['sub'][1][0]} is a flat "
            f"zero (t = {R['sub'][1][3]:+.2f}) and {R['sub'][2][0]} is dead (t = {R['sub'][2][3]:+.2f}). "
            f"Monthly horizon: absent in every era (t = {R['t_month']:.2f}).\n"
            f"- **Tradability — MIRAGE.** **{R['net_ann']:+.2f}%/yr net** for {R['years']:.1f} years, "
            "zero monthly hedge value, and mechanical failure in rate-shock bears.\n"
            f"- **2022 failure — CONFIRMED.** JPY **{R['jpy_2022']:+.2f}%** vs SPY **{R['spy_2022']:+.2f}%** "
            f"top→trough ({R['ratio_2022']}% as large a loss), explained by the yield-direction split "
            f"(Welch t = +{R['t_gap']:.2f}).\n\n"
            "*Every number above is printed by [examples/verify.py](../examples/verify.py) and frozen in "
            "[docs/results.md](../docs/results.md). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)
