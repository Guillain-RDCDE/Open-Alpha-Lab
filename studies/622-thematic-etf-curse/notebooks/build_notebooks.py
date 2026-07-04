"""Generate the two narrative notebooks for Study 622 (Thematic-ETF-Curse).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF panel under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 48 thematic +
# 13 broad ETF launches + SPY + ^IRX, launches 2005-2021, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", fingerprint="d3175d6dbc14",
    n_thematic=48, n_broad=13, launches="2005-2021", hype_vintage=20,
    # calendar-time books: (alpha %/yr, HAC t, beta, months, avg members)
    young36=(-16.30, -3.27, 1.64, 192, 8.7), young36_span=("2008-07-31", "2024-06-30"),
    young12=(-15.00, -2.04, 1.64, 109, 4.5),
    seasoned=(-10.33, -1.82, 1.36, 182, 23.6),
    spread=(-0.26, -0.05, 158),               # (alpha %/yr, t, months) — beta-controlled
    placebo=(0.31, 1.17, 0.99, 72),           # broad-index launches
    # event-time curve: cumulative CAPM-abnormal % by event month (48 funds at every k)
    curve={12: -4.9, 24: -12.4, 36: -22.5, 48: -35.6}, curve_n=48,
    # per-fund first-36m alphas
    perfund_median=-2.18, perfund_pct_neg=65, perfund_n=48,
    # robustness: label -> (alpha %/yr, t)
    rob_lags12=(-16.30, -2.93), rob_min1=(-11.29, -2.32, 232),
    rob_exark=(-16.49, -3.34, 42), rob_pre18=(-18.94, -3.99, 28, 135),
    rob_post18=(-9.56, -0.97, 20, 74),
    # short overlay: gross + (borrow bps -> net %/yr, t)
    short_gross=(16.30, 3.58),
    short=[(300, 13.21, 2.90), (600, 10.21, 2.24), (1000, 6.21, 1.37)],
    # third axis — buy the -50% dip
    dip_events=34, dip_alpha=-19.39, dip_t=-2.80, dip_beta=1.63, dip_n=83,
    dip_fwd12=(-15.1, 76, 34), dip_fwd36=(-36.6, 75, 32),
    # synthetic control: (planted drag %/yr, recovered alpha %/yr, t)
    syn=[(0.0, -0.76, -0.37), (-8.0, -8.76, -4.26)], syn_n=181,
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Buy the dip?: Busted](https://img.shields.io/badge/Buy_the_--50%25_dip%3F-Busted-8b949e?style=flat-square)\n\n"
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

from thematic_etf_curse import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, RF = data.load_prices(), data.load_rf()
    UNIV = data.build_universe(PX, RF, as_of="2026-06-30")
    TH, BR = UNIV["thematic"], UNIV["broad"]
else:
    PX = RF = UNIV = None
    TH = BR = []
print("real cache present:", HAVE_REAL, "| thematic:", len(TH), "| broad placebo:", len(BR))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Thematic ETF Curse — why the fund's birthday is the sell signal 🪤\n"
            "### Solar, cannabis, metaverse, blockchain: what a dollar invested at *every* thematic ETF launch actually earned\n\n"
            + BADGES +
            "Here's a pattern you've lived through even if you never traded it. A theme gets hot — solar in 2008, "
            "robotics in 2016, cannabis in 2018, the metaverse and blockchain in 2021. It's on magazine covers. "
            "Your feed is full of it. And *right then* — never earlier — a fund company launches an ETF so you can "
            "\"get exposure\" with one click.\n\n"
            "A 2023 academic paper (Ben-David, Franzoni, Kim & Moussawi, *Competition for Attention in the ETF "
            "Space*) made a brutal claim about those products: **they launch at peak hype and then lose about 5% "
            "a year, risk-adjusted, for years**. Not because ETFs are bad — broad index ETFs are wonderful — but "
            "because the *launch date itself* marks the top of the theme's attention. The fund is born holding "
            "the theme at its most expensive.\n\n"
            "We test it on **48 real thematic ETF launches (2005–2021)** — the full ARK complex, TAN, HACK, KWEB, "
            "MSOS, BUZZ, METV, BKCH and friends — against **13 boring broad-index ETF launches** as the control.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo regressions and the borrow "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front, and it makes the result STRONGER.** The thematic ETFs that died — and "
            "many did — have vanished from Yahoo's data. Our panel only holds the *survivors*, the best of the "
            "litter. Every negative number below is therefore an **understatement** of the curse."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do thematic ETFs bleed after launch? | **Yes, heavily.** A portfolio that always held the freshly "
            "launched thematics lost about **16% a year** vs what its market risk should have paid (the quants' "
            "*t* = −3.3 — far past the desk's bar). |\n"
            "| Is it just \"new ETFs\" in general? | **No.** The same experiment on broad index-fund launches "
            "(VTI, VOO, Schwab...) shows **zero** bleed. It's the *hype*, not the launch. |\n"
            "| Does the bleed stop? | Not really — older thematics still bled ~10%/yr on this sample. The launch "
            "is when the bleed *starts*, not the only bad year. |\n"
            "| \"It's down 50%, it's cheap now\"? | **Busted.** Buying the halved thematics lost another **37 "
            "percentage points** to the S&P over the next 3 years, three times out of four. |\n"
            "| Can you get rich shorting them? | On paper yes; in practice borrow fees and thin supply eat most "
            "of it. The **free** trade is simply *not buying the launch*. |"
        ),

        md(
            "## 1 · The curse curve — what happens after the first candle\n\n"
            "Take all 48 funds, line them up on their **launch month** (month 1 = first full month of trading), "
            "strip out what the market itself did (each fund's beta × the S&P), and average. This is the pure "
            "\"being a young thematic ETF\" effect, month by month after birth."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.event_curve(UNIV, TH, k_max=48)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(ec.index, ec['cum_ab_pct'], color=RED, lw=2.5)\n"
            "    ax.fill_between(ec.index, ec['cum_ab_pct'], 0, color=RED, alpha=.12)\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    for k in (12, 24, 36, 48):\n"
            "        ax.annotate(f\"{ec.loc[k,'cum_ab_pct']:+.0f}%\", (k, ec.loc[k,'cum_ab_pct']),\n"
            "                    textcoords='offset points', xytext=(0, -14), ha='center', color=RED)\n"
            "    ax.set_xlabel('months since launch'); ax.set_ylabel('cumulative risk-adjusted return (%)')\n"
            "    ax.set_title(f'The curse curve — {len(TH)} thematic ETF launches, average risk-adjusted path after birth')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing — canonical numbers:', R['curve'])\n"
            "print('cumulative risk-adjusted return by event month (canonical):', R['curve'])"
        ),
        md(
            "Read it left to right: **−5% by the first birthday, −12% by the second, −22% by the third, −36% by "
            "month 48**. No dramatic crash on day one — just a steady, grinding bleed as the hype premium leaks "
            "out. And remember: the funds that bled *worst* closed and aren't even in this average.\n\n"
            "> 🔬 **For the quants.** The curve uses each fund's full-sample beta (documented look-ahead, fine "
            "for a descriptive picture). The *headline* test has no look-ahead at all: a calendar-time portfolio "
            "and a Newey-West *t* — notebook 02."
        ),

        md(
            "## 2 · The honest experiment — hold ALL the young thematics, always\n\n"
            "Imagine a fund-of-funds with one dumb rule: **every month, own every thematic ETF that is less than "
            "3 years old**, equal-weighted. That's the strategy \"buy the launches\". Its risk-adjusted return "
            "is the whole claim in one number. We race it against the same rule applied to boring broad-index "
            "launches."
        ),
        code(
            "if HAVE_REAL:\n"
            "    py, _ = st.calendar_time_portfolio(UNIV, TH, 1, 36)\n"
            "    pb, _ = st.calendar_time_portfolio(UNIV, BR, 1, 36)\n"
            "    spy = UNIV['mkt']\n"
            "    fig, ax = plt.subplots()\n"
            "    for s, lbl, c in ((py, 'young THEMATIC launches (first 36m)', RED),\n"
            "                      (pb, 'young BROAD-index launches (placebo)', GREY),\n"
            "                      (spy.reindex(py.index), 'SPY over the same months', GREEN)):\n"
            "        s = s.dropna()\n"
            "        ax.plot(s.index, (1 + s).cumprod(), color=c, lw=2, label=lbl)\n"
            "    ax.set_yscale('log'); ax.legend(); ax.set_ylabel('growth of $1 (log scale)')\n"
            "    ax.set_title('Buy every launch: thematic vs broad vs the market')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing')\n"
            "print(f\"canonical: young thematics alpha {R['young36'][0]:+.2f}%/yr (t={R['young36'][1]:+.2f}) | \"\n"
            "      f\"broad placebo {R['placebo'][0]:+.2f}%/yr (t={R['placebo'][1]:+.2f})\")"
        ),
        md(
            "The young-thematics book lost **−16.3% a year of alpha** — that is, versus what its own (very high) "
            "market risk should have earned. The broad-launch placebo: **+0.3%/yr, statistically zero**. Same "
            "construction, same math, same era. The only difference is *what* was launched: hype versus plumbing.\n\n"
            "The comparison to SPY on the chart looks even crueler than −16%/yr because these funds carry beta "
            "≈ 1.6 — they were *supposed* to beat SPY in the good times just by being risky. They didn't.\n\n"
            "> 💡 **Why does the industry keep doing it?** Because launches follow *attention*. A thematic ETF "
            "gathers assets easiest exactly when the theme is on magazine covers — which is also when it's most "
            "expensive. The fund company harvests fees either way; you hold the bag. That's the paper's whole "
            "mechanism, and our tape agrees."
        ),

        md(
            "## 3 · \"But it's down 50% now — surely it's cheap?\"\n\n"
            "The most seductive counter-move: wait for the crash, buy the wreckage. 34 of our 48 funds did lose "
            "half their value from their post-launch peak. Suppose you bought each one the month after it first "
            "closed −50% down, and held for 3 years. Versus just buying SPY the same day:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dp = st.dip_portfolio(UNIV, TH, dd=0.50, hold=36)\n"
            "    f36 = np.array(dp['fwd36']) * 100\n"
            "    fig, ax = plt.subplots()\n"
            "    f = np.sort(f36)\n"
            "    ax.bar(range(len(f)), f, color=[RED if v < 0 else GREEN for v in f])\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.set_xlabel('each -50% dip-buy event (sorted)'); ax.set_ylabel('3-year return vs SPY (pp)')\n"
            "    ax.set_title(f\"Buying the -50% dip: 3-year result vs SPY, {len(f)} events (mean {f.mean():+.0f} pp)\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing')\n"
            "print(f\"canonical: mean 3y vs SPY {R['dip_fwd36'][0]:+.1f} pp, {R['dip_fwd36'][1]}% of \"\n"
            "      f\"{R['dip_fwd36'][2]} events negative | dip-book alpha {R['dip_alpha']:+.2f}%/yr (t={R['dip_t']:+.2f})\")"
        ),
        md(
            "**Busted.** The average dip-buy lost another **37 points** to the S&P over three years; **three out "
            "of four** dips lost. \"Down 50%\" doesn't mean cheap — it means the hype premium is *half* deflated. "
            "In these products, the knife keeps falling.\n"
        ),

        md(
            "## The takeaway\n\n"
            "1. **The curse is real.** Across 48 launches and 20 years, freshly launched thematic ETFs bled "
            "~16%/yr of risk-adjusted return; the boring-launch placebo bled nothing. And the dead funds Yahoo "
            "forgot would only make it worse.\n"
            "2. **The launch date is the tell.** The product exists *because* the theme just peaked in attention. "
            "You're not early — the launch proves you're late.\n"
            "3. **The dip is not your friend** here. −50% was mid-bleed, not a bottom, three times out of four.\n"
            "4. **The tradable lesson is free.** Shorting the launches works on paper but borrow fees and thin "
            "supply eat most of it (quants' notebook). Simply *declining to buy* the shiny new theme fund — and "
            "buying the boring broad fund instead — captures the entire effect at zero cost.\n\n"
            "*Research & education, not investment advice. Numbers: [docs/results.md](../docs/results.md), "
            "as-of 2026-06-30.*"
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
            "# Thematic ETF Curse — the quant teardown 🪤\n\n"
            + BADGES +
            "**Claim (Ben-David, Franzoni, Kim & Moussawi, RFS 2023).** Specialized/thematic ETFs launch at peak "
            "hype and deliver ≈ −5%/yr risk-adjusted returns over their first years; the launch date is the sell "
            "signal.\n\n"
            "**Design.** 48 thematic ETF launches 2005–2021 (launch = first candle, issuer-inception guard "
            "against ticker reuse), 13 broad plain-vanilla index-ETF launches as placebo, SPY market, ^IRX "
            "risk-free. Primary test: **calendar-time portfolio** of funds in event months 1..W, CAPM on SPY "
            "excess, **Newey-West t (lags 6)**. Calendar-time (Fama 1998) because pooled fund-month regressions "
            "pseudo-replicate the same crash months. One execution lag everywhere; total-return prices; "
            "excess-vs-excess. **Survivorship named: dead thematics are absent from yfinance, which biases "
            "AGAINST the finding.** As-of 2026-06-30, fingerprint `d3175d6dbc14`.\n\n"
            "Frozen canonical numbers live in `R` (mirror of [docs/results.md](../docs/results.md)); real-tape "
            "cells recompute them from the cache."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · Headline — calendar-time CAPM alphas\n\n"
            "$(r_{p,t} - r_{f,t}) = \\alpha + \\beta\\,(r_{m,t} - r_{f,t}) + \\varepsilon_t$, HAC t on $\\alpha$ "
            "(Bartlett, 6 lags). Portfolios: young thematics (event months 1–12, 1–36; ≥3 members), seasoned "
            "thematics (37+), the beta-controlled young−seasoned spread, and the broad-launch placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lbl, res in (('young 1-12', st.young_alpha(UNIV, TH, w=12)),\n"
            "                     ('young 1-36', st.young_alpha(UNIV, TH, w=36)),\n"
            "                     ('seasoned 37+', st.seasoned_alpha(UNIV, TH)),\n"
            "                     ('BROAD placebo 1-36', st.young_alpha(UNIV, BR, w=36))):\n"
            "        rows.append({'book': lbl, 'alpha %/yr': round(res['alpha_ann_pct'], 2),\n"
            "                     'HAC t': round(res['t_alpha'], 2), 'beta': round(res['beta'], 2),\n"
            "                     'months': res['n']})\n"
            "    sp = st.young_vs_seasoned_spread(UNIV, TH, w=36)\n"
            "    rows.insert(3, {'book': 'young - seasoned spread', 'alpha %/yr': round(sp['alpha_ann_pct'], 2),\n"
            "                    'HAC t': round(sp['t_alpha'], 2), 'beta': round(sp['beta'], 2), 'months': sp['n']})\n"
            "    display(pd.DataFrame(rows).set_index('book'))\n"
            "else:\n"
            "    print('cache missing — canonical: young36', R['young36'], '| placebo', R['placebo'])"
        ),
        code(
            "fig, ax = plt.subplots()\n"
            "labels = ['young 1-12', 'young 1-36', 'seasoned 37+', 'broad placebo']\n"
            "alphas = [R['young12'][0], R['young36'][0], R['seasoned'][0], R['placebo'][0]]\n"
            "ts     = [R['young12'][1], R['young36'][1], R['seasoned'][1], R['placebo'][1]]\n"
            "cols = [RED, RED, AMBER, GREY]\n"
            "bars = ax.bar(labels, alphas, color=cols)\n"
            "for b, t in zip(bars, ts):\n"
            "    ax.annotate(f't = {t:+.2f}', (b.get_x() + b.get_width()/2, b.get_height()),\n"
            "                textcoords='offset points', xytext=(0, -14 if b.get_height() < 0 else 4),\n"
            "                ha='center', fontsize=10)\n"
            "ax.axhline(0, color=GREY, lw=1)\n"
            "ax.set_ylabel('CAPM alpha (%/yr)')\n"
            "ax.set_title('Calendar-time CAPM alpha vs SPY — Newey-West t (lags 6), canonical numbers')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**Signal — REAL.** Young thematics (1–36): **−16.30%/yr, HAC t = −3.27**; the 1–12 book −15.00%/yr "
            "(t = −2.04). The placebo is clean (+0.31%/yr, t = +1.17, beta 0.99): the construction does not "
            "manufacture bleed out of *launching* — only out of launching **hype**.\n\n"
            "**The honest nuance, on the stamp:** seasoned thematics bleed too (−10.33%/yr, t = −1.82), and the "
            "beta-controlled young−seasoned spread is **−0.26%/yr, t = −0.05**. The tape certifies *thematics "
            "bleed from the first candle onward*; it cannot certify that youth is **extra** toxic relative to "
            "being a thematic at all. Ben-David et al.'s launch-timing framing survives in the practical sense "
            "(the launch date starts the clock and marked the attention peak), not as a young-vs-old spread.\n\n"
            "> 💡 **In plain words.** New theme funds lost hugely versus their risk; new boring funds didn't. "
            "But old theme funds kept losing as well — the birthday is when the bleeding *starts*, not the only "
            "bad part."
        ),

        md(
            "## 2 · Event-time curve + coverage\n\n"
            "Mean CAPM-abnormal return by event month (per-fund **full-sample** beta — a documented look-ahead, "
            "descriptive only), cumulated; plus the young-book membership over calendar time (the launch waves: "
            "2008 clean energy, 2014-16 robotics/cyber, 2018-21 everything)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.event_curve(UNIV, TH, k_max=48)\n"
            "    _, cnt = st.calendar_time_portfolio(UNIV, TH, 1, 36)\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))\n"
            "    axes[0].bar(ec.index, ec['mean_ab'] * 100, color=RED, alpha=.55, label='mean abnormal (%/mo)')\n"
            "    ax2 = axes[0].twinx(); ax2.plot(ec.index, ec['cum_ab_pct'], color=RED, lw=2.2, label='cumulative (%)')\n"
            "    ax2.grid(False)\n"
            "    axes[0].set_xlabel('event month'); axes[0].set_ylabel('mean abnormal (%/mo)')\n"
            "    ax2.set_ylabel('cumulative (%)'); axes[0].set_title('Event-time CAPM-abnormal returns (48 funds)')\n"
            "    axes[1].plot(cnt.index, cnt.values, color=GREY, lw=1.8)\n"
            "    axes[1].set_title('Young-book membership (event months 1-36)'); axes[1].set_ylabel('# funds')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('cumulative by k:', {k: round(ec.loc[k, 'cum_ab_pct'], 1) for k in (12, 24, 36, 48)})\n"
            "else:\n"
            "    print('canonical curve:', R['curve'])"
        ),
        md(
            "Canonical: **−4.9% (12m), −12.4% (24m), −22.5% (36m), −35.6% (48m)**, all 48 funds contributing at "
            "every horizon. The bleed is steady, not a single crash.\n\n"
            "### Per-fund vs calendar-time — the clustering is the mechanism\n"
            "One first-36-months alpha per fund: median **−2.18%/yr**, **65% negative** (n=48) — much milder "
            "than the calendar-time −16.30%/yr. The gap is the paper's mechanism in one contrast: launches "
            "**cluster at attention peaks**, so the calendar months crowded with young funds (2008, 2021-22) were "
            "precisely the bleed months. Equal-weighting *months* (what a launch-buying dollar experiences) is "
            "the economically relevant footing; equal-weighting *funds* dilutes the crowded disasters."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pf = st.per_fund_alphas(UNIV, TH, w=36)\n"
            "    fig, ax = plt.subplots()\n"
            "    v = pf['alpha_ann_pct'].sort_values()\n"
            "    ax.bar(range(len(v)), v, color=[RED if x < 0 else GREEN for x in v])\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.set_title(f'Per-fund first-36m CAPM alpha (median {v.median():+.1f}%/yr, '\n"
            "                 f'{(v < 0).mean()*100:.0f}% negative)')\n"
            "    ax.set_ylabel('alpha %/yr'); ax.set_xlabel('fund (sorted)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print('canonical per-fund: median', R['perfund_median'], '%/yr,', R['perfund_pct_neg'], '% negative')"
        ),

        md(
            "## 3 · Robustness\n\n"
            "Lag choice, membership floor, the ARK complex, and the vintage split."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [('headline (lags 6, >=3 members)', st.young_alpha(UNIV, TH, w=36)),\n"
            "            ('NW lags = 12', st.young_alpha(UNIV, TH, w=36, lags=12)),\n"
            "            ('min members = 1', st.young_alpha(UNIV, TH, w=36, min_members=1)),\n"
            "            ('ex-ARK', st.young_alpha(UNIV, [t for t in TH if not t.startswith('ARK') and t != 'IZRL'], w=36)),\n"
            "            ('pre-2018 vintage', st.young_alpha(UNIV, [t for t in TH if UNIV['launch'][t].year < 2018], w=36)),\n"
            "            ('2018+ vintage', st.young_alpha(UNIV, [t for t in TH if UNIV['launch'][t].year >= 2018], w=36))]\n"
            "    display(pd.DataFrame([{'variant': l, 'alpha %/yr': round(r['alpha_ann_pct'], 2),\n"
            "                           'HAC t': round(r['t_alpha'], 2), 'months': r['n']} for l, r in rows])\n"
            "            .set_index('variant'))\n"
            "else:\n"
            "    print('canonical:', R['rob_lags12'], R['rob_min1'], R['rob_exark'], R['rob_pre18'], R['rob_post18'])"
        ),
        md(
            "The headline survives everything structural: lags 12 → **t = −2.93**; membership floor dropped → "
            "−11.29%/yr, **t = −2.32**; the entire ARK complex removed → −16.49%/yr, **t = −3.34**. The vintage "
            "split: pre-2018 launches −18.94%/yr (**t = −3.99**); the 2018+ vintage alone is −9.56%/yr at "
            "**t = −0.97** — 74 calendar months that are all one overlapping macro episode (2018→2024), so the "
            "sub-sample carries little independent information. Same sign, not separately certifiable — said out "
            "loud, not hidden.\n\n"
            "> 💡 **In plain words.** However we slice it, the answer keeps its sign; the only slice that can't "
            "clear the bar alone is the shortest one, because six overlapping years is barely one weather system."
        ),

        md(
            "## 4 · Tradability — shorting the curse\n\n"
            "Short the young book, hedge β × SPY long. Costs: 5 bps one-way × NAV amortized over the 36-month "
            "window (entries + exits, both legs); **borrow paid on the full short notional**, swept 300/600/1000 "
            "bps/yr — young niche ETFs are exactly where borrow gets ugly (MSOS-type names ran to double digits "
            "in the wild)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for b in (300.0, 600.0, 1000.0):\n"
            "        so = st.short_overlay(UNIV, TH, w=36, cost_bps=5.0, borrow_bps_ann=b)\n"
            "        rows.append({'borrow bps/yr': int(b), 'net %/yr': round(so['net_ann_pct'], 2),\n"
            "                     'HAC t': round(so['t_net'], 2)})\n"
            "    print(f\"gross: {so['gross_ann_pct']:+.2f}%/yr  t={so['t_gross']:+.2f}  (beta hedge {so['beta_hedge']:.2f})\")\n"
            "    display(pd.DataFrame(rows).set_index('borrow bps/yr'))\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "bb = [r[0] for r in R['short']]; nn = [r[1] for r in R['short']]; tt = [r[2] for r in R['short']]\n"
            "ax.plot(bb, nn, 'o-', color=AMBER, lw=2)\n"
            "for x, y, t in zip(bb, nn, tt):\n"
            "    ax.annotate(f't={t:+.2f}', (x, y), textcoords='offset points', xytext=(8, 6))\n"
            "ax.axhline(0, color=GREY, lw=1)\n"
            "ax.set_xlabel('borrow (bps/yr)'); ax.set_ylabel('net alpha (%/yr)')\n"
            "ax.set_title('The short vs borrow reality (canonical)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**Tradability — FRAGILE.** Net **+13.21%/yr (t = +2.90)** at 300 bps borrow, still clearing at 600 "
            "(**t = +2.24**), dead by 1000 (**t = +1.37**). And the modeled sweep is generous: availability on "
            "young niche ETFs is thin, recalls and squeezes on a β≈1.6 short book are real, and the bleed is "
            "concentrated in the very names with the worst borrow. The *accessible* implementation — refuse the "
            "launch, buy the broad fund — captures the effect at zero cost but produces no short alpha. A real "
            "signal in a hard-to-hold wrapper."
        ),

        md(
            "## 5 · Third axis — buy the −50% dip?\n\n"
            "First month-end ≥50% below the post-launch running max (total-return levels), enter the **next** "
            "month (one lag), hold 36 months. Calendar-time book + per-event forward runs vs SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dp = st.dip_portfolio(UNIV, TH, dd=0.50, hold=36)\n"
            "    f12, f36 = np.array(dp['fwd12']), np.array(dp['fwd36'])\n"
            "    print(f\"events: {dp['n_events']} | book alpha {dp['alpha_ann_pct']:+.2f}%/yr  t={dp['t_alpha']:+.2f}  \"\n"
            "          f\"beta {dp['beta']:.2f}  ({dp['n']} months)\")\n"
            "    print(f\"fwd 12m vs SPY: {f12.mean()*100:+.1f} pp  ({(f12<0).mean()*100:.0f}% negative, n={len(f12)})\")\n"
            "    print(f\"fwd 36m vs SPY: {f36.mean()*100:+.1f} pp  ({(f36<0).mean()*100:.0f}% negative, n={len(f36)})\")\n"
            "else:\n"
            "    print('canonical: alpha', R['dip_alpha'], 't', R['dip_t'], '| fwd36', R['dip_fwd36'])"
        ),
        md(
            "**BUSTED.** The dip-bought book bleeds **−19.39%/yr** of alpha (**t = −2.80**) — *faster* than the "
            "young book itself. Forward vs SPY: **−15.1 pp** over 12m (76% negative), **−36.6 pp** over 36m (75% "
            "negative, n=32). −50% in a thematic is mid-bleed, not value.\n\n"
            "> 💡 **In plain words.** Half-off on a hype fund isn't a sale price — it's the hype being half-way "
            "done deflating."
        ),

        md(
            "## 6 · Synthetic positive control — the machinery is faithful\n\n"
            "Deterministic seeded panel (staggered launches, β ≈ 1.2, idio noise) with a **planted** post-launch "
            "alpha drag over event months 1–36. The null (drag 0) must not fire; the planted drag must be "
            "recovered. *(Machinery proof only — never cited in support of the stamps.)*"
        ),
        code(
            "for drag in (0.0, -0.08):\n"
            "    u = data.synthetic_world(drag_ann=drag, seed=622)\n"
            "    r = st.young_alpha(u, u['thematic'], w=36)\n"
            "    print(f\"planted drag {drag*100:+5.0f}%/yr -> recovered alpha {r['alpha_ann_pct']:+.2f}%/yr  \"\n"
            "          f\"HAC t = {r['t_alpha']:+.2f}  ({r['n']} months)\")"
        ),
        md(
            "Null: −0.76%/yr, t = −0.37 (no manufactured bleed). Planted −8%/yr: recovered −8.76%/yr, t = −4.26.\n\n"
            "## Verdict\n\n"
            "- **Signal — REAL.** Young-thematics calendar-time CAPM alpha **−16.30%/yr, HAC t = −3.27** (W=12: "
            "t = −2.04), robust to lags/floor/ex-ARK; broad-launch placebo clean; **survivorship (named) biases "
            "against the finding**. Nuance on the stamp: the young-vs-seasoned spread is ≈ 0 — the category "
            "bleeds at every age, launch is when it starts.\n"
            "- **Tradability — FRAGILE.** The hedged short clears at modeled borrow (300–600 bps, t = 2.90/2.24) "
            "but dies at real-world niche borrow (1000 bps → t = 1.37); capacity and recalls throttle it; the "
            "free version is avoidance, not alpha.\n"
            "- **Buy the −50% dip? — BUSTED.** Dip book −19.39%/yr (t = −2.80); the average dip lost 36.6 pp to "
            "SPY over 3 years; 75% of events negative.\n\n"
            "*Reproduce: `python examples/verify.py` (as-of 2026-06-30, fingerprint `d3175d6dbc14`). "
            "Sources: [docs/references.md](../docs/references.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in (("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)):
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)
