"""Generate the two narrative notebooks for Study 896 (Risk-Parity + Trend).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached six-ETF
panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance six-ETF panel,
# race 2008-04-01 -> 2026-06-30, 4,591 daily bars, as-of 2026-06-30, fp 003721e41f89).
R = dict(
    start="2008-04-01", end="2026-06-30", years=18.2, n_days=4591, n_cache=4802,
    lookback=60, sma=200,
    p_cagr=6.94, p_vol=9.42, p_sharpe=0.626, p_dd=-19.95, p_wealth=3.39,
    t_cagr=6.15, t_vol=7.11, t_sharpe=0.702, t_dd=-9.68, t_wealth=2.97,
    sharpe_adv=0.077, dd_relief=10.28, ret_diff=-0.93, t_ret=-0.65,
    avg_gate=63.5, avg_risky=0.646, turn_plain=0.99, turn_trend=2.40,
    # bootstrap of the Sharpe difference
    bs_obs=0.077, bs_lo=-0.248, bs_hi=0.392, bs_p=0.672,
    # eras: (name, sharpe_plain, sharpe_trend, adv, dd_plain, dd_trend, t_ret)
    eras=[("era 1  2008-04 -> 2017-05", 0.502, 0.684, 0.181, -19.27, -9.68, 0.08),
          ("era 2  2017-05 -> 2026-06", 0.749, 0.722, -0.027, -19.95, -8.85, -1.09)],
    # crisis windows: (name, plain DD %, trend DD %)
    crises=[("2008 GFC", -19.27, -7.84), ("2020 COVID", -16.60, -7.14),
            ("2022 bond bear", -19.95, -8.76)],
    # costs: (one-way bps, sharpe_plain, sharpe_trend, adv, dd_plain, dd_trend, cagr_plain, cagr_trend, t_ret)
    costs=[(0, 0.626, 0.702, 0.077, -19.95, -9.68, 6.94, 6.15, -0.65),
           (5, 0.620, 0.686, 0.065, -19.97, -9.75, 6.88, 6.03, -0.70),
           (10, 0.615, 0.669, 0.054, -20.00, -9.88, 6.83, 5.90, -0.74),
           (20, 0.604, 0.635, 0.031, -20.04, -10.33, 6.72, 5.65, -0.84)],
    # placebo (200 shuffled gates)
    pl_obs_adv=0.077, pl_mean_adv=-0.049, pl_p_sharpe=0.135,
    pl_obs_dd=-9.68, pl_mean_dd=-17.13, pl_p_dd=0.000, pl_seeds=200,
    # calendar years: (year, plain %, trend %)
    cal=[(2008, -2.39, 10.14), (2009, 7.55, 3.87), (2010, 19.06, 12.55),
         (2011, 12.30, 8.07), (2012, 7.21, -2.50), (2013, -4.26, 4.95),
         (2014, 0.57, 4.73), (2015, -9.17, -1.89), (2016, 10.60, 5.15),
         (2017, 13.11, 9.06), (2018, -2.74, -1.34), (2019, 17.48, 12.36),
         (2020, 12.85, 9.58), (2021, 13.98, 11.67), (2022, -9.43, -3.17),
         (2023, 9.08, 3.24), (2024, 11.87, 9.93), (2025, 18.90, 14.60),
         (2026, 6.70, 3.59)],
    # synthetic control (20 seeds/world)
    syn_null_adv=-0.175, syn_null_dd=4.05, syn_null_share=15,
    syn_pl_adv=0.254, syn_pl_dd=35.70, syn_pl_t=2.00, syn_pl_share=90,
    fingerprint="003721e41f89",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
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

from rp_trend import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_prices()
    RET = st.daily_returns(PX)
    CASH = RET[data.CASH]
    RACE = st.race(PX, RET, CASH, data.SLEEVES)     # headline, gross, excess-of-cash
else:
    PX = RET = CASH = RACE = None
print("real ETF cache present:", HAVE_REAL,
      "| race days:", (0 if RACE is None else RACE["n_days"]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does bolting a trend switch onto risk-parity help? 🔀\n"
            "### Inverse-vol SPY/TLT/GLD/DBC, plus a 200-day \"hold-it-only-if-it's-rising\" gate\n\n"
            + BADGES +
            "**Risk parity** is the grown-up way to hold four very different things — stocks, "
            "long Treasuries, gold, commodities — by giving each a slice of *risk* rather than a "
            "slice of dollars (the calm bond sleeve gets a big weight, the wild commodity sleeve a "
            "small one). It diversifies beautifully in normal times. Its weakness: it rides every "
            "sleeve straight through that sleeve's *own* bear market — long bonds fell ~40% in "
            "2022 and risk-parity just sat there.\n\n"
            "The fix everyone reaches for is a **trend switch**: *hold a sleeve only while it's "
            "above its 200-day average; when it rolls over, move that sleeve's money to cash "
            "(T-bills) until it's rising again.* The pitch is you keep the diversification **and** "
            "duck the sustained downtrends. Does it actually improve things — a better "
            "risk-adjusted return, a smaller worst-case loss — once you count the cost?\n\n"
            "> 📓 **Plain-language layer.** Want the bootstrap CI, the two-era cut, the shuffle "
            "test and the cost sweep? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note.** yfinance daily *total-return* ETFs, 2008 → mid-2026 (18 years — "
            "one history; the four sleeves are young and hand-picked, so read the levels as one "
            "18-year survivor tape). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the trend gate cut the drawdown? | **Yes — genuinely, and it's timing.** Worst "
            f"loss goes from **{R['p_dd']:.0f}%** to **{R['t_dd']:.0f}%** (roughly halved), holds in "
            "both halves of the history and in every crisis, and a shuffle test shows it's the "
            "*timing* that does it (0 of 200 randomised gates match it). |\n"
            "| Does it improve the risk-adjusted return? | **Can't be certified.** The Sharpe "
            f"nudges up ({R['p_sharpe']:.2f} → {R['t_sharpe']:.2f}) but the edge (+{R['sharpe_adv']:.3f}) "
            f"has a confidence interval of **[{R['bs_lo']:.2f}, +{R['bs_hi']:.2f}]** — it straddles "
            "zero, and it even flips sign between the two eras. |\n"
            "| Is it free? | **No.** The gate sits in cash a third of the time, so you give up a "
            f"little growth: **×{R['t_wealth']:.1f}** your money instead of **×{R['p_wealth']:.1f}** "
            "over 18 years. |\n\n"
            "> The honest version: **the trend switch is a real *shield*, not a real *edge*.** It "
            "reliably shrinks the crashes; it does not reliably make you richer per unit of risk."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Start from a risk-parity book across stocks / long bonds / gold / commodities "
            "(each weighted inversely to its own volatility). Then gate each sleeve with its "
            "200-day moving average: above the line, hold it; below the line, park that sleeve's "
            "risk in T-bills. You keep the diversification in good times and step out of the "
            "sustained downtrends — better Sharpe, smaller drawdown.\"*\n\n"
            "This stacks two respectable ideas: **risk parity** (Qian 2005; Asness-Frazzini-"
            "Pedersen 2012) for the diversified core, and **trend / time-series momentum** "
            "(Moskowitz-Ooi-Pedersen 2012; Faber 2007) for the de-risking overlay. We test the "
            "simplest concrete version, rebalanced monthly, with **yesterday's** prices deciding "
            "**today's** gate (one clean lag, no peeking)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two different promises hide in that pitch, and they deserve different grades:\n\n"
            "1. **A smaller worst case** — *can a mechanical rule step out of the sleeves that are "
            "grinding down and shrink the drawdown?* If yes, that's genuinely useful: a −20% "
            "portfolio is one you might bail on at the bottom; a −10% one you can hold.\n"
            "2. **A better deal per unit of risk** — *do you actually earn more Sharpe, not just "
            "less volatility?* That's the alpha claim. Cash drag and whipsaw can quietly eat it.\n\n"
            "Sales decks blur the two. The tape can separate them."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The tape.** {R['n_cache']:,} daily total-return closes of six ETFs, "
            f"{R['start']} → {R['end']} ({R['years']:.0f} years — 2008, 2020 and the 2022 bond "
            "bear all included).\n"
            "- **The two books.** *Plain RP*: inverse-vol weights on the four sleeves, monthly. "
            "*RP+trend*: the same weights, but any sleeve below its 200-day average has its risk "
            "moved to T-bills for the month.\n"
            "- **The race.** Both books measured **excess of cash** (return minus T-bills), on the "
            "same days: Sharpe, worst drawdown, growth.\n"
            "- **The luck test.** Rebuild the gated book 200 times with each sleeve's gate "
            "*shuffled in time* (same amount of time out of the market, zero information). If the "
            "shuffled versions duck the crashes just as well, the shield was only *holding less*, "
            "not timing.\n"
            "- **The honesty tests.** Split the 18 years in half (does the edge survive?), and "
            "charge realistic trading costs (do they erase it?)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The ride itself.** Same starting dollar; the gated book (green) vs plain risk-parity "
            "(grey)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nav_p = (1 + RACE['plain_total']).cumprod()\n"
            "    nav_t = (1 + RACE['trend_total']).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(nav_p.index, nav_p, c=GREY, lw=2, label='plain risk-parity')\n"
            "    ax.plot(nav_t.index, nav_t, c=GREEN, lw=2, label='RP + 200d trend gate')\n"
            "    for nav, col in ((nav_p, GREY), (nav_t, GREEN)):\n"
            "        ax.annotate(f'x{nav.iloc[-1]:.2f}', (nav.index[-1], nav.iloc[-1]), color=col,\n"
            "                    xytext=(8,0), textcoords='offset points', va='center')\n"
            "    ax.set_ylabel('growth of $1'); ax.set_title('Smoother, and a touch lower')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print(f\"terminal wealth: RP+trend x{nav_t.iloc[-1]:.2f}  vs  plain RP x{nav_p.iloc[-1]:.2f}\")\n"
            "else:\n"
            "    print('cache missing - frozen: RP+trend x%.2f vs plain x%.2f' % (R['t_wealth'], R['p_wealth']))"
        ),
        md(
            f"The green line is calmer and finishes a little lower (**×{R['t_wealth']:.1f}** vs "
            f"**×{R['p_wealth']:.1f}**). Now the two halves of the promise, one at a time.\n\n"
            "**Half one: the crashes.** Worst peak-to-trough loss inside each crisis."
        ),
        code(
            "names = [c[0] for c in R['crises']]\n"
            "pdd = [c[1] for c in R['crises']]; tdd = [c[2] for c in R['crises']]\n"
            "if HAVE_REAL:\n"
            "    wins = {'2008 GFC':('2008-04-01','2009-06-30'), '2020 COVID':('2020-02-01','2020-04-30'),\n"
            "            '2022 bond bear':('2022-01-01','2022-12-31')}\n"
            "    names, pdd, tdd = [], [], []\n"
            "    for nm,(a,b) in wins.items():\n"
            "        m = (RACE['plain_total'].index>=pd.Timestamp(a)) & (RACE['plain_total'].index<=pd.Timestamp(b))\n"
            "        names.append(nm); pdd.append(st.max_drawdown(RACE['plain_total'][m])*100)\n"
            "        tdd.append(st.max_drawdown(RACE['trend_total'][m])*100)\n"
            "x = np.arange(len(names)); wdt = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x - wdt/2, pdd, wdt, color=GREY, label='plain RP')\n"
            "ax.bar(x + wdt/2, tdd, wdt, color=GREEN, label='RP + trend')\n"
            "for i,v in enumerate(pdd): ax.annotate(f'{v:.0f}%', (x[i]-wdt/2, v), ha='center', va='top', fontsize=9)\n"
            "for i,v in enumerate(tdd): ax.annotate(f'{v:.0f}%', (x[i]+wdt/2, v), ha='center', va='top', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('max drawdown (%)'); ax.set_ylim(-24, 0)\n"
            "ax.set_title('Every crisis roughly cut in half'); ax.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(names, zip([round(v,1) for v in tdd], [round(v,1) for v in pdd]))))"
        ),
        md(
            f"Each crash shrinks by about half — **{R['crises'][0][1]:.0f}% → {R['crises'][0][2]:.0f}%** "
            f"in 2008, **{R['crises'][2][1]:.0f}% → {R['crises'][2][2]:.0f}%** in the 2022 bond bear "
            "(the gate stepped out of Treasuries as they fell below trend). And the quants notebook "
            "shows this is *timing*, not just holding less: 200 shuffled gates (same time out of the "
            f"market, no information) average a **{R['pl_mean_dd']:.0f}%** drawdown — **0 of 200** "
            "match the real one. The shield earns its badge.\n\n"
            "**Half two: is it a better deal per unit of risk?** Here's the calendar, year by year."
        ),
        code(
            "rows = R['cal']\n"
            "if HAVE_REAL:\n"
            "    cy = st.calendar_years(RACE)\n"
            "    rows = [(int(y), cy.loc[y,'plain_%'], cy.loc[y,'trend_%']) for y in cy.index]\n"
            "yrs = [r[0] for r in rows]; gaps = [r[2]-r[1] for r in rows]\n"
            "cols = [GREEN if g>=0 else RED for g in gaps]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.4))\n"
            "ax.bar([str(y) for y in yrs], gaps, color=cols, width=.7)\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('trend - plain (pp)'); ax.tick_params(axis='x', rotation=60)\n"
            "ax.set_title('When the gate helps (green) vs drags (red), by year')\n"
            "plt.tight_layout(); plt.show()\n"
            "wins = sum(1 for g in gaps if g>0)\n"
            "print(f'gate added return in {wins}/{len(gaps)} years; biggest wins are the crisis years')"
        ),
        md(
            "The pattern is the whole story: the gate **wins big in the crisis years** (2008 "
            "**+12.5 pp**, 2015 **+7.3 pp**, 2022 **+6.3 pp**) and **loses small in the calm bull "
            "years** (2010, 2012, 2019, 2023 each give back 5–10 pp of cash drag and whipsaw). Add "
            f"it all up and the Sharpe barely moves (+{R['sharpe_adv']:.3f}, statistically "
            "indistinguishable from zero) — you paid for the crisis insurance with bull-market "
            "premiums.\n\n"
            "> 🔬 **For the quants:** the Sharpe edge's 95% confidence interval is "
            f"**[{R['bs_lo']:.2f}, +{R['bs_hi']:.2f}]** (straddles zero) and it flips sign between "
            "the two eras. The drawdown relief, by contrast, is certified as timing (shuffle "
            f"p = {R['pl_p_dd']:.3f}). Different grades for different halves."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** *Real on the drawdown* (halved, robust across eras and crises, "
            "certified as timing by the shuffle test), *weak on the Sharpe* (the risk-adjusted "
            f"edge never clears significance, +{R['sharpe_adv']:.3f} with a CI through zero, and it "
            "flips sign era to era).\n"
            "- **Tradability — Fragile.** Cheap to run (low turnover, penny-spread ETFs, no "
            "shorts) and it survives realistic costs — but what survives scrutiny is **risk "
            "control, not extra return**, and it cost a little growth on this one 18-year tape.\n"
            "- **In one line:** a genuine *shield*, not a certified *edge*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The gate is honest about what it is** — a *risk* dial. If a −20% risk-parity "
            "drawdown is the difference between you holding and you capitulating, paying a little "
            "growth to halve it can be rational. Just buy it with open eyes.\n"
            "- **Why doesn't the Sharpe pop?** Because the 200-day gate is *slow* — it catches the "
            "long grinds (2008, 2022) but whipsaws in choppy ranges and always parks in cash a "
            "beat late. The crashes it prevents and the premiums it misses roughly cancel.\n"
            "- **Siblings on this desk.** [68-all-weather](../../68-all-weather/README.md) is the "
            "plain risk-parity book this study gates; [110-faber-timing](../../110-faber-timing/"
            "README.md) is the same 200-day rule on a single asset; "
            "[894-trend-6040](../../894-trend-6040/README.md) puts trend on a 60/40 instead of a "
            "risk-parity budget.\n\n"
            "*Think a faster gate, or gating the whole book instead of sleeve-by-sleeve, would "
            "turn the shield into an edge? Show us a version whose Sharpe advantage clears zero in "
            "both eras and we'll re-grade.*"
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
            "# Risk-Parity + Trend — a quantitative teardown 🔬\n"
            "### excess-vs-excess Sharpe race · paired Sharpe-diff bootstrap · two-era cut · "
            "200-seed shuffled-gate placebo · cost sweep · a two-world synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — *add a 200-day trend gate to an inverse-vol risk-parity book and improve the "
            "excess-of-cash Sharpe and the drawdown* — splits into a **risk claim** (shallower "
            "drawdown, beyond mere de-risking) and a **return claim** (a genuine Sharpe / excess-"
            "return advantage). We grade them separately, on the tape.\n\n"
            "> ⚠️ **Data note.** yfinance daily total-return SPY/TLT/GLD/DBC (sleeves) + BIL (cash), "
            + R['start'] + " → " + R['end'] + f" ({R['n_days']:,} race bars after the 200d SMA + "
            "60d vol burn-in, ann = 252; both legs excess-of-cash minus BIL). Four young, hand-"
            "picked sleeves on one 18-year survivor tape — named on the Signal axis. Offline core "
            "+ synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R['fingerprint'] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the drawdown:* max DD **{R['t_dd']:.2f}% vs "
            f"{R['p_dd']:.2f}%**, robust in both eras & every crisis, shuffled-gate placebo "
            f"**p = {R['pl_p_dd']:.3f}** (200 seeds). *Weak on the Sharpe:* advantage "
            f"**+{R['sharpe_adv']:.3f}**, bootstrap 95% CI **[{R['bs_lo']:.2f}, +{R['bs_hi']:.2f}]** "
            f"(straddles 0), excess-ret diff {R['ret_diff']:.2f}%/yr (HAC t = {R['t_ret']:.2f}), "
            "sign flips era-to-era. |\n"
            f"| **Tradability** | `FRAGILE` | Turnover {R['turn_trend']:.2f}× NAV/yr, penny-spread "
            f"ETFs, no borrow; survives 20 bps (adv {R['costs'][3][3]:+.3f}). But the certified "
            f"deliverable is risk control; terminal wealth ×{R['t_wealth']:.1f} vs "
            f"×{R['p_wealth']:.1f}. |\n\n"
            "> 💡 In plain words: the crash shield is real and certified; the free lunch is not on "
            "this tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The construction, precisely\n\n"
            "Four sleeves $i \\in \\{$SPY, TLT, GLD, DBC$\\}$. Each month $m$, on information known "
            "at the prior close:\n\n"
            "$$w_i = \\frac{1/\\hat\\sigma_i}{\\sum_j 1/\\hat\\sigma_j}, \\qquad "
            "g_i = \\mathbf{1}\\{P_i \\ge \\mathrm{SMA}_{200}(P_i)\\},$$\n\n"
            "with $\\hat\\sigma_i$ the trailing 60-day realized vol. **Plain RP** holds $w_i$ in "
            "each sleeve; **RP+trend** holds the *effective* weight $w_i g_i$ and parks the "
            "complement $\\sum_i w_i(1-g_i)$ in cash (BIL). Both are held through the month (one "
            "`shift(1)` lag). Because $\\sum_i w_i = 1$, the excess-of-cash returns are\n\n"
            "$$r^{plain}_t - c_t = \\sum_i w_i (r_{i,t} - c_t), \\qquad "
            "r^{trend}_t - c_t = \\sum_i w_i\\, g_i\\, (r_{i,t} - c_t),$$\n\n"
            "so the gate simply **zeroes a sleeve's excess contribution while it is below trend** — "
            "it never re-levers into the survivors.\n\n"
            "- **H₁ (risk).** RP+trend cuts max drawdown beyond what any same-frequency exposure "
            "profile would (timing, not de-risking).\n"
            "- **H₂ (return).** RP+trend earns a positive **excess-vs-excess Sharpe advantage** "
            "whose bootstrap CI clears zero and holds across sub-eras.\n\n"
            "We find **H₁ supported** (DD placebo p = 0.000, robust both eras) and **H₂ "
            "unsupported** (Sharpe adv +0.077, CI through zero, sign flips era-to-era)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two inference traps\n\n"
            "1. **Exposure confound.** The gated book sits in cash ~35% of sleeve-months, so ANY "
            "same-frequency de-risking mechanically shrinks the drawdown. The shield only counts "
            "if it beats *shuffled* gates — same time-out-of-market per sleeve, no alignment with "
            "the actual downtrend (200 seeds; the desk bans single-seed baselines).\n"
            "2. **Serial correlation + a paired comparison.** Daily excess returns are "
            "autocorrelated (vol clustering) and the two books share the same tape, so the return "
            "test is a **HAC (Newey-West) t on the paired daily excess difference**, and the "
            "Sharpe-advantage CI is a **paired circular block-bootstrap** of the Sharpe *difference* "
            "(21-day blocks, the pairing preserved).\n\n"
            "Everything is **excess-vs-excess** (both legs minus BIL), so the risk-free convention "
            "cancels and we compare like with like."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** {R['n_cache']:,} daily total-return closes, {R['start']} → {R['end']} "
            f"({R['years']:.0f} yrs of race sample, {R['n_days']:,} days), as-of pinned, fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **Books.** Plain RP vs RP+trend, monthly, one lag; de-risked sleeves earn BIL.\n"
            "- **Headline stats.** CAGR, ann vol, Sharpe (excess-of-cash), max DD, terminal wealth; "
            "**HAC t on the daily excess-return difference**; **paired Sharpe-diff bootstrap**.\n"
            "- **Robustness.** Two-era cut; a per-crisis drawdown ledger; a calendar-year table.\n"
            "- **Placebo.** 200 shuffled-gate seeds → p-values for the Sharpe advantage and the DD "
            "shield.\n"
            "- **Costs.** One-way bps × turnover × NAV per monthly rebalance (long-or-cash, no "
            "borrow): 0 / 5 / 10 / 20 bps.\n"
            "- **Positive control.** Seeded bull/bear-regime worlds: a no-downtrend null (the gate "
            "must not help) vs a planted-bear world (it must light up), 20 seeds each."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline race and the Sharpe-difference bootstrap\n\n"
            "Both books' full stats, then the paired block-bootstrap of the Sharpe *difference* — "
            "the direct test of the Sharpe-advantage claim."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for leg, p in (('plain RP', RACE['plain']), ('RP+trend', RACE['trend'])):\n"
            "        print(f\"{leg:<9}: CAGR {p['cagr_pct']:+6.2f}%  vol {p['vol_ann_pct']:5.2f}%  \"\n"
            "              f\"Sharpe {p['sharpe']:+.3f}  maxDD {p['maxdd_pct']:+7.2f}%  wealth x{p['wealth_mult']:.2f}\")\n"
            "    print(f\"Sharpe advantage {RACE['sharpe_adv']:+.3f}  |  drawdown relief {RACE['dd_relief_pp']:+.2f} pp\")\n"
            "    print(f\"excess-ret diff {RACE['ret_diff_ann_pct']:+.2f}%/yr  HAC t = {RACE['t_ret_diff']:+.2f}\")\n"
            "    bs = st.sharpe_diff_bootstrap(RACE['trend_excess'], RACE['plain_excess'], n_boot=2000)\n"
            "    print(f\"Sharpe-diff bootstrap: obs {bs['obs']:+.3f}  95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]  P(>0)={bs['p_gt0']:.3f}\")\n"
            "    fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "    rng = np.random.default_rng(896); a = RACE['trend_excess'].to_numpy(); b = RACE['plain_excess'].to_numpy()\n"
            "    n = len(a); draws = []\n"
            "    for _ in range(2000):\n"
            "        st_ = (rng.integers(0, n, int(np.ceil(n/21)))[:,None] + np.arange(21)[None,:]).ravel() % n\n"
            "        st_ = st_[:n]; sa, sb = a[st_], b[st_]\n"
            "        draws.append(sa.mean()/sa.std(ddof=1)*np.sqrt(252) - sb.mean()/sb.std(ddof=1)*np.sqrt(252))\n"
            "    ax.hist(draws, bins=40, color=GREY, alpha=.85)\n"
            "    ax.axvline(0, c=RED, lw=1.5, ls='--', label='zero')\n"
            "    ax.axvline(bs['obs'], c=GREEN, lw=2.5, label=f\"observed {bs['obs']:+.3f}\")\n"
            "    ax.set_xlabel('Sharpe advantage (trend - plain)'); ax.set_ylabel('resamples')\n"
            "    ax.set_title(f\"Sharpe edge straddles zero: CI [{bs['lo']:+.2f}, {bs['hi']:+.2f}], P(>0)={bs['p_gt0']:.2f}\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing - frozen:', {k: R[k] for k in ('p_sharpe','t_sharpe','sharpe_adv','bs_lo','bs_hi','bs_p','t_ret')})"
        ),
        md(
            f"> 💡 In plain words: the gate lifts the Sharpe from {R['p_sharpe']:.2f} to "
            f"{R['t_sharpe']:.2f} and halves the drawdown — but the Sharpe *advantage* "
            f"(+{R['sharpe_adv']:.3f}) has a bootstrap CI of **[{R['bs_lo']:.2f}, +{R['bs_hi']:.2f}]** "
            f"and only a {R['bs_p']:.0%} chance of being positive. The excess-return leg is actually "
            f"slightly negative ({R['ret_diff']:.2f}%/yr, t = {R['t_ret']:.2f}): the gate trades a "
            "little return for a lot less risk. Uncertified as an edge."
        ),
        md(
            "### 4b · Two eras — the Sharpe edge doesn't hold, the shield does\n\n"
            "Split the 18 years in half. A real edge should survive both."
        ),
        code(
            "rows = R['eras']\n"
            "if HAVE_REAL:\n"
            "    rows = [(e['era'], e['sharpe_plain'], e['sharpe_trend'], e['sharpe_adv'],\n"
            "             e['maxdd_plain_pct'], e['maxdd_trend_pct'], e['t_ret_diff'])\n"
            "            for e in st.era_cut(PX, RET, CASH, data.SLEEVES)]\n"
            "labels = ['era 1\\n2008-2017', 'era 2\\n2017-2026']\n"
            "advs = [r[3] for r in rows]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(labels, advs, color=[GREEN if v>0 else RED for v in advs], width=.5)\n"
            "a1.axhline(0, c=GREY, lw=1); a1.set_ylabel('Sharpe advantage (trend - plain)')\n"
            "a1.set_title('Sharpe edge flips sign across eras')\n"
            "for i,v in enumerate(advs): a1.annotate(f'{v:+.3f}', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "xx = np.arange(2); w = .38\n"
            "a2.bar(xx-w/2, [r[4] for r in rows], w, color=GREY, label='plain')\n"
            "a2.bar(xx+w/2, [r[5] for r in rows], w, color=GREEN, label='trend')\n"
            "a2.set_xticks(xx); a2.set_xticklabels(labels); a2.set_ylabel('max drawdown (%)')\n"
            "a2.set_title('Drawdown relief holds in both'); a2.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows:\n"
            "    print(f'{r[0]}: Sharpe {r[1]:+.3f}->{r[2]:+.3f} (adv {r[3]:+.3f})  maxDD {r[4]:+.2f}%->{r[5]:+.2f}%  t_ret {r[6]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: era 1 (crisis-dense: GFC, euro crisis) gives the gate a "
            f"**+{R['eras'][0][3]:.3f}** Sharpe edge; era 2 (a long bull with brief shocks) gives "
            f"**{R['eras'][1][3]:+.3f}** — it *inverts*. The whole Sharpe story is the 2008–2011 "
            "window. The drawdown relief, by contrast, is a clean halving in **both** eras. That's "
            "the signature of risk control without a return edge."
        ),
        md(
            "### 4c · The placebo — timing, or just holding less?\n\n"
            "200 shuffled-gate books: each sleeve keeps its on/off *frequency* but the gate is "
            "permuted across months (no alignment with the actual downtrend). If the real book's "
            "drawdown/Sharpe sit inside the shuffled cloud, the \"skill\" was mere de-risking."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_shuffle(PX, RET, CASH, data.SLEEVES, n_seeds=200)\n"
            "    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "    a1.hist(pl['adv_draws'], bins=30, color=GREY, alpha=.85)\n"
            "    a1.axvline(pl['obs_sharpe_adv'], c=GREEN, lw=2.5, label=f\"observed {pl['obs_sharpe_adv']:+.3f}\")\n"
            "    a1.set_xlabel('Sharpe advantage'); a1.set_ylabel('shuffles')\n"
            "    a1.set_title(f\"Sharpe: p = {pl['p_sharpe']:.3f} - NOT certified\"); a1.legend()\n"
            "    a2.hist(pl['dd_draws'], bins=30, color=GREY, alpha=.85)\n"
            "    a2.axvline(pl['obs_maxdd_pct'], c=GREEN, lw=2.5, label=f\"observed {pl['obs_maxdd_pct']:.1f}%\")\n"
            "    a2.set_xlabel('max drawdown (%)')\n"
            "    a2.set_title(f\"DD shield: p = {pl['p_dd']:.3f} - genuine timing\"); a2.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"Sharpe adv: obs {pl['obs_sharpe_adv']:+.3f} vs placebo mean {pl['placebo_mean_adv']:+.3f}  p={pl['p_sharpe']:.3f}\")\n"
            "    print(f\"maxDD: obs {pl['obs_maxdd_pct']:+.2f}% vs placebo mean {pl['placebo_mean_maxdd_pct']:+.2f}%  p={pl['p_dd']:.3f}\")\n"
            "else:\n"
            "    print('cache missing - frozen: p_sharpe %.3f  p_dd %.3f' % (R['pl_p_sharpe'], R['pl_p_dd']))"
        ),
        md(
            f"> 💡 In plain words: random same-frequency gates get you a **{R['pl_mean_dd']:.0f}%** "
            f"drawdown; the *timed* gate gets **{R['t_dd']:.1f}%**, and **0 of 200** shuffles match "
            f"it (p = {R['pl_p_dd']:.3f}) — the shield is *information*, not dilution. The Sharpe "
            f"advantage, though, is beaten by ~1 shuffle in 7 (p = {R['pl_p_sharpe']:.3f}): not "
            "certifiable. This is the cleanest statement of the split verdict."
        ),
        md(
            "### 4d · Costs — cheap to run, and it doesn't change the story\n\n"
            f"Turnover is {R['turn_trend']:.2f}× NAV/yr (vs {R['turn_plain']:.2f}× for plain RP); "
            "the books are long-or-cash so there is no borrow leg."
        ),
        code(
            "rows = R['costs']\n"
            "if HAVE_REAL:\n"
            "    rows = [(c['cost_bps'], c['sharpe_plain'], c['sharpe_trend'], c['sharpe_adv'],\n"
            "             c['maxdd_plain_pct'], c['maxdd_trend_pct'], c['cagr_plain_pct'],\n"
            "             c['cagr_trend_pct'], c['t_ret_diff']) for c in st.cost_sweep(PX, RET, CASH, data.SLEEVES)]\n"
            "print(f\"{'cost':>6} | {'Sh plain':>9} {'Sh trend':>9} {'adv':>7} | {'maxDD p/t':>16} | {'t_ret':>6}\")\n"
            "for cb, shp, sht, adv, ddp, ddt, cgp, cgt, tr in rows:\n"
            "    print(f'{cb:>4.0f}bp | {shp:>9.3f} {sht:>9.3f} {adv:>+7.3f} | {ddp:>7.2f}%/{ddt:>6.2f}% | {tr:>+6.2f}')"
        ),
        md(
            f"> 💡 In plain words: even at a punitive 20 bps one-way, the Sharpe advantage only "
            f"slides {R['costs'][0][3]:+.3f} → {R['costs'][3][3]:+.3f} and the drawdown shield is "
            "intact. Costs are **not** the problem with this strategy; the absent certified edge "
            "is. What fails the bar gross fails it net."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Two seeded bull/bear-regime worlds, 20 seeds each: a **no-downtrend null** (edge=0, "
            "the gate must earn nothing) and a **planted-bear world** (edge=1, sustained down-"
            "grinds the 200-day gate should step out of — it MUST light up)."
        ),
        code(
            "res = [st.synthetic_check(edge=e, n_seeds=20) for e in (0.0, 1.0)]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "labels = ['NULL\\n(no downtrend)', 'PLANTED\\n(bear regimes)']\n"
            "advs = [r['mean_sharpe_adv'] for r in res]\n"
            "ax.bar(labels, advs, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c=RED, lw=1.2, ls='--')\n"
            "for i, r in enumerate(res):\n"
            "    ax.annotate(f\"adv={r['mean_sharpe_adv']:+.3f}\\n(DD relief {r['mean_dd_relief_pp']:+.1f}pp)\",\n"
            "                (i, r['mean_sharpe_adv']), ha='center', va='bottom' if advs[i]>=0 else 'top')\n"
            "ax.set_ylabel('mean Sharpe advantage (20 seeds)')\n"
            "ax.set_title('Control: null earns nothing, planted world lights up')\n"
            "plt.tight_layout(); plt.show()\n"
            "for name, r in zip(('null', 'planted'), res):\n"
            "    print(f\"{name:<8}: mean Sharpe adv {r['mean_sharpe_adv']:+.3f} +/- {r['sd_sharpe_adv']:.3f}  \"\n"
            "          f\"DD relief {r['mean_dd_relief_pp']:+.2f}pp  share adv>0 {r['share_adv_pos']*100:.0f}%\")"
        ),
        md(
            f"> 💡 In plain words: with no persistent downtrend the gate only adds whipsaw — it "
            f"*hurts* the Sharpe (mean {R['syn_null_adv']:+.3f}, positive on only "
            f"{R['syn_null_share']:.0f}% of seeds), exactly as a null should. Plant sustained bear "
            f"regimes and the same gate lifts the Sharpe (mean {R['syn_pl_adv']:+.3f}, positive on "
            f"{R['syn_pl_share']:.0f}% of seeds) and cuts drawdown ~{R['syn_pl_dd']:.0f} pp. The "
            f"machinery can bank a real trend premium when one exists — so the thin real-tape edge "
            "is the tape talking, not a blind harness. *(A faithful-engine / power check only — "
            "never cited in support of a stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the drawdown:* max DD **{R['t_dd']:.2f}% vs "
            f"{R['p_dd']:.2f}%** (halved), robust in both eras and every crisis window (2008, 2020, "
            f"2022 each cut ~in half), certified as timing by the shuffled-gate placebo "
            f"(**p = {R['pl_p_dd']:.3f}**, 200 seeds; placebo mean DD {R['pl_mean_dd']:.2f}%). "
            f"*Weak on the Sharpe:* advantage **+{R['sharpe_adv']:.3f}**, bootstrap CI "
            f"**[{R['bs_lo']:.2f}, +{R['bs_hi']:.2f}]** through zero, excess-ret diff "
            f"{R['ret_diff']:.2f}%/yr (HAC t = {R['t_ret']:.2f}), sign flips era-to-era. Four young "
            "survivor sleeves on one 18-year tape, named.\n"
            f"- **Tradability `FRAGILE`** — turnover {R['turn_trend']:.2f}× NAV/yr, penny-spread "
            f"ETFs, no borrow, robust to 20 bps (adv {R['costs'][3][3]:+.3f}). But the certified "
            f"deliverable is risk control — terminal wealth ×{R['t_wealth']:.1f} vs "
            f"×{R['p_wealth']:.1f}, and the uncertified Sharpe edge rests on one crisis-front-loaded "
            "tape. Not INVESTABLE as an *edge*; deployable as a *shield*.\n"
            "- **In one line:** adding trend to risk-parity **genuinely halves the drawdown** but "
            "**does not certifiably improve the Sharpe** — a Mixed signal in a cheap-to-run Fragile "
            "vehicle."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why no Sharpe pop?** The 200-day gate catches the *long* grinds (2008, 2022) but "
            "whipsaws in ranges and lags every turn by construction. The premiums it misses in "
            "bull years roughly offset the crashes it prevents — hence a halved drawdown but a "
            "flat, uncertified Sharpe.\n"
            "- **Risk overlay vs return engine.** The DD placebo (p = 0.000) is the study's most "
            "durable fact: a trailing 200-day filter *forecasts* which sleeve is about to keep "
            "falling. That is bankable as **position sizing / risk control** even though the Sharpe "
            "edge never certifies.\n"
            "- **The obvious next studies.** A faster gate, gating the whole book on its own "
            "equity curve rather than sleeve-by-sleeve, or re-levering the risk budget onto the "
            "surviving sleeves instead of parking in cash. Compare "
            "[68-all-weather](../../68-all-weather/README.md) (the plain book), "
            "[110-faber-timing](../../110-faber-timing/README.md) (the single-asset gate) and "
            "[894-trend-6040](../../894-trend-6040/README.md) (trend on 60/40).\n\n"
            "*The reproducible core is offline and deterministic; the signal is the per-sleeve "
            "200-day SMA gate with one month of lag. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
