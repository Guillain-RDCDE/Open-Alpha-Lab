"""Generate the two narrative notebooks for Study 520 (External-Financing-Anomaly).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached fundamentals + price
panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 45-name large-cap
# survivor basket, fundamentals fp ca0833b2ddb8, prices fp 4106bae4e4d7, signal years 2022-2024).
R = dict(
    n_names=45, n_nameyears=223, fp_fund="ca0833b2ddb8", fp_px="4106bae4e4d7",
    px_start="2018-01-02", px_end="2026-06-25", report_lag=90, hold_days=252, asof="2026-06-26",
    n_cross=3,
    # gross long-short
    ls_mean=-8.54, ls_vol=13.19, ls_sharpe=-0.648, ls_t=-1.122, ls_hac=-1.687, ls_hit=33.3,
    # net (10 bps/leg + 50 bps/yr borrow)
    net_mean=-9.24, net_t=-1.214,
    # year-by-year: (year, long%, short%, spread%, n)
    years=[(2022, 20.3, 25.0, -4.7, 40), (2023, 14.6, 12.3, 2.3, 45), (2024, 14.0, 37.2, -23.2, 45)],
    # placebo
    placebo_real=-8.54, placebo_null=0.08, placebo_null_std=7.95, placebo_p=0.862,
    # robustness: (label, mean%, t)
    robust=[("median", -10.03, -1.305), ("tercile", -8.54, -1.122), ("quartile", -6.70, -1.175)],
    # synthetic control (20 seeds)
    syn_t_planted=7.615, syn_t_null=-0.097, syn_spread=9.06, syn_frac2=100, syn_seeds=20,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Raisers_underperform_here%3F: Busted](https://img.shields.io/badge/Raisers_underperform_here%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from external_financing_anomaly import data, strategy as st

FUND = data.fetch_fundamentals()
PX = data.fetch_prices()
HAVE_REAL = (not FUND.empty) and (not PX.empty)
if HAVE_REAL:
    SIG = st.by_calendar_year(st.scaled_xfin(FUND))
    FWD = st.forward_returns(PX, SIG.index, report_lag_days=data.REPORT_LAG_DAYS)
else:
    SIG = FWD = None
print("real external-financing cache present:", HAVE_REAL,
      "| name-years:", (0 if FUND is None or FUND.empty else FUND.shape[0]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Should you avoid companies that raise lots of money? 💸\n"
            "### The external-financing anomaly — a real academic finding that **backfired** on the AI-capex era\n\n"
            + BADGES +
            "Here is a tidy piece of finance-professor wisdom: when a company raises **lots** of outside "
            "money — borrowing heavily *and* selling new shares — it tends to **underperform** afterward. "
            "The story is half common-sense (firms drowning in fresh cash tend to waste it on empire-"
            "building) and half cynicism (managers love to sell stock when their shares are **expensive**). "
            "So the trade writes itself: avoid the big **raisers**, favour the disciplined **retirers** who "
            "are paying *down* debt and buying *back* stock.\n\n"
            "It is a genuine published result (Bradshaw, Richardson & Sloan, 2006). The question this desk "
            "always asks is the same: **does it still work when you actually try to trade it — on real "
            "names, with real costs, with no peeking at the future?** Here the answer is a flat **no**, and "
            "for an instructive reason.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Two data notes up front.** (1) We use a fixed **45-name large-cap basket** (names still "
            "trading today) — that carries **survivorship**: the companies that raised money and then "
            "*blew up* are missing, which can only make raisers look *better* than they really were. "
            "(2) Free fundamentals (yfinance) go back only ~5 fiscal years, so we get **just three** "
            "complete yearly tests. Thin tape, stated openly. House style in "
            "[METHODOLOGY.md](../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the big money-raisers underperform here? | **No — the opposite.** Going long the "
            f"disciplined retirers and short the big raisers **lost {abs(R['ls_mean']):.1f}%/yr**. The "
            "raisers were the *winners*. |\n"
            "| Is that just bad luck in one year? | **Mostly 2024.** In 2024 the heaviest raisers — the "
            "debt-funded **AI-capex mega-caps** — beat the retirers by **+23 points**. The anomaly assumes "
            "raising money is a red flag; in the AI build-out it was a green light. |\n"
            "| Could the result be a fluke of how we sorted? | **It's just noise.** A *random* sort beats "
            f"our real one **{R['placebo_p']*100:.0f}%** of the time. There is no signal to find. |\n"
            "| So is the professor wrong? | **Not necessarily** — but on *this* basket, in *this* window, "
            "with *these* survivors, the effect is absent (and reversed). |\n\n"
            "> A real academic anomaly, honestly replicated on a small survivor basket with real costs, "
            "**disappears** — exactly the outcome the desk sees most often."
        ),

        md(
            "## 1 · The claim\n\n"
            "> *\"A company's **net external financing** — the cash it raised from lenders and "
            "shareholders, minus what it paid back — predicts **lower** future returns. Firms that raise a "
            "lot overinvest and over-issue; firms that return capital are disciplined. Buy the disciplined, "
            "avoid the thirsty.\"*\n\n"
            "You can read this straight off the **cash-flow statement** — the bottom third, *financing "
            "activities*: debt issued minus debt repaid, plus shares issued minus shares bought back minus "
            "dividends paid. Scale it by the size of the company (its assets) and you have the signal. "
            "Big positive = a thirsty raiser. Big negative = a disciplined retirer."
        ),

        md(
            "## 2 · So what?\n\n"
            "If raising money really were a reliable red flag, you could build a market-neutral book — long "
            "the retirers, short the raisers — and earn a premium for the discipline. That's the academic "
            "promise. The traps: (1) **survivorship** — we only see the raisers that *survived*, not the "
            "ones that raised and died, so the short leg is flattered; and (2) the **regime** — *why* a "
            "firm raises money matters. Borrowing to fund a doomed acquisition is one thing; borrowing to "
            "build the data centres everyone is paying for is another."
        ),

        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name** large-cap basket and pull each firm's **cash-flow "
            f"statement** and **balance sheet** ({R['n_nameyears']} name-years). For each fiscal year we "
            "compute the scaled net external financing, then:\n\n"
            "1. **Sort** the basket from biggest retirer to biggest raiser.\n"
            "2. **Bet:** go long the bottom third (retirers), short the top third (raisers), equal-weighted.\n"
            f"3. **Wait honestly:** treat the numbers as public only **{R['report_lag']} days** after the "
            "fiscal year ends (a 10-K filing lag), enter the *next* trading day, and hold **one year**. No "
            "look-ahead.\n"
            "4. **Repeat** each year and ask whether the long-short made money — testing it against a "
            "thousands-of-shuffles \"could this be luck?\" null."
        ),

        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Year by year: did the disciplined retirers beat the thirsty raisers?** Here is the spread "
            "(long retirers minus short raisers) for each of the three complete years. Above zero = the "
            "anomaly works; below zero = the raisers won."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(SIG, FWD, quantile=0.3)\n"
            "    yrs = [int(d.year) for d in ls.index]; sp = (ls['spread']*100).tolist()\n"
            "else:\n"
            "    yrs = [y[0] for y in R['years']]; sp = [y[3] for y in R['years']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "cols = [GREEN if v>0 else RED for v in sp]\n"
            "ax.bar([str(y) for y in yrs], sp, color=cols, width=.6)\n"
            "for i,v in enumerate(sp): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long-short spread (%)')\n"
            "ax.set_title('Long retirers / short raisers — the anomaly fails 2 of 3 years')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by year:', {y: round(v,1) for y,v in zip(yrs, sp)})"
        ),
        md(
            f"Two of the three years are **red**. The worst is **2024** "
            f"(**{R['years'][2][3]:+.1f}%**): that was the year the market's biggest external-finance "
            "raisers were the AI-infrastructure mega-caps, and they ran the hardest. The anomaly says "
            "\"avoid the raisers\"; the tape said \"the raisers are exactly who you wanted.\""
        ),
        md(
            "**Is the overall result distinguishable from random?** We shuffle the financing labels "
            "thousands of times (destroying any real link) and see how often a *random* sort beats ours."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(SIG, FWD, quantile=0.3, n_shuffle=2000)\n"
            "    real = pl['real_mean']*100; nullm = pl['null_mean']*100; nstd = pl['null_std']*100; p = pl['p_value']\n"
            "else:\n"
            "    real = R['placebo_real']; nullm = R['placebo_null']; nstd = R['placebo_null_std']; p = R['placebo_p']\n"
            "rng = np.random.default_rng(520)\n"
            "draws = rng.normal(nullm, nstd, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random label shuffles')\n"
            "ax.axvline(real, c=RED, lw=2.5, label=f'real sort {real:+.1f}%')\n"
            "ax.set_xlabel('long-short spread (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real sort sits INSIDE the luck cloud: placebo p = {p:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real {real:+.1f}%   null mean {nullm:+.1f}% (std {nstd:.1f})   placebo p = {p:.2f}')"
        ),
        md(
            f"The red line sits **squarely inside** the grey cloud of random outcomes — a random sort beats "
            f"the real one **{R['placebo_p']*100:.0f}%** of the time. There is simply **no signal** in the "
            "external-financing sort on this tape."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The long-short loses **{R['ls_mean']:+.1f}%/yr** (the sign is "
            f"*reversed*), and a random sort beats it {R['placebo_p']*100:.0f}% of the time. No edge.\n"
            "- **Tradability — Mirage.** With only **three** complete yearly tests there is nothing to "
            "trade, and costs only make the loss worse.\n"
            "- **\"Do big raisers underperform here?\" — Busted.** In this 2022–2024 mega-cap window the "
            "heaviest raisers were the *winners* — debt-funded AI-capex beat balance-sheet discipline."
        ),

        md(
            "## 6 · Could you actually trade it? — costs only deepen the hole\n\n"
            "For completeness, here is the spread **gross** vs **net** of realistic trading costs and short "
            "borrow. When the gross number is already negative, costs can't rescue it — they just dig deeper."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.summary(st.long_short(SIG, FWD, quantile=0.3)['spread'])['mean']*100\n"
            "    n = st.summary(st.long_short(SIG, FWD, quantile=0.3, cost_bps=10, borrow_ann_bps=50)['spread_net'])['mean']*100\n"
            "else:\n"
            "    g = R['ls_mean']; n = R['net_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['gross','net of costs\\n+ borrow'], [g, n], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([g,n]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short return (%/yr)')\n"
            "ax.set_title('A negative edge that costs make worse'); plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr   net {n:+.1f}%/yr')"
        ),
        md(
            f"Gross **{R['ls_mean']:+.1f}%/yr**, net **{R['net_mean']:+.1f}%/yr**. There is no version of "
            "this trade that pays on this tape."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **Survivorship is the thumb on the scale.** The short (raiser) leg is built only from "
            "raisers that *survived* to 2026; the ones that raised and cratered are gone, which flatters "
            "the raisers. A clean test needs a point-in-time universe (CRSP/Compustat), not a survivor "
            "basket.\n"
            "- **Regime matters.** The anomaly is about *wasteful* financing. The 2022–2024 window was an "
            "AI-capex boom where heavy borrowing funded the most-wanted assets on Earth — the opposite of "
            "waste. A full-cycle, pre-2020 sample would treat it more fairly.\n"
            "- **Depth is the binding constraint.** Free fundamentals give ~5 fiscal years; the original "
            "study used decades. Three cross-sections can't certify *anything*.\n\n"
            "*Think the external-financing anomaly is alive on large caps? Show a long-retirers/short-"
            "raisers book clearing **t ≥ 2** after honest costs on a point-in-time universe — then we'll talk.*"
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
            "# The External-Financing Anomaly — a quantitative teardown 🔬\n"
            "### Scaled net-external-financing sort · long-retirers/short-raisers · one-sample & HAC *t* · "
            "label-shuffle placebo · cut-robustness · costs + borrow · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Bradshaw–"
            "Richardson–Sloan (2006) is a *pointed* academic factor; the desk's job is to replicate it "
            "honestly on a small survivor basket with real costs and let the verdict fall where it lands. "
            "Here it lands `NONE` × `MIRAGE` — and the synthetic control proves that is the **tape**, not a "
            "broken engine.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **45-name large-cap** basket, names still trading in "
            "2026 — a *survivor* panel that flatters the short (raiser) leg. Real data: yfinance "
            "`.cashflow` / `.balance_sheet` + daily adjusted closes; free fundamentals reach only ~5 "
            "fiscal years, so just **3 complete annual cross-sections** survive the lag + 12-month hold. "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Long-retirers/short-raisers **{R['ls_mean']:+.1f}%/yr** "
            f"(one-sample **t = {R['ls_t']:.2f}**, HAC **t = {R['ls_hac']:.2f}**); label-shuffle placebo "
            f"**p = {R['placebo_p']:.2f}** — the sign is *reversed* and a random sort wins "
            f"{R['placebo_p']*100:.0f}% of the time. |\n"
            f"| **Tradability** | `MIRAGE` | Only **{R['n_cross']}** complete annual cross-sections survive "
            f"yfinance's 5-year depth; net of 10 bps/leg + 50 bps borrow the spread is "
            f"**{R['net_mean']:+.1f}%/yr**. Nothing to trade. |\n"
            f"| **Raisers underperform here?** | `BUSTED` | In a 2022–2024 mega-cap window the heaviest "
            f"raisers (AI-capex) **out**-earned (2024 short leg beat the long leg by "
            f"{R['years'][2][1]-R['years'][2][2]:+.0f}pts). The engine recovers a *planted* penalty at "
            f"**t = {R['syn_t_planted']:.1f}**, so the flat result is the tape, not the sort. |\n\n"
            "> 💡 In plain words: a genuine published anomaly, replicated honestly, evaporates here — wrong "
            "sign, no significance, three data points. On-brand for a pointed factor on a survivor basket."
        ),

        md(
            "## 1 · The claim, formalised\n\n"
            "For firm $i$ in fiscal year $t$, net external financing from the cash-flow statement is\n\n"
            "$$\\mathrm{XFIN}_{i,t} = \\underbrace{(\\Delta\\text{Equity})_{i,t}}_{\\text{issued}-\\text{repurchased}-\\text{dividends}} "
            "\\;+\\; \\underbrace{(\\Delta\\text{Debt})_{i,t}}_{\\text{issued}-\\text{repaid}},$$\n\n"
            "scaled by average total assets $\\tfrac12(A_{i,t}+A_{i,t-1})$. Sort the cross-section each "
            "year; long the bottom tercile (retirers, low XFIN), short the top tercile (raisers, high "
            "XFIN). The spread is\n\n"
            "$$\\widehat{\\Lambda}_t = \\frac1{n_L}\\!\\sum_{i\\in \\text{retirers}}\\! r_{i,t+1} \\;-\\; "
            "\\frac1{n_S}\\!\\sum_{i\\in \\text{raisers}}\\! r_{i,t+1},$$\n\n"
            "with $r_{i,t+1}$ the 12-month forward return entered **one day after** the (lagged) public "
            "date. **H₁:** $\\overline{\\Lambda} > 0$ and significant. We find $\\overline{\\Lambda} < 0$ "
            "and insignificant — H₁ rejected on this tape."
        ),

        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the annual spread series against zero, "
            "$t = \\overline\\Lambda / (s_\\Lambda/\\sqrt k)$, with a HAC variant for the (mild) serial "
            "dependence. Two honesty problems sit on top. **(a) Survivorship:** the raiser leg is built "
            "only from raisers that *survived*, so the short leg's return is biased *up* — against the "
            "anomaly. **(b) Power:** with $k=3$ years even a real effect would struggle to clear $t=2$, so "
            "we run a **label-shuffle placebo** (is the sort better than random?) and a **synthetic "
            "control** (would the engine find a *planted* penalty?) to separate \"no effect\" from \"no "
            "power\"."
        ),

        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket; **{R['n_nameyears']}** "
            f"name-years of cash-flow / balance-sheet detail; daily closes {R['px_start']}→{R['px_end']}. "
            "**Survivor** panel — named on the Signal axis.\n"
            "- **Signal.** Scaled XFIN per fiscal year, pooled into the *calendar year* its fiscal year "
            "ends in (different names close their books on different dates).\n"
            f"- **Timing.** Treat XFIN as public **{R['report_lag']} days** after FY-end; enter the close "
            f"**one trading day later** (no same-bar fill); hold **{R['hold_days']} trading days**; drop "
            "any window that overruns the tape (no partial-year bar).\n"
            "- **Bet.** Long bottom tercile (retirers) − short top tercile (raisers), equal-weighted.\n"
            "- **Null #1 (one-sample & HAC t)** of the annual spread vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** Shuffle XFIN labels *within each year* 2000×; "
            "$p = \\Pr[\\text{shuffled }\\overline\\Lambda \\ge \\text{observed}]$.\n"
            "- **Costs.** 10 bps one-way × 2 legs per annual rebalance + 50 bps/yr borrow on the short.\n"
            "- **Positive control.** A deterministic panel with a **planted** financing penalty, averaged "
            "over **20 RNG seeds** (Welch-style): a large planted penalty must light up; the null must not."
        ),

        md("## 4 · The teardown"),
        md(
            "### 4a · The annual spread + the placebo cloud\n\n"
            "Left: the long-short spread each year (red = raisers won). Right: the real mean spread against "
            "a 2000-draw label-shuffle null — it should sit in the right tail if the sort beats noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(SIG, FWD, quantile=0.3)\n"
            "    yrs = [int(d.year) for d in ls.index]; sp = (ls['spread']*100).tolist()\n"
            "    pl = st.placebo_pvalue(SIG, FWD, quantile=0.3, n_shuffle=2000)\n"
            "    real = pl['real_mean']*100; nullm = pl['null_mean']*100; nstd = pl['null_std']*100; p = pl['p_value']\n"
            "else:\n"
            "    yrs = [y[0] for y in R['years']]; sp = [y[3] for y in R['years']]\n"
            "    real = R['placebo_real']; nullm = R['placebo_null']; nstd = R['placebo_null_std']; p = R['placebo_p']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar([str(y) for y in yrs], sp, color=[GREEN if v>0 else RED for v in sp], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('spread (%)'); a1.set_title('Annual long-short (red = raisers won)')\n"
            "for i,v in enumerate(sp): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom' if v>0 else 'top',fontsize=9)\n"
            "rng = np.random.default_rng(520); draws = rng.normal(nullm, nstd, 4000)\n"
            "a2.hist(draws, bins=50, color=GREY, alpha=.85, label='label-shuffle null')\n"
            "a2.axvline(real, c=RED, lw=2.5, label=f'real {real:+.1f}%')\n"
            "a2.set_xlabel('mean spread (%)'); a2.set_title(f'Inside the luck cloud: p = {p:.2f}'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by year:', {y: round(v,1) for y,v in zip(yrs,sp)})\n"
            "print(f'real {real:+.1f}%  null {nullm:+.1f}% (std {nstd:.1f})  placebo p = {p:.2f}')"
        ),
        md(
            f"> 💡 In plain words: 2 of 3 years are red; the real mean **{R['ls_mean']:+.1f}%/yr** sits "
            f"squarely inside the null cloud (placebo **p = {R['placebo_p']:.2f}**). Not just "
            "insignificant — on the *wrong side* of random."
        ),
        md(
            "### 4b · Significance + robustness to the cut\n\n"
            "The one-sample and HAC *t* of the spread, and whether changing the cut (median / tercile / "
            "quartile) rescues anything. A real signal should be sign-stable and clear *t* = 2; this one is "
            "sign-stable in the **wrong** direction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for q,label in ((0.5,'median'),(0.3,'tercile'),(0.25,'quartile')):\n"
            "        s = st.summary(st.long_short(SIG, FWD, quantile=q)['spread'])\n"
            "        rob.append((label, s['mean']*100, s['t']))\n"
            "    s30 = st.summary(st.long_short(SIG, FWD, quantile=0.3)['spread']); thac = s30['tstat_hac']\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2]) for r in R['robust']]; thac = R['ls_hac']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([r[0] for r in rob], [r[2] for r in rob], color=RED, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='+t=2 bar'); ax.axhline(-2, ls='--', c=GREY)\n"
            "for i,r in enumerate(rob): ax.annotate(f'{r[1]:+.1f}%',(i,r[2]),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('one-sample t'); ax.set_ylim(-2.4, 2.4)\n"
            "ax.set_title('Every cut is negative and short of the bar'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (cut, mean%, t):', [(r[0], round(r[1],2), round(r[2],2)) for r in rob])\n"
            "print(f'tercile HAC t = {thac:.2f}')"
        ),
        md(
            f"> 💡 In plain words: median **{R['robust'][0][1]:+.1f}%** (t {R['robust'][0][2]:.2f}), tercile "
            f"**{R['robust'][1][1]:+.1f}%** (t {R['robust'][1][2]:.2f}), quartile "
            f"**{R['robust'][2][1]:+.1f}%** (t {R['robust'][2][2]:.2f}). Robustly absent — the cut doesn't "
            "matter when there's nothing to cut."
        ),
        md(
            "### 4c · Costs — they only deepen a negative edge\n\n"
            "Gross vs net (10 bps/leg one-way + 50 bps/yr borrow on the short). When the gross is already "
            "below zero, costs are just insult to injury."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.summary(st.long_short(SIG, FWD, quantile=0.3)['spread'])['mean']*100\n"
            "    n = st.summary(st.long_short(SIG, FWD, quantile=0.3, cost_bps=10, borrow_ann_bps=50)['spread_net'])['mean']*100\n"
            "else:\n"
            "    g = R['ls_mean']; n = R['net_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['gross','net'], [g, n], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([g,n]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('return (%/yr)'); ax.set_title('Costs deepen the loss')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr   net {n:+.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: gross **{R['ls_mean']:+.1f}%/yr** → net **{R['net_mean']:+.1f}%/yr**. "
            "Nothing here to monetise."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a real financing penalty (big raisers underperform), "
            "averaged over 20 RNG seeds: the same sort must light up (planted) and stay flat (null). This "
            "separates \"the anomaly is absent here\" from \"our engine is broken\"."
        ),
        code(
            "sc = st.synthetic_control(seeds=tuple(range(20)))\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['planted\\npenalty','null\\n(no penalty)'], [sc['mean_t_planted'], sc['mean_t_null']],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "ax.annotate(f\"t={sc['mean_t_planted']:.1f}\",(0,sc['mean_t_planted']),ha='center',va='bottom')\n"
            "ax.annotate(f\"t={sc['mean_t_null']:.1f}\",(1,sc['mean_t_null']),ha='center',va='bottom')\n"
            "ax.set_ylabel('mean one-sample t (20 seeds)')\n"
            "ax.set_title('Engine recovers a planted penalty; stays flat under the null'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"planted mean t = {sc['mean_t_planted']:.2f} (frac t>=2: {sc['frac_t_above_2']*100:.0f}%)  \"\n"
            "      f\"null mean t = {sc['mean_t_null']:.2f}  planted spread = {sc['mean_spread_planted']*100:+.1f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: with a real planted penalty the engine reaches **t = {R['syn_t_planted']:.1f}** "
            f"({R['syn_frac2']}% of seeds clear t=2); the null sits flat at **t = {R['syn_t_null']:.2f}**. "
            "The machinery is faithful — so the flat real-tape result is a property of *this tape* "
            "(wrong-sign regime, 3 cross-sections, survivors), **not** a broken sort. This control is a "
            "power proof only; it is never cited toward the real-tape stamp."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — long-retirers/short-raisers **{R['ls_mean']:+.1f}%/yr** (one-sample "
            f"**t = {R['ls_t']:.2f}**, HAC **t = {R['ls_hac']:.2f}**), label-shuffle placebo "
            f"**p = {R['placebo_p']:.2f}**. The sign is *reversed* and indistinguishable from noise. "
            "Carries an explicit **survivorship** caveat (the dead raisers are absent, flattering the short "
            "leg — an upper bound on under-performance).\n"
            f"- **Tradability `MIRAGE`** — only **{R['n_cross']}** complete annual cross-sections survive "
            f"yfinance's 5-fiscal-year depth; net of costs the spread is **{R['net_mean']:+.1f}%/yr**. "
            "Nothing to trade.\n"
            f"- **Raisers underperform here? `BUSTED`** — in 2022–2024 the heaviest raisers (debt-funded "
            f"AI-capex) **out**-earned the retirers (2024 spread {R['years'][2][3]:+.1f}%). The synthetic "
            f"control reaches **t = {R['syn_t_planted']:.1f}** on a planted penalty, so the failure is the "
            "tape, not the engine."
        ),

        md(
            "## 6 · Could you trade it? — the honest picture\n\n"
            "There is no tradeable region to plot: the spread is negative every cut, every horizon we have, "
            "gross and net. The single operational truth is the year-by-year long vs short leg — the short "
            "(raiser) leg out-earns the long (retirer) leg in 2 of 3 years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(SIG, FWD, quantile=0.3)\n"
            "    yrs = [int(d.year) for d in ls.index]\n"
            "    longs = (ls['long_ret']*100).tolist(); shorts = (ls['short_ret']*100).tolist()\n"
            "else:\n"
            "    yrs = [y[0] for y in R['years']]; longs = [y[1] for y in R['years']]; shorts = [y[2] for y in R['years']]\n"
            "x = np.arange(len(yrs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, longs, .4, color=GREEN, label='long leg (retirers)')\n"
            "ax.bar(x+.2, shorts, .4, color=RED, label='short leg (raisers)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([str(y) for y in yrs])\n"
            "ax.set_ylabel('12-month return (%)'); ax.set_title('Raisers out-earn retirers in 2 of 3 years'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long (retirers):', [round(v,1) for v in longs]); print('short (raisers):', [round(v,1) for v in shorts])"
        ),
        md(
            "> 💡 In plain words: the red bars (raisers) top the green bars (retirers) in 2022 and 2024. "
            "Short the raisers and you were short the winners. **MIRAGE** — there is no vehicle here, only "
            "a thin, wrong-signed, survivor-biased sample."
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Point-in-time data is the fix.** CRSP/Compustat with delisting returns removes the "
            "survivorship that flatters the raiser leg and extends the sample to decades — the only way to "
            "test a factor that needs many cross-sections.\n"
            "- **Decompose the signal.** Split XFIN into its debt and equity legs (Pontiff–Woodgate is the "
            "equity-issuance leg); the equity side is the historically stronger predictor.\n"
            "- **Condition on regime.** The anomaly is about *wasteful* financing; a capex-boom dummy or a "
            "valuation interaction would separate \"raising to waste\" from \"raising to build\".\n\n"
            "*The reproducible core is offline and deterministic; the signal is the cash-flow statement's "
            "net external financing scaled by average total assets. Methods and sources: "
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
