"""Generate the two narrative notebooks for Study 657 (Larry Portfolio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
IJS/IEF/SPY/SHY tape under ../_cache/ and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance IJS/IEF/SPY/SHY,
# 2002-07-30 -> 2026-06-30, annual rebalance, 2 bps costs).
R = dict(
    start="2002-07-30", end="2026-06-30", n=6018,
    larry_cagr=5.99, larry_vol=7.1, larry_sharpe=0.609, larry_dd=-20.8,
    sixty_cagr=8.65, sixty_vol=10.2, sixty_sharpe=0.659, sixty_dd=-29.8,
    spy_cagr=11.24, spy_vol=18.8, spy_sharpe=0.544, spy_dd=-55.2,
    ijs_cagr=10.18, ijs_vol=24.0, ijs_sharpe=0.436, ijs_dd=-60.1,
    ief_cagr=3.60, ief_vol=6.8, ief_sharpe=0.315, ief_dd=-23.9,
    cagr_gap=-2.66, mean_diff_pt=-2.75, mean_diff_lo=-4.92, mean_diff_hi=-0.52,
    hac_t_ret=-2.29,
    sharpe_gap=-0.050, sharpe_lo=-0.292, sharpe_hi=0.187, larry_win_frac=32,
    corr_larry_spy=0.57, corr_sixty_spy=0.96,
    # small-value premium (IJS - SPY)
    prem_whole=0.15, prem_whole_t=0.06, prem_whole_n=6017,
    prem_early=5.08, prem_early_n=1114, prem_early_t=1.20,
    prem_late=-0.97, prem_late_n=4903, prem_late_t=-0.35,
    prem_diff_t=-1.11, split_year=2007,
    # synthetic control
    syn_null_mean=0.01, syn_null_sd=1.06, syn_null_fire=1, syn_null_cagr_gap=-1.23,
    syn_planted_t=2.79, syn_planted_recovered=6.71, syn_planted_cagr_gap=2.55,
    syn_planted_larry_sharpe=0.686, syn_planted_sixty_sharpe=0.290,
    fp_joint="8940a2d59087", fp_ijs="65bb496d17d4", fp_ief="8b77a51a60b6",
    fp_spy="85085933a99d", fp_shy="fba67966794f",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Decayed%3F: Mixed](https://img.shields.io/badge/Decayed%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from larry_portfolio import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    RETS = st.to_returns(PX)
else:
    PX = RETS = None
print("real cache present:", HAVE_REAL, "| tape rows:", (0 if PX is None else len(PX)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# A small, spicy slice of stocks — can it replace a whole 60/40? 🌾\n"
            "### The Larry Portfolio — Swedroe's bet that a little of the *best* equity "
            "factor beats a lot of the average one\n\n"
            + BADGES +
            "Larry Swedroe's pitch is simple and appealing: don't just buy \"the market\" for "
            "your equity sleeve — buy the slice of the market with the **highest expected "
            "return per unit of risk**, small-cap **value** stocks, and you'll need far less "
            "of it. Put 30% in small-cap value and 70% in safe bonds, the story goes, and "
            "you'll get returns like a 60/40 stock/bond portfolio — with much less of the "
            "stomach-churning that comes with owning that much stock.\n\n"
            "It's an elegant idea. Does it actually work?\n\n"
            "> 📓 **Plain-language layer.** Want the bootstrap CIs, the HAC *t*-stats and the "
            "synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Real daily total-return closes for four ETFs (IJS small-cap "
            f"value, IEF bonds, SPY stocks, SHY cash) from {R['start']} to {R['end']} — "
            "yfinance, cached, annually rebalanced, 2 bps costs. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is Larry calmer than 60/40? | **Yes, clearly.** Volatility "
            f"**{R['larry_vol']:.1f}%** vs {R['sixty_vol']:.1f}%, worst drawdown "
            f"**{R['larry_dd']:.1f}%** vs {R['sixty_dd']:.1f}%, and it moves with the stock "
            f"market only about half as much (correlation {R['corr_larry_spy']:.2f} vs "
            f"{R['corr_sixty_spy']:.2f}). |\n"
            f"| Does it still MATCH 60/40's return? | **No.** Larry earned "
            f"**{R['larry_cagr']:.2f}%/yr** vs 60/40's **{R['sixty_cagr']:.2f}%/yr** — a gap "
            f"of **{R['cagr_gap']:+.2f} points a year**, and that gap is not a coin-flip: it "
            "clears the desk's statistical bar for \"real\". |\n"
            f"| So is it a *worse* risk-adjusted bet? | **Not provably.** On a Sharpe-ratio "
            "basis (return per unit of risk, above cash) the two are a statistical **tie** — "
            "Larry gives back most, but not certifiably all, of its return shortfall in "
            "reduced risk. |\n"
            "| Why didn't it work as advertised? | Because the whole trick depends on "
            "small-cap value earning a real *premium* over the plain market — and on this "
            "24-year tape, it basically **didn't**. |\n\n"
            "> Half the pitch is true. The other half needed a premium that never showed up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Small-cap value stocks have the highest expected return of any major equity "
            "asset class. So instead of putting 60% of your portfolio in the broad market to "
            "get stock-like returns, put a much smaller slice — about 30% — into small-cap "
            "value, and put the rest in safe bonds. You get 60/40-like returns, but with far "
            "less total equity risk.\"* — the logic behind Larry Swedroe's \"Larry "
            "Portfolio\", built on Fama & French's finding that small AND cheap stocks "
            "carried the largest historical premium of any corner of the market."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this works, it's a genuinely useful trick — a way to dial down portfolio risk "
            "*without* giving up return, just by being smarter about *which* stocks make up "
            "your smaller equity sleeve. That would matter to anyone nervous about a 2008- or "
            "2022-style crash but unwilling to accept the return drag of a plain 60/40 (which "
            "this desk already measured in [Study 97](../../97-balancing-act/)).\n\n"
            "But it rests entirely on one assumption: that small-cap value *really does* earn "
            "meaningfully more than the broad market, consistently enough to make up for "
            "holding half as much of it. If that premium is small, or gone, the trick collapses "
            "into just... owning less stock, with less return to show for it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Build both portfolios for real.** 30% small-cap-value ETF (IJS) / 70% bond "
            "ETF (IEF), rebalanced once a year, small trading costs — pinned against a fixed "
            "60% S&P 500 (SPY) / 40% bonds (IEF), same rules.\n"
            f"- **Race them, {R['start']} → {R['end']}.** Compare return, volatility, "
            "drawdown, and the risk-adjusted (Sharpe) number.\n"
            "- **Check whether the return gap (if any) is real or luck**, with a bootstrap "
            "confidence interval.\n"
            "- **Look under the hood.** Has the small-cap-value premium itself actually shown "
            "up on this ETF, and has it faded since Swedroe first wrote about the strategy?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    larry = st.rebalanced_blend(RETS, {'IJS': 0.30, 'IEF': 0.70}, cost_bps=2.0)\n"
            "    sixty = st.rebalanced_blend(RETS, {'SPY': 0.60, 'IEF': 0.40}, cost_bps=2.0)\n"
            "    rf = RETS['SHY']\n"
            "    sl, ss = st.stats(larry, rf=rf), st.stats(sixty, rf=rf)\n"
            "    lc, sc = sl['cagr']*100, ss['cagr']*100\n"
            "    lv, sv = sl['vol']*100, ss['vol']*100\n"
            "else:\n"
            "    lc, sc = R['larry_cagr'], R['sixty_cagr']\n"
            "    lv, sv = R['larry_vol'], R['sixty_vol']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(['Larry\\n(30% IJS/70% IEF)','60/40\\n(60% SPY/40% IEF)'], [lc, sc],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([lc, sc]): a1.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('CAGR (%)'); a1.set_title('Return: Larry trails')\n"
            "a2.bar(['Larry','60/40'], [lv, sv], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([lv, sv]): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annualised volatility (%)'); a2.set_title('Risk: Larry is calmer')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CAGR: Larry {lc:.2f}%  60/40 {sc:.2f}%   |   vol: Larry {lv:.1f}%  60/40 {sv:.1f}%')"
        ),
        md(
            f"Larry earned **{R['larry_cagr']:.2f}%/yr** against 60/40's "
            f"**{R['sixty_cagr']:.2f}%/yr** — a real, **{abs(R['cagr_gap']):.2f}-point** "
            f"shortfall — while running **{R['larry_vol']:.1f}%** volatility against "
            f"**{R['sixty_vol']:.1f}%**. The risk cut is real. The return-match isn't.\n\n"
            "**Is that return gap just noise?** We check with a bootstrap — reshuffle chunks "
            "of the history thousands of times and see how often the gap could vanish by luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot = st.bootstrap_diff(larry, sixty, metric='mean', n_boot=1000, seed=657)\n"
            "    lo, hi, pt = boot['ci95'][0]*100, boot['ci95'][1]*100, boot['point']*100\n"
            "else:\n"
            "    lo, hi, pt = R['mean_diff_lo'], R['mean_diff_hi'], R['mean_diff_pt']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.errorbar([0], [pt], yerr=[[pt-lo],[hi-pt]], fmt='o', color=RED, capsize=8, ms=10)\n"
            "ax.set_xlim(-1, 1); ax.set_xticks([])\n"
            "ax.set_ylabel('Larry - 60/40 mean return (%/yr)')\n"
            "ax.set_title('The 95% confidence band sits entirely below zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'return gap {pt:+.2f}%/yr, 95% CI [{lo:+.2f}%, {hi:+.2f}%]')"
        ),
        md(
            f"The whole 95% confidence band — **[{R['mean_diff_lo']:+.2f}%, "
            f"{R['mean_diff_hi']:+.2f}%]** — sits below zero. This isn't sampling noise: "
            "Larry's return shortfall is a genuine, repeatable feature of this 24-year "
            "history, not a fluke of the particular path returns happened to take.\n\n"
            "**But does that make it a worse *risk-adjusted* bet?** Not provably — the Sharpe "
            "ratios (return per unit of risk) are a statistical tie:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sh_l, sh_s = sl['sharpe'], ss['sharpe']\n"
            "else:\n"
            "    sh_l, sh_s = R['larry_sharpe'], R['sixty_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['Larry','60/40'], [sh_l, sh_s], color=[AMBER, GREY], width=.5)\n"
            "for i,v in enumerate([sh_l, sh_s]): ax.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe ratio (excess of cash)')\n"
            "ax.set_title(f'Sharpe gap {sh_l-sh_s:+.3f} — CI [{R[\"sharpe_lo\"]:+.3f}, {R[\"sharpe_hi\"]:+.3f}] includes 0')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe: Larry {sh_l:.3f}  60/40 {sh_s:.3f}')"
        ),
        md(
            "**So why the return gap, if not from taking dumber risk?** Because the whole "
            "construction hinges on small-cap value earning MORE than its share of market "
            "risk implies — a genuine premium. Let's check whether that premium ever showed "
            "up:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    spread = st.premium_series(RETS)\n"
            "    ec = st.era_contrast(spread, f'{R[\"split_year\"]}-01-01')\n"
            "    e, l = ec['early_ann_pct'], ec['late_ann_pct']\n"
            "else:\n"
            "    e, l = R['prem_early'], R['prem_late']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar([f'{R[\"start\"][:4]}-{R[\"split_year\"]}', f'{R[\"split_year\"]}-{R[\"end\"][:4]}'],\n"
            "       [e, l], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([e, l]): ax.annotate(f'{v:+.2f}%/yr',(i,v),ha='center',\n"
            "    va='bottom' if v>0 else 'top')\n"
            "ax.set_ylabel('small-value minus S&P 500 (%/yr)')\n"
            "ax.set_title('The small-value premium (IJS - SPY): mild, then gone')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-{R[\"split_year\"]}: {e:+.2f}%/yr   post-{R[\"split_year\"]}: {l:+.2f}%/yr')"
        ),
        md(
            f"There's the answer. Over the whole 24-year tape small-cap value beat the S&P "
            f"500 by just **{R['prem_whole']:+.2f}%/yr** — statistically nothing. Split it in "
            f"two: a mild **{R['prem_early']:+.2f}%/yr** before 2007, then "
            f"**{R['prem_late']:+.2f}%/yr** since — neither half is individually reliable "
            "either, and we can't even certify the drop is a real \"decay\" rather than the "
            "premium never having been solid to begin with. Without a real premium, holding "
            "less stock just means earning less — the risk cut was real, but it wasn't free."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Real, mechanical risk reduction (lower vol, shallower "
            "drawdowns, half the correlation to a stock crash) — but the \"60/40-like "
            "returns\" promise is **not kept**: a certified, statistically real shortfall. "
            "Risk-adjusted (Sharpe), the two portfolios are a tie.\n"
            "- **Tradability — Fragile.** Cheap and easy to run (two liquid ETFs, one "
            "rebalance a year) — but there's no certified premium underneath it to make the "
            "trade-off worthwhile going forward.\n"
            "- **\"Has the premium decayed?\" — Mixed.** We can't prove it decayed — but we "
            "can't prove it was ever robustly there in the first place, on this tape."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The Fama-French factor library goes back to 1926** — far longer than any "
            "small-cap-value ETF's history. A natural follow-up redoes this test on the "
            "full academic factor data, where the historical premium is larger and more "
            "persistent, and asks how much of *that* difference is real vs an ETF-tracking "
            "artefact (fees, sampling, rebalance rules).\n"
            "- **Sibling studies:** [513-size-effect](../../513-size-effect/) and "
            "[530-book-to-market-value](../../530-book-to-market-value/) tear down the plain "
            "size and value premia individually — this study's finding (premium absent on a "
            "modern tape) lines up with both.\n\n"
            "*Think the premium is about to come back? Show a net, certifiable small-value "
            "edge on a live, forward-looking sample — then we'll talk.*"
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
            "# The Larry Portfolio — a quantitative teardown 🔬\n"
            "### The return-match HAC/bootstrap · the Sharpe-difference bootstrap · the "
            "equity-risk arithmetic · a justified 2007 era split on the premium's decay · a "
            "CAPM-neutral synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Swedroe's construction — 30% small-cap value / 70% bonds vs a 60/40 — is a clean, "
            "falsifiable claim with an academic anchor (Banz 1981; Fama-French 1992/1993) and "
            "an obvious mechanism: if small-value's expected return exceeds what its market "
            "beta implies by enough, a smaller weighted sleeve of it can match a larger "
            "weighted sleeve of the plain market. The job here is to measure whether that "
            "premium showed up, and if not, exactly where the pitch breaks.\n\n"
            "> ⚠️ **Data note.** IJS/IEF/SPY/SHY daily total-return closes "
            f"({R['start']} → {R['end']}), yfinance, cached, joint window bound by IEF/SHY's "
            "shared inception. No survivorship (broad, still-listed ETFs). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (joint fingerprint `" + R["fp_joint"] +
            "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | risk cut real (vol {R['larry_vol']:.1f}% vs "
            f"{R['sixty_vol']:.1f}%, maxDD {R['larry_dd']:.1f}% vs {R['sixty_dd']:.1f}%) · "
            f"return-match **fails**: CAGR gap **{R['cagr_gap']:+.2f} pts/yr**, HAC "
            f"**t = {R['hac_t_ret']:.2f}**, bootstrap CI **[{R['mean_diff_lo']:+.2f}%, "
            f"{R['mean_diff_hi']:+.2f}%]** excludes 0 · Sharpe gap **tied** (CI "
            f"[{R['sharpe_lo']:+.3f}, {R['sharpe_hi']:+.3f}] includes 0) |\n"
            f"| **Tradability** | `FRAGILE` | cheap (2 bps, annual rebalance) but the "
            f"load-bearing premium is **t = {R['prem_whole_t']:.2f}** whole-sample — "
            "statistically absent |\n"
            f"| **Decayed?** | `MIXED` | era-diff **t = {R['prem_diff_t']:.2f}**; neither era "
            "individually significant |\n\n"
            "> 💡 In plain words: the risk reduction is real physics of the weights; the "
            "return-match needed a premium that never certifiably showed up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{SV}_t$, $r^{M}_t$, $r^{B}_t$ be daily total returns on small-cap value "
            "(IJS), the market (SPY) and bonds (IEF). Larry's construction: "
            "$w = (0.30, 0, 0.70)$ over $(SV, M, B)$; the benchmark: $w' = (0, 0.60, 0.40)$. "
            "The claim decomposes into three testable pieces:\n\n"
            "- **H₁ (return match).** $E[r_{\\text{Larry}}] \\approx E[r_{60/40}]$ — the "
            "portfolios earn statistically indistinguishable returns.\n"
            "- **H₂ (risk cut).** $\\sigma_{\\text{Larry}} \\ll \\sigma_{60/40}$ and "
            "$\\text{corr}(\\text{Larry}, SPY) \\ll \\text{corr}(60/40, SPY)$ — pure "
            "arithmetic on the weights, always true by construction whenever $SV$'s beta to "
            "$M$ isn't dramatically above $60/30 = 2$.\n"
            "- **H₃ (the mechanism).** $E[r^{SV}_t] - \\beta_{SV} \\cdot E[r^M_t] > 0$ — "
            "small-value earns a genuine premium *beyond* what its market beta implies. This "
            "is the assumption H₁ needs to be simultaneously true with H₂.\n\n"
            "We find **H₂ trivially true**, **H₁ false and certified** (the return gap is "
            "real, not luck), and **H₃ statistically absent** on this tape — which is exactly "
            "*why* H₁ fails: without a premium, a 30% sleeve of anything can't match a 60% "
            "sleeve of the market it's a levered bet on."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two genuinely different statistical questions get two different tools:\n\n"
            "- **The return gap** (H₁) is a **mean-difference** question — Newey-West "
            "(1987) HAC *t* on the daily (Larry − 60/40) series, plus a **circular block "
            "bootstrap** (21-day blocks — long enough to span a trading month of "
            "autocorrelation — 2,000 resamples) 95% CI on the mean.\n"
            "- **The Sharpe gap** is a **ratio-difference** question — a HAC *t* on the same "
            "difference series would just answer H₁ again (a shared cash leg cancels out of "
            "a two-arm difference), so the Sharpe comparison needs its own bootstrap: "
            "resample blocks, recompute *each arm's own* annualised Sharpe per resample, "
            "difference the ratios.\n"
            "- **H₃ (the premium)** gets the same HAC-*t*-on-a-spread machinery as sibling "
            "study 637's decision-day split, plus an **externally justified** era split at "
            f"{R['split_year']}-01-01 (AQR's and Swedroe's own documented post-GFC value "
            "drawdown), tested as a Welch *t* of the era **difference**, never eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** IJS/IEF/SPY/SHY daily total-return closes {R['start']} → "
            f"{R['end']} ({R['n']:,} rows) — IEF/SHY's shared inception is the binding "
            "constraint (same window as sibling study 97-balancing-act).\n"
            "- **Construction.** Annual rebalance (first trading day of the year), 2 bps "
            "one-way cost on turnover, no execution lag needed (fixed-weight calendar "
            "rebalancing carries no signal).\n"
            "- **Headline.** CAGR/vol/Sharpe(xs-of-cash SHY)/maxDD for Larry, 60/40, and each "
            "leg alone.\n"
            "- **Return-match test.** HAC *t* + 21-day-block bootstrap CI on the mean-return "
            "difference.\n"
            "- **Sharpe test.** A *separate* block-bootstrap on the Sharpe-ratio difference.\n"
            "- **Risk claim.** Vol, maxDD, correlation-to-SPY — reported directly, no "
            "inferential test needed.\n"
            f"- **Decay test.** HAC *t* on the whole-sample (IJS − SPY) spread, era split at "
            f"{R['split_year']}-01-01, Welch *t* of the era difference.\n"
            "- **Control.** CAPM-neutral (β=1) synthetic 3-asset world, planted premium knob; "
            "the premium-detection statistic must not fire across 20 null seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline race\n\n"
            "CAGR / vol / Sharpe(xs) / max drawdown for every arm."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rf = RETS['SHY']\n"
            "    larry = st.rebalanced_blend(RETS, {'IJS': 0.30, 'IEF': 0.70}, cost_bps=2.0)\n"
            "    sixty = st.rebalanced_blend(RETS, {'SPY': 0.60, 'IEF': 0.40}, cost_bps=2.0)\n"
            "    rows = {\n"
            "        'Larry (30/70)': st.stats(larry, rf=rf),\n"
            "        '60/40': st.stats(sixty, rf=rf),\n"
            "        '100% SPY': st.stats(st.single_asset(RETS, 'SPY'), rf=rf),\n"
            "        '100% IJS': st.stats(st.single_asset(RETS, 'IJS'), rf=rf),\n"
            "        '100% IEF': st.stats(st.single_asset(RETS, 'IEF'), rf=rf),\n"
            "    }\n"
            "    for name, d in rows.items():\n"
            "        print(f\"{name:<16} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  \"\n"
            "              f\"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%\")\n"
            "    labels = list(rows.keys())\n"
            "    cagrs = [d['cagr']*100 for d in rows.values()]\n"
            "    vols = [d['vol']*100 for d in rows.values()]\n"
            "else:\n"
            "    labels = ['Larry (30/70)', '60/40', '100% SPY', '100% IJS', '100% IEF']\n"
            "    cagrs = [R['larry_cagr'], R['sixty_cagr'], R['spy_cagr'], R['ijs_cagr'], R['ief_cagr']]\n"
            "    vols = [R['larry_vol'], R['sixty_vol'], R['spy_vol'], R['ijs_vol'], R['ief_vol']]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "x = np.arange(len(labels))\n"
            "ax.scatter(vols, cagrs, s=140,\n"
            "           color=[AMBER, GREY, RED, RED, GREEN])\n"
            "for i, lab in enumerate(labels):\n"
            "    ax.annotate(lab, (vols[i], cagrs[i]), textcoords='offset points',\n"
            "                xytext=(8, 4), fontsize=9)\n"
            "ax.set_xlabel('annualised volatility (%)'); ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('Risk vs return: Larry sits well left of 60/40, but also below it')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: Larry ({R['larry_vol']:.1f}% vol, {R['larry_cagr']:.2f}% "
            f"CAGR) sits well LEFT of 60/40 ({R['sixty_vol']:.1f}%, {R['sixty_cagr']:.2f}%) on "
            "the risk axis — genuinely calmer — but also below it on the return axis. The "
            "\"free lunch\" promise (same return, less risk) is half right."
        ),
        md(
            "### 4b · Does Larry match 60/40's RETURN?\n\n"
            "HAC *t* on the daily difference, plus a circular block bootstrap (21-day "
            "blocks — long enough to span a trading month's autocorrelation)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    diff = (larry - sixty).dropna()\n"
            "    t_ret = st.hac_tstat(diff.to_numpy())\n"
            "    boot = st.bootstrap_diff(larry, sixty, metric='mean', n_boot=1000, seed=657)\n"
            "    pt, lo, hi = boot['point']*100, boot['ci95'][0]*100, boot['ci95'][1]*100\n"
            "else:\n"
            "    t_ret = R['hac_t_ret']\n"
            "    pt, lo, hi = R['mean_diff_pt'], R['mean_diff_lo'], R['mean_diff_hi']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhspan(lo, hi, color=RED, alpha=.18)\n"
            "ax.plot([0], [pt], 'o', color=RED, ms=12)\n"
            "ax.set_xlim(-1, 1); ax.set_xticks([])\n"
            "ax.set_ylabel('mean return diff, Larry - 60/40 (%/yr)')\n"
            "ax.set_title(f'HAC t = {t_ret:+.2f}  |  95% CI [{lo:+.2f}%, {hi:+.2f}%]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'point {pt:+.2f}%/yr  CI [{lo:+.2f}%, {hi:+.2f}%]  HAC t = {t_ret:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the return gap clears the bar two ways — HAC "
            f"*t* = **{R['hac_t_ret']:.2f}** and a bootstrap CI **[{R['mean_diff_lo']:+.2f}%, "
            f"{R['mean_diff_hi']:+.2f}%]** that never touches zero. **H₁ rejected**: the "
            "return-match claim does not survive."
        ),
        md(
            "### 4c · Is the risk-adjusted (Sharpe) edge real?\n\n"
            "A *different* test — a bootstrap on the Sharpe-ratio difference itself (a HAC *t* "
            "on the raw difference series would just re-answer 4b, since the shared cash leg "
            "cancels out of a two-arm subtraction)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot_sh = st.bootstrap_diff(larry, sixty, rf=rf, metric='sharpe', n_boot=1000, seed=657)\n"
            "    pt_s, lo_s, hi_s = boot_sh['point'], boot_sh['ci95'][0], boot_sh['ci95'][1]\n"
            "    win = boot_sh['frac_a_wins']*100\n"
            "else:\n"
            "    pt_s, lo_s, hi_s = R['sharpe_gap'], R['sharpe_lo'], R['sharpe_hi']\n"
            "    win = R['larry_win_frac']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhspan(lo_s, hi_s, color=GREY, alpha=.25)\n"
            "ax.plot([0], [pt_s], 'o', color=GREY, ms=12)\n"
            "ax.set_xlim(-1, 1); ax.set_xticks([])\n"
            "ax.set_ylabel('Sharpe diff, Larry - 60/40')\n"
            "ax.set_title(f'CI [{lo_s:+.3f}, {hi_s:+.3f}] includes 0  |  Larry wins {win:.0f}% of resamples')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe diff point {pt_s:+.3f}  CI [{lo_s:+.3f}, {hi_s:+.3f}]  Larry win rate {win:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: unlike the return gap, the Sharpe gap's CI "
            f"(**[{R['sharpe_lo']:+.3f}, {R['sharpe_hi']:+.3f}]**) straddles zero — Larry buys "
            "back most of its shortfall in reduced variance. It's not certifiably *worse* "
            "risk-adjusted, just certifiably lower-returning in absolute terms."
        ),
        md(
            "### 4d · The equity-risk claim — arithmetic, not inference\n\n"
            "Vol, drawdown, correlation to SPY: these follow mechanically from running 30% "
            "vs 60% equity weight and need no statistical test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy100 = st.single_asset(RETS, 'SPY')\n"
            "    corr_l, corr_s = larry.corr(spy100), sixty.corr(spy100)\n"
            "else:\n"
            "    corr_l, corr_s = R['corr_larry_spy'], R['corr_sixty_spy']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['Larry','60/40'], [corr_l, corr_s], color=[AMBER, GREY], width=.5)\n"
            "for i,v in enumerate([corr_l, corr_s]): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 1.05); ax.set_ylabel('daily correlation to SPY')\n"
            "ax.set_title('Larry moves with a stock crash about half as much')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'corr to SPY: Larry {corr_l:.2f}  60/40 {corr_s:.2f}')"
        ),
        md(
            f"> 💡 In plain words: correlation to SPY drops from **{R['corr_sixty_spy']:.2f}** "
            f"to **{R['corr_larry_spy']:.2f}** — half the equity crash exposure, exactly what "
            "running 30% instead of 60% equity weight predicts. **H₂ confirmed**, trivially."
        ),
        md(
            "### 4e · The mechanism — has the small-value premium itself shown up?\n\n"
            "Whole-sample HAC *t* on the daily (IJS − SPY) spread, then a justified era split "
            f"at **{R['split_year']}-01-01** (AQR's and Swedroe's own documented value-drawdown "
            "turning point), tested as a difference — not eyeballed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spread = st.premium_series(RETS)\n"
            "    ps = st.premium_stats(spread)\n"
            "    ec = st.era_contrast(spread, f'{R[\"split_year\"]}-01-01')\n"
            "    whole_t, e, l = ps['hac_t'], ec['early_ann_pct'], ec['late_ann_pct']\n"
            "    et, lt, dt = ec['hac_t_early'], ec['hac_t_late'], ec['welch_t_diff']\n"
            "else:\n"
            "    whole_t, e, l = R['prem_whole_t'], R['prem_early'], R['prem_late']\n"
            "    et, lt, dt = R['prem_early_t'], R['prem_late_t'], R['prem_diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar([f'{R[\"start\"][:4]}-{R[\"split_year\"]}\\n(t={et:+.2f})',\n"
            "        f'{R[\"split_year\"]}-{R[\"end\"][:4]}\\n(t={lt:+.2f})'],\n"
            "       [e, l], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([e, l]): ax.annotate(f'{v:+.2f}%/yr',(i,v),ha='center',\n"
            "    va='bottom' if v>0 else 'top')\n"
            "ax.set_ylabel('IJS - SPY (%/yr)')\n"
            "ax.set_title(f'Whole-sample t={whole_t:+.2f}  |  era-diff t={dt:+.2f} (not certified)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'whole-sample HAC t = {whole_t:+.2f}  |  early {e:+.2f}% (t={et:+.2f})  '\n"
            "      f'late {l:+.2f}% (t={lt:+.2f})  |  era-diff Welch t = {dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the premium is statistically nothing over the whole tape "
            f"(*t* = **{R['prem_whole_t']:.2f}**), and neither era is individually significant "
            f"either (*t* = {R['prem_early_t']:.2f} pre-2007, *t* = {R['prem_late_t']:.2f} "
            f"since). The era-difference *t* = **{R['prem_diff_t']:.2f}** doesn't certify a "
            "decay — the honest read is that this ETF-tape premium may never have been solid "
            "enough to certify in the first place. **H₃ not supported.**"
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "A CAPM-neutral (β=1) synthetic 3-asset world with a TUNABLE planted small-value "
            "premium. The pass/fail target is the premium-detection statistic itself — the "
            "same HAC-*t*-on-a-spread primitive used above on the real IJS-SPY tape."
        ),
        code(
            "null_ts, null_gaps = [], []\n"
            "for s_ in range(20):\n"
            "    panel = data.synthetic_world(premium=0.0, seed=657 + s_)\n"
            "    d = st.synthetic_detect(panel)\n"
            "    null_ts.append(d['premium_hac_t']); null_gaps.append(d['cagr_gap'])\n"
            "null_ts = np.asarray(null_ts); null_gaps = np.asarray(null_gaps)\n"
            "panel = data.synthetic_world(premium=0.05, seed=657)\n"
            "planted = st.synthetic_detect(panel)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (premium=0), 20 seeds')\n"
            "ax.scatter([1], [planted['premium_hac_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted premium = +5.0%/yr')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('premium HAC t (SV - MKT spread)')\n"
            "ax.set_title('Control: the null rarely fires; a planted premium lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  '\n"
            "      f'mean CAGR gap at premium=0: {null_gaps.mean()*100:+.2f} pts/yr')\n"
            "print(f'planted t = {planted[\"premium_hac_t\"]:+.2f}  '\n"
            "      f'CAGR gap flips to {planted[\"cagr_gap\"]*100:+.2f} pts/yr')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"*t* = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires in only "
            f"{R['syn_null_fire']}/20 — well-calibrated. Note that even with ZERO premium, "
            f"the Larry-vs-60/40 CAGR gap still averages **{R['syn_null_cagr_gap']:+.2f} "
            "pts/yr** across those seeds — the mechanical cost of running less equity beta, "
            "the exact same fact measured on the real tape in 4a-4b. Once a genuine premium "
            f"is planted (+5.0%/yr), the detector recovers it cleanly "
            f"(*t* = {R['syn_planted_t']:.2f}) and the portfolio gap flips positive "
            f"(**{R['syn_planted_cagr_gap']:+.2f} pts/yr**, Larry Sharpe "
            f"{R['syn_planted_larry_sharpe']:.3f} vs 60/40 {R['syn_planted_sixty_sharpe']:.3f}) "
            "— the machinery *can* find a real premium and correctly translate it into the "
            "headline race; it simply doesn't find one on the real 2002-2026 tape. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — risk cut real and mechanical (vol {R['larry_vol']:.1f}% "
            f"vs {R['sixty_vol']:.1f}%, maxDD {R['larry_dd']:.1f}% vs {R['sixty_dd']:.1f}%, "
            f"corr-to-SPY {R['corr_larry_spy']:.2f} vs {R['corr_sixty_spy']:.2f}) · "
            f"return-match **fails**, certified (CAGR gap **{R['cagr_gap']:+.2f} pts/yr**, "
            f"HAC *t* = **{R['hac_t_ret']:.2f}**, bootstrap CI "
            f"**[{R['mean_diff_lo']:+.2f}%, {R['mean_diff_hi']:+.2f}%]** excludes 0) · "
            f"Sharpe gap is a **statistical tie** (CI [{R['sharpe_lo']:+.3f}, "
            f"{R['sharpe_hi']:+.3f}] includes 0).\n"
            "- **Tradability `FRAGILE`** — trivially cheap to run (2 liquid ETFs, annual "
            f"rebalance, 2 bps) but the load-bearing premium is statistically absent "
            f"(*t* = {R['prem_whole_t']:.2f} whole-sample).\n"
            f"- **\"Decayed? `MIXED`\"** — era-difference *t* = {R['prem_diff_t']:.2f} doesn't "
            "certify decay; neither era is individually significant either. Consistent with "
            "siblings 513-size-effect and 530-book-to-market-value finding the underlying "
            "premia absent on modern tapes."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The academic factor library is longer and different.** Fama-French's own "
            "SMB/HML series go back to 1926 and use a much broader, more granular universe "
            "than a single ETF proxy — a natural follow-up is redoing this exact test on "
            "that data, separating the ETF-implementation question (fees, sampling, "
            "reconstitution rules) from the factor's own historical existence.\n"
            "- **Live capacity and concentration risk are unexplored here.** Small-cap value "
            "is a much smaller, less liquid corner of the market than the S&P 500 — capacity "
            "and factor-crowding effects on a genuinely large allocation are a separate "
            "question this study doesn't size.\n"
            "- **Dedup map:** [513-size-effect](../../513-size-effect/) (size premium, "
            "stock basket), [530-book-to-market-value](../../530-book-to-market-value/) "
            "(value premium, fundamentals), [655-ivy-portfolio](../../655-ivy-portfolio/) "
            "(5-asset diversification, not factor concentration), "
            "[68-all-weather](../../68-all-weather/) (risk parity), "
            "[97-balancing-act](../../97-balancing-act/) (the plain 60/40 this study races "
            "against, identical construction).\n\n"
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
