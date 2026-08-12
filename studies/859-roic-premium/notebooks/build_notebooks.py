"""Generate the two narrative notebooks for Study 859 (Return-on-Invested-Capital Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached prices + EDGAR
events under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR ROIC + yfinance prices,
# 32 valid large-cap non-financial names, period ends 2009-06 -> 2026-05, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=32, n_names_basket=44, n_events=844,
    end_lo="2009-06-27", end_hi="2026-05-29", fp_prices="030498da471a",
    # primary calendar long-short (roic level, terciles, staleness 200)
    n_months=202, avg_n=16.5, ls_span_lo="2009-09-30", ls_span_hi="2026-06-30",
    ls_mean_bps=13.5, ls_ann=1.62, ls_t_iid=0.48, ls_t_nw=0.40, ls_sharpe=0.12,
    ls_hit=50, ls_long_bps=179.3, ls_short_bps=165.8, ls_turn=0.06,
    ls120_mean_bps=37.2, ls120_t_nw=1.12,
    chg_mean_bps=32.1, chg_t_nw=1.14, chg_sharpe=0.28, chg_n=189,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (840, 1.23, 0.96, 0.27, 0.47, 50, 0.3096),
        63: (818, 4.94, 4.58, 0.37, 0.29, 51, 0.3615),
        126: (811, 8.37, 10.05, -1.68, -0.85, 51, 0.8228),
    },
    mono63=(4.58, 5.00, 4.94),
    # era split (calendar LS), cut at 2018-01-01
    era_early_n=100, era_early_bps=-5.6, era_early_t=-0.14,
    era_late_n=102, era_late_bps=32.2, era_late_t=0.60,
    # contrast: signal -> (mean bps, NW t, sharpe, months)
    contrast={
        "roic": (13.5, 0.40, 0.12, 202),
        "roic_chg": (32.1, 1.14, 0.28, 189),
        "roe": (-65.5, -1.36, -0.50, 89),
        "gp": (-164.1, -1.59, -0.85, 60),
    },
    roic_roe_rank_corr=0.752,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (8.1, 0.98, 0.24, 0.07), (20, 100): (2.8, 0.33, 0.08, 0.02)},
    # synthetic control
    syn_null_mean=-0.38, syn_null_sd=1.43, syn_null_fire=2, syn_null_seeds=12,
    syn_planted_bps=825.4, syn_planted_t=23.63,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Adds over ROE%2FGP%3F: No](https://img.shields.io/badge/Adds_over_ROE%2FGP%3F-No-8b949e?style=flat-square)\n\n"
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

from roic_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, EV = data.load_real()
else:
    PX = EV = None
print("real cache present:", HAVE_REAL, "| events:", (0 if EV is None else len(EV)),
      "| names:", (0 if EV is None else EV['ticker'].nunique()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# ROIC — the \"quality compounder\" number. Does it actually sort stocks? 🏭\n"
            "### Strip leverage out of ROE, divide unlevered profit by all the capital deployed — "
            "and discover it forecasts nothing on the mega-caps everyone owns\n\n"
            + BADGES +
            "Return on **invested capital** is the number quality investors swear by. The pitch: "
            "return on **equity** (ROE) can be juiced by borrowing — pile on debt or buy back "
            "stock and ROE goes up without the business getting any better. **ROIC** fixes that. "
            "It takes *unlevered* operating profit (NOPAT) and divides by **all** the capital the "
            "business actually uses — debt **plus** equity **minus** cash. High, steady ROIC is "
            "supposed to be the fingerprint of a wide-moat compounder; low ROIC flags a "
            "capital-destroyer. So: buy the high-ROIC names, short the low-ROIC ones, collect the "
            "quality premium.\n\n"
            "We ran it on 32 of the biggest US companies. It sorts **nothing** — and the plainer "
            "signals it's supposed to improve on do even worse.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 44 large US non-financial names (32 yield a valid ROIC) from "
            "EDGAR, 2009→2026; a genuinely **thin** panel and a **current-survivors** basket. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does high ROIC beat low ROIC in future **returns**? | **No.** A long-short that "
            f"buys the top third and shorts the bottom third earns **+{R['ls_ann']:.1f}%/yr** on "
            f"paper — but that's statistical zero (robust *t* = **+{R['ls_t_nw']:.2f}**, the bar "
            "is 2). |\n"
            "| Is it at least consistent? | **No.** Over 2 quarters the high-ROIC names actually "
            "*under*-perform; and the spread's sign **flips** between the pre-2018 and post-2018 "
            "halves of the sample. |\n"
            f"| Does ROIC beat plain **ROE** / **gross profitability**? | **It's moot — none of "
            f"them work here.** On these mega-caps ROE and gross profitability come out "
            f"*negative*; ROIC is just the least bad. And ROIC tracks ROE **+{R['roic_roe_rank_corr']:.2f}** "
            "— basically the same bet with the leverage scrubbed off. |\n"
            "| Why? | The quality premium lives in **smaller, junkier** companies. A basket of "
            "44 giant survivors is exactly where it's weakest — the capital-destroyers you'd want "
            "to short already got bought or delisted. |\n\n"
            "> A beautiful ratio with real accounting logic — that forecasts the *business* far "
            "better than it forecasts the *stock*, and among mega-caps forecasts the stock not at "
            "all."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"ROE lies when a company is levered. ROIC doesn't — it's unlevered profit over "
            "all the capital at work. Buy high ROIC, short low ROIC, harvest quality.\"*\n\n"
            "It's a respectable idea. Profitability **is** a compensated factor in the academic "
            "record (Novy-Marx 2013; Fama-French 2015; AQR's Quality-Minus-Junk). The question is "
            "whether **this specific, unlevered, cash-adjusted** version of return-on-capital "
            "earns a spread — and whether making the ratio 'cleaner' than ROE actually buys you "
            "anything."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a tidy balance-sheet ratio sorted the returns of the most-analysed companies on "
            "earth, it would be the easiest money in the market — every value fund computes ROIC. "
            "That ubiquity is the reason to be suspicious: a number this famous, on names this "
            "scrutinised, is exactly the kind the market should already price."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} big non-financial names, ROIC = "
            "unlevered operating profit (a full trailing year) ÷ invested capital (debt + equity "
            "− cash), known only on the **filing date** of the 10-Q/10-K — never before the number "
            "is public.\n"
            "- **The return test.** Each month, rank the names on ROIC, buy the top third, short "
            "the bottom third, hold for the next month. Does the spread make money — and can we "
            "tell it apart from noise?\n"
            "- **The head-to-head.** Run the *same* test on plain ROE and on gross profitability. "
            "Does the 'cleaner' ROIC actually beat them?\n"
            "- **The mirage check.** If the spread can't beat a coin-flip relabelling of the "
            "names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The money question first: do high-ROIC stocks out-return low-ROIC stocks?** We rank "
            "the names into thirds each month and track the long-top / short-bottom portfolio."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='roic', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ann, tnw = s['ann_pct'], s['t_nw']\n"
            "    cum = (1+ls['ls']).cumprod()\n"
            "else:\n"
            "    ann, tnw = R['ls_ann'], R['ls_t_nw']\n"
            "    cum = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "if cum is not None:\n"
            "    ax.plot(cum.index, cum.values, color=AMBER, lw=1.8); ax.axhline(1.0, c='k', lw=.8)\n"
            "    ax.set_ylabel('growth of $1 (gross, long-short)')\n"
            "    ax.set_title(f'A wandering line that means nothing: +{ann:.1f}%/yr but robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: +{ann:.1f}%/yr gross, but Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"It drifts around **+{R['ls_ann']:.1f}%/yr** gross — but that's a rounding error, and "
            f"the robust *t*-statistic is **+{R['ls_t_nw']:.2f}**, nowhere near the **2** we require "
            f"to call something real. This is what *nothing* looks like: a faint wobble the "
            "statistics can't distinguish from zero.\n\n"
            "**The tell:** if ROIC really sorted returns, the thirds would form a ladder — bottom "
            "lowest, top highest. Watch what they actually do over the next quarter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bottom third\\n(low ROIC)','middle third','top third\\n(high ROIC)'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('No ladder: the ROIC thirds are basically tied (non-monotone)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by ROIC third (low->high):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            "No staircase. The middle third is the *highest*, the top and bottom are tied — and "
            "if you stretch the horizon to two quarters (in the quants notebook) the high-ROIC "
            "third actually *under*-performs. There is no monotone quality ladder here.\n\n"
            "**Now the headline question: is ROIC at least better than the plainer signals it "
            "claims to improve on?** Same long-short, run on ROIC, on plain ROE, and on gross "
            "profitability."
        ),
        code(
            "sigs = ['roic','roe','gp']; labels=['ROIC\\n(this study)','ROE\\n(200)','Gross prof.\\n(122)']\n"
            "if HAVE_REAL:\n"
            "    ts = []\n"
            "    for c in sigs:\n"
            "        lc = st.calendar_ls(PX, EV, signal_col=c, n_buckets=3, min_names=6, staleness_days=200)\n"
            "        ts.append(st.calendar_ls_stats(lc)['t_nw'])\n"
            "else:\n"
            "    ts = [R['contrast'][c][1] for c in sigs]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "cols = [GREY if t>0 else RED for t in ts]\n"
            "ax.bar(labels, ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('long-short robust t-stat'); ax.set_title('None of the quality signals clear the bar — ROE & GP go negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robust t by signal:', {l.split(chr(10))[0]: round(t,2) for l,t in zip(labels,ts)})"
        ),
        md(
            f"There it is. **None** of them work on these mega-caps — and plain ROE "
            f"({R['contrast']['roe'][1]:+.2f}) and gross profitability "
            f"({R['contrast']['gp'][1]:+.2f}) are actually *negative*. ROIC "
            f"({R['contrast']['roic'][1]:+.2f}) is merely the least bad. And because ROIC and ROE "
            f"move together (rank correlation **+{R['roic_roe_rank_corr']:.2f}**), the 'cleaner' "
            "ratio is largely the same bet with the leverage scrubbed off. Stripping leverage out "
            "of ROE changed the story not at all."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** ROIC does not sort returns here: +{R['ls_ann']:.1f}%/yr gross, "
            f"robust *t* = +{R['ls_t_nw']:.2f}, thirds non-monotone, sign flips across eras.\n"
            "- **Tradability — Mirage.** It's zero before costs; there's nothing to charge costs "
            "against.\n"
            "- **Does ROIC add anything over ROE / gross profitability? — No.** None of the "
            "quality signals certify on mega-cap survivors; ROIC is only the least bad and tracks "
            f"ROE +{R['roic_roe_rank_corr']:.2f}.\n\n"
            "> The honest one-liner: *ROIC is a great way to describe a business and a useless way "
            "to rank the biggest stocks — and making the ratio 'cleaner' than ROE buys you "
            "nothing.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where the premium actually lives.** The quality/profitability factor is strongest "
            "in **small, junky, distress-prone** names — precisely the cross-section a mega-cap "
            "**survivor** basket excludes. A live, point-in-time small-cap universe (with the "
            "delisted names still in it) is where this test would have teeth.\n"
            "- **The survivorship trap is on the short side.** Our low-ROIC short leg can only "
            "hold companies that *survived* to today; the genuine capital-destroyers were acquired "
            "or delisted, so the short leg is defanged by construction.\n"
            "- **Sibling studies:** [200-roe-quality](../../200-roe-quality/) (levered ROE), "
            "[122-gross-profitability](../../122-gross-profitability/) (GrossProfit/Assets), "
            "[242-quality-minus-junk](../../242-quality-minus-junk/) (the AQR composite), and "
            "[521-cash-based-operating-profitability](../../521-cash-based-operating-profitability/) "
            "(accruals stripped from the numerator). See [docs/references.md](../docs/references.md) "
            "for the exact dedup.\n\n"
            "*Think ROIC earns a premium on a live small-cap tape with the dead names included? "
            "Build it, show a certifiable net spread on the size you'd actually run — then we'll "
            "talk.*"
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
            "# Return-on-Invested-Capital Premium — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · the "
            "ROIC-vs-ROE-vs-gross-profitability contrast · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **ROIC = NOPAT / invested capital is a cleaner, unlevered quality gauge than "
            "ROE and sorts future returns** — is tested on a survivor mega-cap panel, alongside the "
            "sharper question: *does the unlevered refinement add anything over plain ROE "
            "([200](../../200-roe-quality/)) and gross profitability "
            "([122](../../122-gross-profitability/))?*\n\n"
            "> ⚠️ **Data note.** EDGAR `OperatingIncomeLoss` (→ TTM NOPAT), `StockholdersEquity`, "
            "`LongTermDebtNoncurrent`, `CashAndCashEquivalentsAtCarryingValue` + yfinance adjusted "
            "closes, " + str(R["n_names"]) + " valid names, ends " + R["end_lo"] + " → "
            + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on the **filing date**. "
            "Survivorship named on the Signal axis. A flat 21% tax rate is a common scalar — "
            "**invariant to the sort**. Numbers in [`docs/results.md`](../docs/results.md) (prices "
            "fingerprint `" + R["fp_prices"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `NONE` | calendar tercile long-short "
            f"**{R['ls_mean_bps']:+.1f} bps/mo** (+{R['ls_ann']:.1f}%/yr gross), one-sample "
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; best specs "
            f"(staleness-120 {R['ls120_t_nw']:+.2f}, ROIC-change {R['chg_t_nw']:+.2f}) still ≪ 2; "
            "event drift flat & sign-flipping |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; fails "
            "**before** costs |\n"
            f"| **Adds over ROE/GP?** | `NO` | ROIC NW *t* {R['contrast']['roic'][1]:+.2f} vs ROE "
            f"{R['contrast']['roe'][1]:+.2f} vs GP {R['contrast']['gp'][1]:+.2f} (both negative); "
            f"ROIC↔ROE rank corr +{R['roic_roe_rank_corr']:.2f} |\n\n"
            "> 💡 In plain words: on mega-cap survivors the quality premium is **absent**, ROIC "
            "included; the unlevered refinement is the least-bad of a losing set and largely the "
            "same bet as ROE."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "For name $i$ at fiscal quarter $q$, filed on date $F_{i,q}$:\n\n"
            "$$\\text{NOPAT}_{i,q} = \\text{OpInc}^{\\text{TTM}}_{i,q}\\,(1-\\tau), \\qquad "
            "\\text{IC}_{i,q} = \\text{Debt}_{i,q} + \\text{Equity}_{i,q} - \\text{Cash}_{i,q}, "
            "\\qquad \\text{ROIC}_{i,q} = \\frac{\\text{NOPAT}_{i,q}}{\\text{IC}_{i,q}}.$$\n\n"
            "- **H₁ (sorts returns).** A cross-sectional long-short on $\\text{ROIC}$ earns a "
            "positive forward spread.\n"
            "- **H₂ (adds value).** $\\text{ROIC}$'s spread beats plain ROE and gross "
            "profitability on the same panel.\n"
            "- **H₃ (tradable).** Any spread survives realistic long-short costs + borrow.\n\n"
            "The flat $\\tau = 0.21$ is a **common scalar**: $(1-\\tau)$ multiplies every name's "
            "NOPAT identically, so the cross-sectional ROIC *ranking* — and every long-short "
            "number — is invariant to it (only reported magnitudes move). We find **H₁ rejected** "
            f"(NW *t* = {R['ls_t_nw']:+.2f}), **H₂ rejected** (ROIC is the least-bad of a losing "
            "set), and therefore **H₃ moot**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, because "
            "balance-sheet signals are persistent and filings cluster: a calendar series of "
            "monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest work "
            "the desk's `REAL` bar is written against. The panel is thin, so we sort into "
            "**terciles** (not quintiles) and require ≥ 6 names in the cross-section. The pooled "
            "event drift + a **label-shuffle placebo** is the cross-check; the ROIC-vs-ROE-vs-GP "
            "contrast answers the value-add question on the identical panel."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) balance-sheet quarters across "
            f"{R['n_names']} valid names (of {R['n_names_basket']}), ends {R['end_lo']} → "
            f"{R['end_hi']}, each stamped with its 10-Q/10-K filing date (point-in-time).\n"
            "- **Primary.** Monthly tercile long-short on `roic`, one execution lag (rank at "
            "month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, and the "
            "tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the ROIC-**change** variant; an era "
            "split at 2018 (the TCJA statutory-rate change).\n"
            "- **Value-add.** The same long-short on `roic` vs `roe` (200) vs `gp` (122), plus the "
            "ROIC↔ROE cross-sectional rank correlation.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-premium knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh ROIC signals into terciles each month, long top / short bottom "
            "equal-weight, earn next month's return. The decisive statistic is the HAC *t* of the "
            "monthly long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='roic', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_chg = st.calendar_ls(PX, EV, signal_col='roic_chg', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s_chg = st.calendar_ls_stats(ls_chg)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo (+{s['ann_pct']:.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  ROIC-change variant: {s_chg['mean_bps']:+.1f} bps/mo, NW t = {s_chg['t_nw']:+.2f}, Sharpe {s_chg['sharpe']:.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=AMBER, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: +{R['ls_ann']:.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: XBRL coverage widens over time')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, ROIC-change NW t = {R['chg_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_mean_bps']:.0f} bps/month** is a rounding error "
            f"(~+{R['ls_ann']:.1f}%/yr gross), and the HAC *t* is **+{R['ls_t_nw']:.2f}** — the "
            f"monthly returns are indistinguishable from zero (n={R['n_months']}, avg cross-section "
            f"{R['avg_n']:.0f}). Every specification agrees: staleness-120 NW *t* = "
            f"+{R['ls120_t_nw']:.2f}, ROIC-change NW *t* = +{R['chg_t_nw']:.2f}. Right-signed by a "
            "hair, never remotely certified."
        ),
        md(
            "### 4b · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by ROIC; top-minus-bottom forward drift with a label-shuffle null. "
            "If there were a sort, the terciles would form a ladder."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in st.HORIZONS:\n"
            "        es = st.event_summary(PX, EV, horizon=h, n_buckets=3, n_draws=4000)\n"
            "        rows.append((h, es['ls_mean']*100, es['t'], es['ls_win']*100, es['p_placebo']))\n"
            "    mono = st.bucket_means(st.event_drift_frame(PX, EV, horizon=63), 3)*100\n"
            "else:\n"
            "    for h in st.HORIZONS:\n"
            "        d = R['drift'][h]; rows.append((h, d[3], d[4], d[5], d[6]))\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "hs = [r[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "a1.bar([f'{h}d' for h in hs], ts, color=GREY, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Flat / negative: no horizon clears |t|=2')\n"
            "a2.bar(['bottom','middle','top'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('No ladder (non-monotone terciles)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **flat and then negative** "
            f"(*t* from {R['drift'][21][4]:+.2f} at 1m to {R['drift'][126][4]:+.2f} at 2q — high "
            f"ROIC *under*-returns low ROIC over two quarters), the label-shuffle placebo *p* runs "
            f"{R['drift'][21][6]:.2f}–{R['drift'][126][6]:.2f}, and the 63-day terciles are "
            f"**non-monotone** ({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / "
            f"{R['mono63'][2]:+.2f}% low→high). The event study and the calendar long-short agree: "
            "no return sort."
        ),
        md(
            "### 4c · Era split — and the sign flips\n\n"
            "Split the calendar long-short at 2018 (the TCJA statutory-rate change)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = ls[ls.index < '2018-01-01']['ls'].to_numpy(); l = ls[ls.index >= '2018-01-01']['ls'].to_numpy()\n"
            "    eb, et, en = e.mean()*1e4, st.newey_west_t(e), len(e)\n"
            "    lb, lt, ln = l.mean()*1e4, st.newey_west_t(l), len(l)\n"
            "else:\n"
            "    eb, et, en = R['era_early_bps'], R['era_early_t'], R['era_early_n']\n"
            "    lb, lt, ln = R['era_late_bps'], R['era_late_t'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'2009-2017\\n(n={en})', f'2018-2026\\n(n={ln})'], [eb, lb], color=[RED, AMBER], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Sign flips across eras — the opposite of robust')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2009-2017: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2018-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"pre-2018, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) after — the "
            "tiny full-sample positive is **entirely a post-2018 artefact**, and the sign is "
            "*negative* in the first half. A signal whose sign flips across the sample is not a "
            "signal."
        ),
        md(
            "### 4d · The headline contrast — does ROIC add anything over ROE / GP?\n\n"
            "The *same* calendar long-short on `roic`, plain `roe` (200) and gross profitability "
            "`gp` (122), plus the ROIC↔ROE cross-sectional rank correlation."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.contrast(PX, EV, staleness_days=200)\n"
            "    order = ['roic','roic_chg','roe','gp']\n"
            "    ts = [c[k]['t_nw'] for k in order]; rho = c['roic_roe_rank_corr']\n"
            "else:\n"
            "    order = ['roic','roic_chg','roe','gp']\n"
            "    ts = [R['contrast'][k][1] for k in order]; rho = R['roic_roe_rank_corr']\n"
            "labels = ['ROIC','ROIC\\nchange','ROE\\n(200)','Gross prof.\\n(122)']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels, ts, color=[GREY if t>0 else RED for t in ts], width=.6)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('long-short robust t-stat')\n"
            "ax.set_title(f'None certify; ROE & GP go negative; ROIC↔ROE rank corr = +{rho:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robust t by signal:', {k: round(t,2) for k,t in zip(order,ts)}, ' ROIC-ROE rank corr', round(rho,2))"
        ),
        md(
            f"> 💡 In plain words: the headline result. **None** of the quality signals clear the "
            f"bar; plain ROE ({R['contrast']['roe'][1]:+.2f}) and gross profitability "
            f"({R['contrast']['gp'][1]:+.2f}) are outright **negative** on this panel, and ROIC "
            f"({R['contrast']['roic'][1]:+.2f}) is only the *least bad*. With a ROIC↔ROE rank "
            f"correlation of **+{R['roic_roe_rank_corr']:.2f}**, the unlevered ratio is largely the "
            "same bet as ROE. The 'cleaner gauge' refinement earns nothing certifiable here. (ROE "
            f"and GP run on thinner sub-panels — {R['contrast']['roe'][3]} and "
            f"{R['contrast']['gp'][3]} months — because fewer names report GrossProfit / positive "
            "equity every quarter; read the *signs and magnitudes*, not the literal *t*.)"
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both "
            "legs) + short borrow — though a sub-0.5 gross *t* already settles it."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for cb, bb in [(10.0,50.0),(20.0,100.0)]:\n"
            "        nt = st.calendar_ls_net(ls, cost_bps=cb, borrow_bps_ann=bb)\n"
            "        rows.append((cb, bb, nt['net_ann_pct'], nt['net_t_nw'], nt['net_sharpe']))\n"
            "else:\n"
            "    for (cb,bb),v in R['net'].items(): rows.append((cb, bb, v[1], v[2], v[3]))\n"
            "labels = [f'{int(cb)}bps +\\n{int(bb)}bps borrow' for cb,bb,_,_,_ in rows]\n"
            "anns = [r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(labels, anns, color=[AMBER, RED], width=.5)\n"
            "for i,(cb,bb,a,t,sh) in enumerate(rows): ax.annotate(f'+{a:.2f}%/yr\\n(NW t={t:+.2f})',(i,a),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net long-short (%/yr)')\n"
            "ax.set_title('Costs barely matter — but the gross edge was already zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: +{a:.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: turnover is tiny (~{R['ls_turn']:.2f}/mo — a slow quarterly "
            f"balance-sheet signal), so costs barely bite; but net NW *t* = "
            f"{R['net'][(20,100)][2]:+.2f}, Sharpe {R['net'][(20,100)][3]:.2f}. You cannot be paid "
            "for a spread that was zero to begin with. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted premium (high-ROIC names drift "
            "up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=859 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=859)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted premium (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null hovers at zero; a planted premium lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), crossing |t|=2 "
            f"{R['syn_null_fire']}/12 times (a touch rich for a HAC *t* over ~180 synthetic months, "
            f"but within noise); a planted premium reads NW *t* = {R['syn_planted_t']:.2f}. The "
            f"machinery is unbiased and powered, so the real-tape +{R['ls_t_nw']:.2f} is a genuine "
            "null, not a broken pipeline. *(Power check only — never cited in support of a "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo (+{R['ls_ann']:.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; best specs (staleness-120 "
            f"{R['ls120_t_nw']:+.2f}, ROIC-change {R['chg_t_nw']:+.2f}) ≪ 2; event drift flat and "
            "negative at 2q; era split flips sign. Right-signed by a hair, real by no measure.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: "
            f"+{R['net'][(20, 100)][1]:.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; fails before costs.\n"
            f"- **Adds over ROE / GP? `NO`** — ROIC NW *t* {R['contrast']['roic'][1]:+.2f} vs ROE "
            f"{R['contrast']['roe'][1]:+.2f} vs GP {R['contrast']['gp'][1]:+.2f} (both negative); "
            f"ROIC↔ROE rank corr +{R['roic_roe_rank_corr']:.2f}. None certify on mega-cap "
            "survivors; ROIC is the least bad and largely the same bet as ROE. The quality premium "
            "lives in the junkier cross-section a survivor basket omits."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Where the premium lives.** Profitability/quality is strongest in **small, "
            "distress-prone** names; a live point-in-time small-cap universe *with the delisted "
            "names retained* is where ROIC would get a fair test. This survivor mega-cap panel is "
            "the least favourable ground for it — and the short leg is defanged (the true "
            "capital-destroyers already left the index).\n"
            "- **Denominator honesty.** Invested capital uses *long-term* (noncurrent) debt per "
            "the study spec; adding short-term borrowings and operating leases would refine the "
            "ratio but is unlikely to resurrect a signal this flat.\n"
            "- **Dedup map:** [200-roe-quality](../../200-roe-quality/) (levered ROE), "
            "[122-gross-profitability](../../122-gross-profitability/) (GrossProfit/Assets), "
            "[242-quality-minus-junk](../../242-quality-minus-junk/) (the AQR composite), "
            "[521-cash-based-operating-profitability](../../521-cash-based-operating-profitability/) "
            "(accruals stripped from the numerator). None ranks on NOPAT ÷ invested capital.\n\n"
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
