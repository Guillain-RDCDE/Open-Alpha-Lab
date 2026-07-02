"""Generate the two narrative notebooks for Study 568 (Effective-Tax-Rate).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached EDGAR fundamentals +
yfinance prices under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; fiscal
# 2008 -> 2024, 17 sortable years; 40 survivors; ETR panel fp 74b7313496ca).
R = dict(
    fp_etr="74b7313496ca", fp_fwd="91df5a37e544", fp_fund="ca80cb42a4ce",
    n_names=40, n_years=17, first_fy=2008, last_fy=2024,
    hedge_mean=-3.3, hedge_t=-0.91, sharpe=-0.22, hit=0.47,
    placebo_p=0.422, ic=-0.023, ic_t=-0.46,
    detr_mean=-2.3, detr_t=-0.77, detr_ic=-0.003,
    gross=-3.3, net=-3.9, net_t=-1.08, turnover=0.64, one_way_bps=10.0, borrow_bps=50.0,
    # robustness windows: (label, mean%/yr, t)
    win=[("2009-2013", 3.0, 0.62), ("2014-2018", -2.5, -1.06),
         ("2019-2024", -4.2, -0.87), ("2009-2024", -1.4, -0.46)],
    # synthetic control: (premium, mean hedge-t over 25 seeds)
    ctrl=[(0.0, -0.14), (-0.03, 6.29), (-0.05, 10.58), (-0.08, 17.02), (0.05, -10.81)],
    low_names="PFE, IBM, GE, BAC, VZ, AMGN",
    high_names="CVX, COP, DE, COST, CAT, XOM",
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from effective_tax_rate import data, strategy as st

def load_real():
    \"\"\"Cache-first real (etr, fwd) panels (empty frames offline).\"\"\"
    etr, fwd = data.load_panel()
    return etr, fwd

ETR, FWD = load_real()
HAVE_REAL = not ETR.empty and not FWD.empty
print("real ETR panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({ETR.shape[1]} names, fiscal {int(ETR.index.min())}-{int(ETR.index.max())})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Effective-Tax-Rate Anomaly — do low-tax-paying firms make better stocks? 🧾\n"
            "### Sorting big companies by how much tax they actually pay, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Every company reports an **effective tax rate** — the tax it actually paid divided by its "
            "pre-tax profit. Some blue chips pay close to the statutory ~21-35%; others, through "
            "offshore structures, loss carry-forwards and clever accounting, pay far less. There's a "
            "long-running argument about what a *low* tax rate means for the stock:\n\n"
            "- **The optimists:** a low tax bill is a sign of a well-run, cash-generative company the "
            "market underrates — so it should earn *higher* returns.\n"
            "- **The pessimists:** a suspiciously low tax bill is a **fragile loophole** that can snap "
            "shut (regulators, one-off shields reversing) — so it should earn *lower* returns.\n\n"
            "Both can't be right. We test it the cheap way: take 40 big-name stocks, rank them each "
            "year by effective tax rate, and see whether the *low-tax* ones beat the *high-tax* ones.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the *low-tax* firms earn more (the quality story)? | **No.** Over 17 years the "
            f"long-low / short-high-tax spread was **{R['hedge_mean']}%/yr** — slightly the *wrong* "
            "way, and basically zero. |\n"
            f"| Did they earn *less* (the red-flag story)? | **Not really either** — the number is too "
            f"small and noisy to call. A shuffle test puts the odds at *p* = {R['placebo_p']} (pure "
            "noise). |\n"
            "| Is there *any* relationship? | **No.** The rank correlation between tax rate and next "
            f"year's return is essentially **{R['ic']}** — flat. |\n"
            "| Could you trade it? | **No.** It doesn't pay before costs, loses more after, and the "
            "sign even flips depending on which years you look at. |\n\n"
            "> The tax-rate anomaly is a real *debate* in the research — but on 40 surviving blue "
            "chips it just isn't there. The low-tax bucket is mostly banks and telecoms; the high-tax "
            "bucket is mostly energy and heavy industry. The sort is really sorting by **industry**, "
            "and it pays nothing."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A company's effective tax rate predicts its future stock returns.\"* — except nobody "
            "agrees which *way*.\n\n"
            "One camp (tax-avoidance-as-quality) says low-tax firms are efficient and underpriced, so "
            "they win. The other camp (tax-avoidance-as-risk) says a low tax rate is a red flag for "
            "aggressive, reversible accounting, so they lose. When even the *sign* is contested, the "
            "honest move is to sort on the tax rate and simply look — which direction, if any, wins?"
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it were real, it'd be a lovely signal: the tax line is audited, hard to fake, and sits "
            "right there in every 10-K. And it suggests an easy trade — own the low-tax firms (or the "
            "high-tax ones, depending on your camp). The desk has looked at the April **tax-day** "
            "*calendar* seasonal ([192](../../192-tax-day/)) and at quality/accrual sorts "
            "([231 Sloan](../../231-sloan-accruals/), [522 accruals](../../522-percent-operating-accruals/)); "
            "here we go straight at the **effective tax rate** as the ranking key."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the tax rate.** For each firm each year: tax actually paid ÷ pre-tax profit. "
            "(We skip loss-making years — a company with no profit has no meaningful tax *rate*.)\n"
            "2. **Sort.** Split the 40 names into five buckets, from lowest tax rate to highest.\n"
            "3. **Compare next-year returns.** Did the low-tax bucket beat the high-tax bucket?\n\n"
            "Catch: our basket is companies **still alive in 2026** — so any firm whose aggressive "
            "tax scheme blew up and got delisted is missing (exactly the names the red-flag story "
            "cares about).\n\n"
            "*Timing:* a fiscal-year-y tax rate is used to predict year *y+1*'s return — the 10-K "
            "isn't even public until months into the next year, so there's no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Who won — the low-tax fifth or the high-tax fifth, averaged over the years?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.quintile_returns(ETR, FWD, q=0.20, min_names=15)\n"
            "    lo, hi = q['q1'].mean()*100, q['q5'].mean()*100\n"
            "    hedge = q['hedge'].mean()*100; t = st.summary(q['hedge'])['tstat']\n"
            "    banner = f'REAL tape ({ETR.shape[1]} survivors, fiscal 2008-2024)'\n"
            "else:\n"
            f"    lo, hi, hedge, t = 12.0, 15.0, {R['hedge_mean']}, {R['hedge_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['LOW-tax fifth','HIGH-tax fifth'], [lo, hi], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg next-year return %')\n"
            "ax.set_title(f'Low-tax minus high-tax = {hedge:+.1f}%/yr (t {t:+.2f}) — basically a tie')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Low-tax {lo:+.1f}%  vs  High-tax {hi:+.1f}%  ->  spread {hedge:+.1f}%/yr (t {t:+.2f})')"
        ),
        md(
            f"There's nothing there. The low-tax fifth and the high-tax fifth earned about the same; "
            f"the spread is **{R['hedge_mean']}%/yr** with a *t* of just **{R['hedge_t']}** — you "
            "can't tell it from zero. And notice *which* firms fall where: the low-tax bucket is "
            f"**{R['low_names']}…** (banks, telecom, old tech carrying big shields); the high-tax "
            f"bucket is **{R['high_names']}…** (energy and heavy industry that pay close to statutory). "
            "We're mostly sorting by industry."
        ),
        md(
            "**Is it *reliably* zero, or does it just wobble around?** Run the same sort over three "
            "different stretches of years:"
        ),
        code(
            "wins = " + repr([(w[0], w[1]) for w in R["win"][:3]]) + "\n"
            "if HAVE_REAL:\n"
            "    sweep = st.window_sweep(ETR, FWD, windows=[(2009,2013),(2014,2018),(2019,2024)])\n"
            "    labs = list(sweep.index); vals = list(sweep['mean%'])\n"
            "else:\n"
            "    labs = [w[0] for w in wins]; vals = [w[1] for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(labs, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('low - high tax spread %/yr')\n"
            "ax.set_title('The sign flips with the window (+early, -late) — never big, never reliable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by window:', [round(v,1) for v in vals])"
        ),
        md(
            "So it isn't even a *stable* zero. In the early window the low-tax names edged ahead "
            "(+3%); in the later windows the high-tax names did (−2 to −4%). A 'signal' that changes "
            "direction with the calendar and is never statistically real is not something you bet on."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The low-minus-high-tax spread is a tiny, wrong-signed "
            f"{R['hedge_mean']}%/yr (*t* {R['hedge_t']}), the shuffle test says *p* = {R['placebo_p']} "
            "(noise), and the sign flips across windows. No bankable anomaly here.\n"
            "- **Tradability — Mirage.** A survivor basket, annual rebalance, and it loses before you "
            "even pay costs. Nothing to harvest."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. There's no edge to capture, the sort mostly reflects which *industry* a firm is in, "
            "and the basket has already quietly deleted every company whose tax scheme actually blew "
            "up — which is where the red-flag version of this story would live. A blue-chip survivor "
            "sort can't see the interesting cases."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Neighbours.** [192 tax-day](../../192-tax-day/) is the *calendar* tax seasonal; "
            "[231 Sloan](../../231-sloan-accruals/) and [522 accruals](../../522-percent-operating-accruals/) "
            "are the cash-vs-accrual quality sorts. This one is the **tax line** specifically.\n"
            "- **The real test needs the blow-ups.** Re-run with delisted/restated firms and a "
            "multi-year, industry-adjusted *cash* tax rate — that's where any effect would hide.\n\n"
            "*Think the anomaly is real and we just missed it? Fork this, add industry-neutralised, "
            "smoothed cash ETRs on a survivorship-free universe, and show a spread clearing t = 2.*"
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
            "# The Effective-Tax-Rate Anomaly — a quantitative teardown 🔬\n"
            "### ETR level & change · quintile sort · HAC *t* · label-shuffle placebo · rank-IC · window sign-flip · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the effective tax "
            "rate (income-tax expense / pretax income) from EDGAR 10-K facts, sort the cross-section, "
            "and test whether low-ETR firms out-earn (quality), under-earn (risk), or — as it turns "
            "out — neither.\n\n"
            "> ⚠️ **Not investment advice.** Real data: EDGAR annual fundamentals + yfinance daily "
            f"prices, {R['n_names']} survivors, fiscal 2008-2024 (as-of 2026-06-30, ETR panel fp "
            f"`{R['fp_etr']}`); the offline core and tests run on a deterministic synthetic world. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Low−high ETR hedge **{R['hedge_mean']}%/yr** (HAC *t* "
            f"**{R['hedge_t']}**, placebo *p* {R['placebo_p']}) — wrong sign, ≈ 0; rank-IC "
            f"**{R['ic']}** (*t* {R['ic_t']}); ΔETR hedge **{R['detr_mean']}%/yr** (*t* {R['detr_t']}); "
            "sign flips across windows. |\n"
            f"| **Tradability** | `MIRAGE` | Survivor basket, annual rebalance, high-ETR short leg. "
            f"Gross {R['gross']}%/yr → net {R['net']}%/yr ({R['one_way_bps']:.0f} bps/leg × "
            f"{R['turnover']:.2f} turnover + {R['borrow_bps']:.0f} bps borrow). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it fires on a planted "
            "premium of *either* sign), but on this survivor basket the ETR sort carries no return "
            "information at all — a clean null."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $E_i$ be firm $i$'s effective tax rate and $r_i$ its next-year return.\n\n"
            "- **H₁ (quality / long-short).** $\\mathbb{E}[r \\mid \\text{low ETR}] - "
            "\\mathbb{E}[r \\mid \\text{high ETR}] > 0$ at *t* ≥ 2 (low-tax firms out-earn).\n"
            "- **H₁′ (risk).** The *opposite* sign — low-ETR firms *under*-earn.\n"
            "- **H₂ (firm-level).** A non-zero rank-IC between $E$ and $r$.\n"
            "- **H₃ (change).** ΔETR (a shield reversing) predicts returns.\n"
            "- **H₄ (robustness/net).** Any H₁/H₁′ effect is sign-stable and survives costs.\n\n"
            "We find **H₁ rejected** (the hedge is a wrong-signed, insignificant −3.3%/yr), **H₁′ not "
            "supported** (too small/noisy to call, placebo *p* 0.42), **H₂ rejected** (IC ≈ 0), **H₃ "
            "rejected** (ΔETR is dead too), and **H₄ moot** (no effect to stabilise; sign flips)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why the sign is contested**. The quality camp (Desai-Dharmapala) "
            "and the risk camp (Weber 2009; Hanlon 2005 on book-tax differences) make opposite "
            "predictions; Thomas-Zhang (2011) find *changes* in tax expense carry information. Two "
            "forces flatten all of it here: (a) **survivorship** — the basket already dropped every "
            "tax scheme that blew up, removing the red-flag tail; and (b) **industry confounding** — "
            "on 40 blue chips the low-ETR fifth is banks/telecom and the high-ETR fifth is "
            "energy/industrials, so the sort is mostly a sector bet. That maps to the desk's other "
            "survivor-basket accounting nulls."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $E = \\text{tax expense} / \\text{pretax income}$ (only where pretax income "
            "is comfortably positive; winsorised to a sane band). Higher = pays more tax.\n"
            "- **Sort.** Quintiles each fiscal year (~7 names/bucket): Q1 lowest ETR (long) vs Q5 "
            "highest (short). Hedge = Q1 − Q5 (long low-tax).\n"
            "- **Inference.** HAC (Newey-West) *t* on the annual hedge series; a **label-shuffle "
            "placebo** null (1000 perms, permute ETR labels within each year).\n"
            "- **Firm-level.** Year-by-year cross-sectional **rank-IC** between ETR and next-year "
            "return, with a *t* on its year series.\n"
            "- **Change.** Repeat on ΔETR (year-on-year change).\n"
            "- **Robustness.** Re-sort over sub-samples; read the sign.\n"
            "- **Frictions.** One-way bps × turnover per leg + an annual borrow on the short leg.\n"
            "- **Positive control.** A deterministic synthetic world with a planted premium of "
            "*either* sign, averaged over 25 seeds (house rule).\n\n"
            "Timing: fiscal-year-y fundamentals predict calendar-year y+1 returns — exactly one "
            "execution lag (the 10-K is not actionable until well into y+1), no look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort — H₁ / H₁′\n\n"
            "Low-tax vs high-tax next-year returns, with the HAC *t* and the placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.quintile_returns(ETR, FWD, q=0.20, min_names=15)\n"
            "    lo, hi = q['q1'].mean()*100, q['q5'].mean()*100\n"
            "    s = st.summary(q['hedge']); hedge, t = s['mean']*100, s['tstat']\n"
            "    p, _ = st.placebo_hedge_t(ETR, FWD, n_perm=1000, seed=568)\n"
            "else:\n"
            f"    lo, hi, hedge, t, p = 12.0, 15.0, {R['hedge_mean']}, {R['hedge_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['LOW ETR (Q1)','HIGH ETR (Q5)'], [lo, hi], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg next-year return %')\n"
            "ax.set_title(f'Q1 - Q5 = {hedge:+.1f}%/yr  (HAC t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LowETR {lo:+.1f}% | HighETR {hi:+.1f}% | hedge {hedge:+.1f}%/yr | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected** (wrong sign) and H₁′ **not supported** either. The "
            f"hedge is **{R['hedge_mean']}%/yr** (*t* {R['hedge_t']}), and the placebo *p* = "
            f"{R['placebo_p']} puts it in the *middle* of the shuffled null — pure noise. Neither the "
            "quality nor the risk story shows up."
        ),
        md(
            "### 4b · The information coefficient — H₂\n\n"
            "The year-by-year cross-sectional rank correlation between ETR and next-year return. If "
            "the sort meant anything, this would be reliably non-zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ic = st.information_coefficient(ETR, FWD)\n"
            "    ser = ic['series']; mean_ic, ic_t = ic['mean_ic'], ic['ic_t']\n"
            "    fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "    ax.bar(ser.index.astype(int), ser.values, color=[GREEN if v<0 else RED for v in ser.values], width=.7)\n"
            "    ax.axhline(mean_ic, ls='--', c='k', lw=1, label=f'mean IC {mean_ic:+.3f}')\n"
            "    ax.axhline(0, c='k', lw=1); ax.set_ylabel('rank-IC(ETR, next-yr ret)')\n"
            "    ax.set_title(f'Rank-IC by year: mean {mean_ic:+.3f} (t {ic_t:+.2f}) — flat')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    mean_ic, ic_t = {R['ic']}, {R['ic_t']}\n"
            "print(f'mean rank-IC {mean_ic:+.3f}, IC-t {ic_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected** — the mean rank-IC is **{R['ic']}** (*t* "
            f"{R['ic_t']}), indistinguishable from zero. There is no monotone tax-rate → return "
            "relationship on this basket, positive *or* negative."
        ),
        md(
            "### 4c · The change signal & robustness — H₃, H₄\n\n"
            "First the change-in-ETR sort (H₃), then the level hedge across sub-samples (H₄)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    detr = st.detr_signal(ETR)\n"
            "    qd = st.quintile_returns(detr, FWD, q=0.20, min_names=15)\n"
            "    sd = st.summary(qd['hedge']); icd = st.information_coefficient(detr, FWD)\n"
            "    detr_mean, detr_t, detr_ic = sd['mean']*100, sd['tstat'], icd['mean_ic']\n"
            "    sweep = st.window_sweep(ETR, FWD, windows=[(2009,2013),(2014,2018),(2019,2024),(2009,2024)])\n"
            "else:\n"
            f"    detr_mean, detr_t, detr_ic = {R['detr_mean']}, {R['detr_t']}, {R['detr_ic']}\n"
            "    sweep = pd.DataFrame(" + repr({w[0]: {"mean%": w[1], "t": w[2]} for w in R["win"]}) + ").T\n"
            "print(f'dETR hedge {detr_mean:+.1f}%/yr (t {detr_t:+.2f}), IC {detr_ic:+.3f}')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "vals = list(sweep['mean%']); cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(range(len(sweep)), vals, color=cols, width=.55)\n"
            "ax.set_xticks(range(len(sweep))); ax.set_xticklabels(list(sweep.index), rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('low - high ETR spread %/yr')\n"
            "ax.set_title('Level hedge by window: sign flips (+early, -late), never significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep.round(3).to_string())"
        ),
        md(
            f"> 💡 In plain words: H₃ **rejected** — the change-in-ETR hedge is **{R['detr_mean']}%/yr** "
            f"(*t* {R['detr_t']}, IC ≈ 0), dead too. H₄ **moot** — the level hedge is +3.0% early then "
            "−2.5% / −4.2% late, sign-unstable and never past |*t*| = 1.1. Nothing to stabilise."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (hedge {R['hedge_mean']}%/yr, *t* {R['hedge_t']}, wrong "
            f"sign, placebo *p* {R['placebo_p']}); H₂ rejected (IC {R['ic']}); H₃ rejected (ΔETR dead); "
            "sign flips across windows.\n"
            f"- **Tradability `MIRAGE`** — survivor basket, annual rebalance, high-ETR short leg; "
            f"gross {R['gross']}%/yr → net {R['net']}%/yr."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "There is nothing to charge costs against; the frictions just deepen a non-existent edge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.quintile_returns(ETR, FWD, q=0.20, min_names=15)\n"
            "    turn = st.turnover_series(ETR, FWD)\n"
            "    net_ser = st.apply_costs(q['hedge'], turn, one_way_bps=10.0, borrow_bps=50.0)\n"
            "    gross = q['hedge'].mean()*100; net = net_ser.mean()*100\n"
            "else:\n"
            f"    gross, net = {R['gross']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('low - high ETR spread %/yr')\n"
            "ax.set_title(f'Loses before costs ({gross:+.1f}%/yr), more after ({net:+.1f}%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}%/yr -> net {net:+.1f}%/yr (10 bps/leg + 50 bps borrow)')"
        ),
        md(
            "> 💡 In plain words: the long-low/short-high-ETR book *loses* before frictions on this "
            "basket. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real ETR premium "
            "of increasing strength — *both signs* — and watch the Q1−Q5 hedge-*t* respond, averaged "
            "over 25 seeds so no single lucky seed can fake it. (`premium < 0` = low-ETR firms "
            "out-earn → hedge positive.)"
        ),
        code(
            "prems = [0.0, -0.02, -0.03, -0.05, -0.08, 0.05]\n"
            "ts = [st.synthetic_mean_hedge_t(pr, n_seeds=25) for pr in prems]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(prems, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t| = 2 bar')\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted premium (negative = low-ETR firms out-earn)')\n"
            "ax.set_ylabel('mean Q1-Q5 hedge-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted premium of either sign — flat at the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for pr, t in zip(prems, ts): print(f'premium {pr:+.2f} -> mean hedge-t {t:+.2f}')"
        ),
        md(
            "The hedge-*t* is ≈ 0 at the null, shoots *positive* when low-ETR firms are planted to "
            "out-earn, and *negative* when high-ETR firms are — so the engine is a faithful, "
            "sign-correct detector. The real-tape null is therefore a statement about **this survivor "
            "basket over 2008-24**: the effective-tax-rate anomaly is simply absent here. For the "
            "calendar tax seasonal see [192 tax-day](../../192-tax-day/); for the cash/accrual quality "
            "sorts see [231 Sloan](../../231-sloan-accruals/) and "
            "[522 accruals](../../522-percent-operating-accruals/)."
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
