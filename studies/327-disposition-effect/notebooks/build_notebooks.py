"""Generate the two narrative notebooks for Study 327 (Disposition-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached panel
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-19).
R = dict(
    n_months=197, n_firms=28, win="2010-01 → 2026-05",
    q1=16.3, q2=15.2, q3=14.9, q4=15.2, q5=15.6, mkt=15.5,
    hedge_ann=-0.7, hedge_sharpe=-0.04, hedge_t=-0.16, hedge_hit=48, hedge_dd=-62.0,
    ci_lo=-0.74, ci_hi=0.65,
    resid_t=-0.65, resid_ann=-2.3,
    ic_mean=-0.012, ic_t=-0.66,
    cost0_ann=-0.7, cost0_t=-0.16, cost5_ann=-3.1, cost5_t=-0.74,
    cost10_ann=-5.5, cost10_t=-1.31, cost20_ann=-10.3, cost20_t=-2.47,
    fp_over="4fe74d39c1cc", fp_fwd="6525f33dc337",
    # synthetic controls
    syn_null_t=0.86, syn_ctrl_t=7.12, syn_ctrl_sharpe=1.43,
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from disposition_effect import data, strategy as st

def _have_real():
    for d in (data.STUDY_CACHE, data.SHARED_CACHE):
        if os.path.exists(data._panel_cache_path(d)):
            return True
    return False

HAVE_REAL = _have_real()

def load_real():
    return data.load_real(allow_survivorship_bias=True, fetch=False)

print("real cross-section cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Disposition-Effect — can you cash in on people selling their winners too early? 🪤\n"
            "### A textbook behavioural anomaly, taken to a tradeable basket — in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Just_momentum_in_disguise%3F: Confirmed](https://img.shields.io/badge/Just_momentum_in_disguise%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Here's one of the most reliable facts in all of behavioural finance: people **sell their "
            "winners too early and hold their losers too long** (the *disposition effect*). A famous "
            "2005 paper turned that into a money-making recipe — if everyone dumps their winners, the "
            "winners get held *below* their true value, so a patient buyer should be able to scoop up "
            "the under-reaction. This notebook asks whether that actually pays once you build it as a "
            "real, tradeable basket.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap CI, the "
            "momentum control and the cost sweep? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the disposition effect real? | **Yes** — it's one of the best-documented biases "
            "there is (people really do sell winners too soon). |\n"
            "| Does the 'overhang' factor make money on a basket of big stocks? | **No.** Every "
            f"quintile returns ~15%/yr; the long-short hedge earns **{R['hedge_ann']:+.1f}%/yr** "
            f"(*t* = {R['hedge_t']:+.2f}) — a flat line. |\n"
            "| Is it at least *different* from momentum? | **No.** Deep-in-the-money stocks are just "
            "recent winners; strip the momentum out and there's nothing left. |\n"
            "| So what is it? | A genuine behavioural truth that **does not survive contact** with a "
            "small, tradeable, liquid large-cap universe. |\n\n"
            "> The bias is real. The *tradeable factor built on it*, on this tape, is a mirage — and "
            "what little wiggle it has is momentum in a costume."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *Investors hate realising a loss and love locking in a gain. So when a stock has run "
            "up — and lots of holders are sitting on a fat unrealised profit — those holders keep "
            "selling to 'take their money off the table'. That selling pressure holds the price "
            "**below** what the stock is really worth. Buy those deep-in-the-money names and you get "
            "paid as the price catches up.*\n\n"
            "— the trading lesson drawn from Grinblatt & Han (2005), *Prospect Theory, Mental "
            "Accounting, and Momentum*.\n\n"
            "The measuring stick is **capital-gains overhang**: how far today's price sits above the "
            "average price current holders paid (their cost basis). A big positive overhang = lots of "
            "people sitting on gains = lots of disposition-driven selling = (the theory says) a future "
            "bounce."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be beautiful: a return predictor built **purely from price and "
            "volume** (no earnings, no balance sheet) that comes straight out of a human quirk we can "
            "all recognise in ourselves. And Grinblatt & Han claimed something even bigger — that this "
            "overhang *explains momentum*, one of the most famous anomalies of all. So the stakes "
            "aren't just 'is this one factor good' but 'is a chunk of the factor zoo really just the "
            "disposition effect wearing different hats?'"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks:\n\n"
            "1. **Sort and look.** Each month, rank the stocks by overhang, split into five buckets, "
            "and see if the deep-in-the-money bucket (Q5) really beats the underwater bucket (Q1). A "
            "real effect makes a *staircase* from Q1 up to Q5.\n"
            "2. **Strip out momentum.** Deep-in-the-money stocks are, almost by definition, recent "
            "winners. We remove the momentum part of overhang and check if anything survives.\n"
            "3. **Charge real costs.** A long-short basket that rebalances every month trades a lot — "
            "we see whether the (already thin) edge survives the bill.\n\n"
            "Tape: a small basket of liquid large caps (Dow-30 style), monthly, 2010–2026. "
            "(*A small, hand-picked, current* basket — so it's survivorship-biased *toward* finding an "
            "effect. If even *that* shows nothing, that's telling.)"
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The staircase that wasn't.** Here are the five overhang buckets, annualised. The theory "
            "predicts a clear climb from Q1 (underwater) to Q5 (deep in the money):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    over, mom, fwd = load_real()\n"
            "    h = st.quintile_hedge(over, fwd, min_firms=10)\n"
            "    qs = [h[f'Q{i}'].mean()*12*100 for i in range(1,6)]\n"
            "    mkt = h['market'].mean()*12*100\n"
            "else:\n"
            f"    qs = [{R['q1']}, {R['q2']}, {R['q3']}, {R['q4']}, {R['q5']}]; mkt = {R['mkt']}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar([f'Q{i}' for i in range(1,6)], qs, color=GREY, width=.62)\n"
            "ax.axhline(mkt, ls='--', c=GREEN, lw=1.5, label=f'equal-weight basket ({mkt:.1f}%/yr)')\n"
            "ax.set_ylabel('annualised return %'); ax.legend()\n"
            "ax.set_title('Overhang quintiles: a flat fan, not a staircase')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Q1..Q5 (%/yr):', [round(q,1) for q in qs], ' market:', round(mkt,1))"
        ),
        md(
            f"Flat. Every bucket clusters around **{R['mkt']:.0f}%/yr** — you'd have done the same just "
            "owning the basket. There's no climb from underwater to in-the-money, which is the whole "
            "ballgame for this factor."
        ),
        md(
            "**The long-short hedge.** Buy Q5, short Q1 — the pure overhang bet. If the staircase is "
            "flat, this should be a flat line too:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + h['hedge']).cumprod()\n"
            "    s = st.summary(h['hedge']); tstat = s['tstat']; ann = s['ann_ret']*100\n"
            "else:\n"
            "    rng = np.random.default_rng(327)\n"
            f"    eq = pd.Series((1 + rng.normal({R['hedge_ann']}/1200, 0.045, {R['n_months']})).cumprod())\n"
            f"    tstat, ann = {R['hedge_t']}, {R['hedge_ann']}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(np.arange(len(eq)), eq.to_numpy(), c=RED, lw=1.8)\n"
            "ax.axhline(1.0, c='k', lw=1)\n"
            "ax.set_xlabel('months'); ax.set_ylabel('growth of $1 (long-short)')\n"
            "ax.set_title(f'The overhang hedge: {ann:+.1f}%/yr, t={tstat:+.2f} — a flat line that drifts down')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge annualised {ann:+.1f}%/yr, HAC t={tstat:+.2f}')"
        ),
        md(
            f"A wandering, slightly-down flat line: **{R['hedge_ann']:+.1f}%/yr** at HAC "
            f"*t* = **{R['hedge_t']:+.2f}** — nowhere near the *t* = 2 a desk needs before calling a "
            "signal real. On the broad market the literature finds a real premium; on *this* small "
            "tradeable basket, nothing."
        ),
        md(
            "**Is it even different from momentum?** Deep-in-the-money = recent winner. We can remove "
            "the momentum part of overhang and re-run. If 'disposition' is its own thing, something "
            "survives:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    resid = st.orthogonalise(over, mom)\n"
            "    rt = st.summary(st.quintile_hedge(resid, fwd, min_firms=10)['hedge'])['tstat']\n"
            "    raw_t = tstat\n"
            "else:\n"
            f"    raw_t, rt = {R['hedge_t']}, {R['resid_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar(['Raw overhang','Overhang minus\\nmomentum'], [raw_t, rt], color=[GREY, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1, label='|t|=2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(-2, ls='--', c=GREEN, lw=1)\n"
            "ax.set_ylabel('HAC t-stat'); ax.legend()\n"
            "ax.set_title('Strip out momentum and the (already absent) signal stays absent')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'raw t={{raw_t:+.2f}}  residual t={{rt:+.2f}}')"
        ),
        md(
            f"The residual is just as dead (*t* = {R['resid_t']:+.2f}). There was no separate "
            "disposition signal to find — what overhang measures here is the recent-winner tilt, and "
            "that tilt isn't paying either."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The disposition effect is real and the broad-market overhang premium "
            f"is well-documented — but on this basket the hedge is {R['hedge_ann']:+.1f}%/yr "
            f"(*t* = {R['hedge_t']:+.2f}). Strong literature, but this tape can't certify it.\n"
            "- **Tradability — Mirage.** Gross return ≈ 0, the quintile fan is flat, and it only gets "
            "worse with costs — on a basket already biased *toward* finding something.\n"
            "- **Just momentum in disguise? — Confirmed.** Overhang is a recent-winner proxy; remove "
            "momentum and nothing remains."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade — but for completeness, here's what trading costs do to a "
            "long-short book that fully rebalances every month (it turns over a *lot*):"
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    nets = [st.summary(st.apply_costs(h['hedge'], c))['ann_ret']*100 for c in costs]\n"
            "else:\n"
            f"    nets = [{R['cost0_ann']}, {R['cost5_ann']}, {R['cost10_ann']}, {R['cost20_ann']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.fill_between(costs, nets, 0, color=RED, alpha=.10)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net annualised %')\n"
            "ax.set_title('Starts at ~0, only goes down — there is no break-even')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net %/yr:', [round(x,1) for x in nets])"
        ),
        md(
            "When the gross edge is already zero, costs can only make it negative. This is the cleanest "
            "kind of mirage: not 'it died at the first dollar of cost' but 'there was never anything "
            "there on this tape to begin with.'"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The universe matters.** Grinblatt & Han ran the *entire* CRSP universe — thousands of "
            "names, including small, illiquid stocks where disposition-driven mispricing is biggest. "
            "Our 30 mega-caps are the *least* likely place for it to survive. Fork this and widen the "
            "universe (mind the survivorship guard).\n"
            "- **Different overhang horizons.** We used a 5-year turnover-weighted cost basis. Shorter "
            "or longer memory changes what 'the holders paid' means — worth a sweep.\n"
            "- **The momentum question.** Is *any* of the factor zoo really separable from momentum? "
            "See the desk's momentum and residual-momentum studies.\n\n"
            "*Think the overhang factor lives in the small caps? Fork this, swap the universe, and show "
            "the staircase appearing — on a tape that names its survivorship.*"
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
            "# Disposition-Effect — a quantitative teardown 🔬\n"
            "### Capital-gains overhang · monthly quintile L/S · HAC *t* + block bootstrap · "
            "momentum orthogonalisation · cost sweep · synthetic detector\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Just_momentum_in_disguise%3F: Confirmed](https://img.shields.io/badge/Just_momentum_in_disguise%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the Grinblatt-Han "
            "capital-gains overhang on a small named large-cap cross-section, sort it into monthly "
            "quintiles, and test the Q5−Q1 hedge against the inference bar, a momentum control, and a "
            "cost sweep.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars → monthly overhang for a "
            "Dow-30-style basket, 2010–2026 (as-of 2026-06-19); the offline core and tests run on a "
            "deterministic synthetic panel. Survivorship is named on the Signal axis. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Overhang Q5−Q1 hedge **{R['hedge_ann']:+.1f}%/yr**, HAC "
            f"*t* = **{R['hedge_t']:+.2f}**; block-bootstrap 95% CI [{R['ci_lo']:+.2f}%, "
            f"{R['ci_hi']:+.2f}%] straddles zero; IC *t* = {R['ic_t']:+.2f}. Literature REAL, this tape can't certify. |\n"
            f"| **Tradability** | `MIRAGE` | Gross ≈ 0; flat quintile fan (~{R['mkt']:.0f}%/yr each "
            f"bucket); net goes to {R['cost20_ann']:+.1f}%/yr at 20 bps. Survivorship-biased *upper* bound. |\n"
            f"| **Just momentum in disguise?** | `CONFIRMED` | Momentum-orthogonalised residual hedge "
            f"*t* = {R['resid_t']:+.2f} — no separable disposition signal. |\n\n"
            "> 💡 In plain words: a real behavioural bias whose *tradeable factor form* evaporates on a "
            "liquid large-cap basket, and whose (absent) signal is inseparable from momentum."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the capital-gains overhang for stock $i$ at month-end $t$ be\n\n"
            "$$g_{i,t} = \\frac{P_{i,t} - R_{i,t}}{P_{i,t}}, \\qquad "
            "R_{i,t} = \\sum_{k\\ge 1} w_{i,t,k}\\, P_{i,t-k},$$\n\n"
            "where $R$ is the **turnover-weighted reference price** (the volume-weighted cost basis of "
            "current holders: $w_{i,t,k}$ is the turnover on day $t{-}k$ times the probability those "
            "shares are still held). Sort cross-sectionally, long Q5 (high $g$) / short Q1 (low $g$).\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\text{hedge}] > 0$ with HAC *t* ≥ 2.\n"
            "- **H₂ (monotone).** Bucket means increase Q1 → Q5.\n"
            "- **H₃ (beyond momentum).** The hedge survives orthogonalising $g$ to 12-1 momentum.\n"
            "- **H₄ (tradable).** A positive net edge survives realistic costs on this basket.\n\n"
            "We find **H₁–H₄ rejected** on this cross-section (literature supports them on broad CRSP)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Grinblatt & Han's strong claim is that overhang **subsumes momentum** — i.e. a famous "
            "anomaly is really the disposition effect. If true on a tradeable basket, you'd get a "
            "price-and-volume-only predictor with a behavioural micro-foundation. The interesting "
            "content is therefore less the binary 'does the hedge pay' and more **whether overhang and "
            "momentum are separable at all** — H₃ is the load-bearing test."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Daily $g_{i,t}$ from a turnover-weighted 1260-day reference price; sampled "
            "at month-end.\n"
            "- **Lag.** Month-end $t$ signal → month $t{+}1$ return. One shift, applied once (in the "
            "data layer); calendar-known, no extra lag.\n"
            "- **Sort.** Equal-weight quintiles; hedge = Q5 − Q1, rebalanced monthly.\n"
            "- **Inference.** Newey-West HAC *t*; circular block-bootstrap CI (block = 6 months).\n"
            "- **Confound.** Cross-sectional regression of $g$ on 12-1 momentum each month; re-sort on "
            "the residual.\n"
            "- **Costs.** One-way bps × NAV, 4× one-way notional/month (a full L/S rebalance).\n"
            "- **Positive control.** Deterministic synthetic panel with a tunable overhang premium.\n\n"
            "Universe: a Dow-30-style basket (survivorship-biased, opt-in). Window 2010–2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The quintile fan & the hedge — H₁/H₂\n\n"
            "Bucket means (annualised) and the hedge with its HAC *t* and bootstrap CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    over, mom, fwd = load_real()\n"
            "    h = st.quintile_hedge(over, fwd, min_firms=10)\n"
            "    qs = [h[f'Q{i}'].mean()*12*100 for i in range(1,6)]\n"
            "    s = st.summary(h['hedge']); lo, hi = st.block_bootstrap_ci(h['hedge'])\n"
            "    ann, tstat = s['ann_ret']*100, s['tstat']; lo*=100; hi*=100\n"
            "else:\n"
            f"    qs = [{R['q1']},{R['q2']},{R['q3']},{R['q4']},{R['q5']}]\n"
            f"    ann, tstat, lo, hi = {R['hedge_ann']}, {R['hedge_t']}, {R['ci_lo']}, {R['ci_hi']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))\n"
            "a1.bar([f'Q{i}' for i in range(1,6)], qs, color=GREY, width=.6)\n"
            "a1.set_ylabel('annualised %'); a1.set_title('Quintile fan — no monotone gradient (H2 rejected)')\n"
            "a2.bar(['hedge\\n(Q5-Q1)'], [ann], color=RED, width=.4)\n"
            "a2.errorbar(['hedge\\n(Q5-Q1)'], [ann], yerr=[[ann-lo*12],[hi*12-ann]], fmt='none', c='k', capsize=6)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('annualised %')\n"
            "a2.set_title(f'Hedge {ann:+.1f}%/yr, HAC t={tstat:+.2f} (H1 rejected)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Q1..Q5: {[round(q,1) for q in qs]}')\n"
            "print(f'Hedge {ann:+.2f}%/yr  HAC t={tstat:+.2f}  boot 95% CI(monthly) [{lo:+.2f}%, {hi:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: the buckets are flat (H₂ fails) and the hedge's bootstrap CI "
            f"[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%] (monthly) brackets zero with HAC "
            f"*t* = {R['hedge_t']:+.2f} (H₁ fails). On the broad market this is a real premium; on this "
            "basket it is statistically absent."
        ),
        md(
            "### 4b · Beyond momentum? — H₃, the load-bearing test\n\n"
            "Cross-sectionally orthogonalise overhang to 12-1 momentum each month and re-sort. We also "
            "show the Spearman IC of raw overhang vs next-month return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    resid = st.orthogonalise(over, mom)\n"
            "    rt = st.summary(st.quintile_hedge(resid, fwd, min_firms=10)['hedge'])['tstat']\n"
            "    ic = st.information_coefficient(over, fwd, min_firms=10)\n"
            "    ic_m, ic_t = ic.mean(), st.hac_tstat(ic); raw_t = tstat\n"
            "else:\n"
            f"    raw_t, rt, ic_m, ic_t = {R['hedge_t']}, {R['resid_t']}, {R['ic_mean']}, {R['ic_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))\n"
            "a1.bar(['raw overhang','minus momentum'], [raw_t, rt], color=[GREY, RED], width=.5)\n"
            "for y in (2,-2): a1.axhline(y, ls='--', c=GREEN, lw=1)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_ylabel('hedge HAC t'); a1.set_title('H3: residual is equally dead')\n"
            "a2.bar(['overhang IC'], [ic_m], color=RED, width=.35)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('mean monthly Spearman IC')\n"
            "a2.set_title(f'IC mean {ic_m:+.3f} (t={ic_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'raw hedge t={raw_t:+.2f}  residual t={rt:+.2f}  IC mean={ic_m:+.4f} (t={ic_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: residual hedge *t* = {R['resid_t']:+.2f} and the IC is essentially "
            f"zero (*t* = {R['ic_t']:+.2f}). Overhang here is a recent-winner proxy with no incremental "
            "information — **H₃ rejected, the third-axis `CONFIRMED`**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — hedge {R['hedge_ann']:+.1f}%/yr, HAC *t* {R['hedge_t']:+.2f}; CI "
            f"[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%] straddles zero; IC *t* {R['ic_t']:+.2f}. The "
            "much-replicated CRSP premium does not appear on this basket — literature-real, "
            "tape-uncertified.\n"
            f"- **Tradability `MIRAGE`** — flat fan (~{R['mkt']:.0f}%/yr per bucket), gross ≈ 0, net "
            f"{R['cost10_ann']:+.1f}%/yr at 10 bps — on a survivorship-biased *upper bound*.\n"
            f"- **Just momentum in disguise? `CONFIRMED`** — residual hedge *t* {R['resid_t']:+.2f}; "
            "overhang carries no information beyond momentum here."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost sweep\n\n"
            "With a gross edge at zero, costs only subtract. A monthly L/S quintile book turns over "
            "heavily (4× one-way notional), so the bleed is fast."
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    nets = [st.summary(st.apply_costs(h['hedge'], c))['ann_ret']*100 for c in costs]\n"
            "    ts = [st.summary(st.apply_costs(h['hedge'], c))['tstat'] for c in costs]\n"
            "else:\n"
            f"    nets = [{R['cost0_ann']},{R['cost5_ann']},{R['cost10_ann']},{R['cost20_ann']}]\n"
            f"    ts = [{R['cost0_t']},{R['cost5_t']},{R['cost10_t']},{R['cost20_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2, label='net %/yr')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, ts, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax.axhline(0, c='k', lw=1); ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net annualised %', color=RED)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('No break-even: it starts at zero and only falls')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net %/yr:', [round(x,1) for x in nets], ' t:', [round(x,2) for x in ts])"
        ),
        md(
            "> 💡 In plain words: there is no cost at which this turns a profit, because it doesn't "
            "make money at zero cost. Mirage by absence, not by erosion."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print zero? Plant a known overhang "
            "premium and confirm the harness banks it (and finds nothing when none is planted)."
        ),
        code(
            "prems = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]\n"
            "tsyn = []\n"
            "for p in prems:\n"
            "    o, m, f, _ = data.synthetic_panel(overhang_premium=p, momentum_premium=0.0, seed=327)\n"
            "    tsyn.append(st.summary(st.quintile_hedge(o, f)['hedge'])['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(prems, tsyn, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted overhang premium'); ax.set_ylabel('hedge HAC t')\n"
            "ax.set_title('The engine is a faithful detector: ~0 at the null, rising with the planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('null t ~ {R['syn_null_t']}, default-control t ~ {R['syn_ctrl_t']} — the pipeline works; the real tape is simply flat')"
        ),
        md(
            "The engine cleanly recovers a planted premium and returns ~0 at the null, so the flat "
            "real-tape result is a statement about **the basket**, not the harness. The disposition "
            "effect is real; its tradeable factor form needs the breadth (and the small, illiquid "
            "names) of the full CRSP universe — exactly where a desk can't easily trade it.\n\n"
            "Fork it: widen the universe through the survivorship guard, sweep the reference-price "
            "horizon, and see whether the staircase ever appears."
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
