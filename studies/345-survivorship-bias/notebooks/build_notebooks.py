"""Generate the two narrative notebooks for Study 345 (Survivorship-Bias).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic (the dead names can be toggled on and off);
the real-tape cell reads the cached panel parquet under ../_cache/ if present and otherwise
quotes the frozen headline numbers in ``R`` (mirroring docs/results.md).
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    fp_synth="8d30d737ae8c", n_assets=200, n_periods=240, n_dead=60,
    # death-rate sweep: (rate, full_t, surv_t, full_mean, surv_mean, delta_sharpe)
    dr=[0.00, 0.15, 0.30, 0.45],
    full_t=[2.87, 2.09, 1.38, 0.27],
    surv_t=[2.87, 2.60, 2.83, 2.84],
    full_mean=[9.83, 7.20, 4.87, 1.00],
    surv_mean=[9.83, 8.75, 9.60, 9.85],
    delta_sharpe=[0.00, 0.10, 0.30, 0.54],
    # the headline (death_rate 0.30)
    h_full_t=1.38, h_surv_t=2.83, h_full_mean=4.87, h_surv_mean=9.60, h_delta_sharpe=0.30,
    # real illustration (survivors-only convenience universe)
    real_window="2005-02 → 2026-05", real_months=255, real_names=20, real_fp="ba8283a7a103",
    real_sharpe=0.78, real_mean=17.78, real_t=3.52,
)


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

from survivorship_bias import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def real_panel():
    return data.load_real(fetch=False, allow_survivorship_bias=True)

print("real panel cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Survivorship-Bias — the dead companies you forgot to include 🪦\n"
            "### How much of a 'great strategy' is just the losers nobody put in the test?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Manufactures_an_edge%3F: Confirmed](https://img.shields.io/badge/Manufactures_an_edge%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "When you build a backtest from *today's* list of companies, you quietly delete "
            "everyone who went bankrupt, got bought out, or fell out of the index. The losers "
            "vanish — *before the test ever sees them*. This notebook takes one strategy, runs "
            "it on a tape that **keeps the companies that died** and on the same tape with them "
            "**deleted**, and watches a winning strategy appear out of thin air.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the death-rate sweep "
            "and the bootstrap? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does deleting the dead companies change the backtest? | **It can invent the "
            "whole result.** On a panel with *no real edge at all*, deleting the firms that "
            f"went bankrupt turns a nothing (*t* = {R['h_full_t']:.1f}) into a textbook "
            f"'discovery' (*t* = {R['h_surv_t']:.1f}). |\n"
            "| How big is the effect? | **As big as the death rate.** The more companies die, "
            "the bigger the lie — and at a realistic deletion rate it's the *entire* edge. |\n"
            "| Is the effect a coding artefact? | **No — the control proves it.** With zero "
            "deaths the two tapes are identical and the gap is exactly **0.00**. The harness "
            "isn't cheating; the deletion is. |\n"
            "| Can you trade the 'edge'? | **There's nothing there.** Put the dead companies "
            "back and the strategy's edge evaporates. It was never real. |\n\n"
            "> A backtest run on survivors-only data isn't measuring a strategy. It's measuring "
            "which companies happened to still be alive when you made the list."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Here's a tempting strategy: **buy the stocks that fell the most last year** — the "
            "beaten-down names, the bargains, the ones due for a bounce. Backtest it on the "
            "S&P 500 and it looks *wonderful*: those losers came roaring back.\n\n"
            "The catch nobody mentions: you backtested it on the S&P 500 **as it is today**. "
            "Every company that bought the 'bargain' dip and then went to **zero** — Lehman, "
            "Bear Stearns, the dot-com flameouts, the 2009 bankruptcies — is *gone from your "
            "list*. You only ever 'bought' the losers that lived. The strongest version of the "
            "warning: **survivorship bias can manufacture an edge from a strategy that has "
            "none**, and most people never check."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, a huge fraction of impressive-looking backtests are measuring nothing but "
            "*who survived*. Mutual-fund studies have known this for decades — funds that "
            "blew up get scrubbed from the database, so the survivors' average return is "
            "overstated by a percent or more a year (Brown–Goetzmann–Ibbotson–Ross 1992, "
            "Malkiel 1995). The same trap sits under every strategy backtested on 'the current "
            "index, projected back.' The scary part isn't that it *inflates* a real edge — "
            "it's that it can **invent one from pure noise**. That's what we'll show."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We build a fake stock market where **we control who dies**. A fraction of the "
            "companies are doomed: their price bleeds, they take a near-total wipe-out, and "
            "then they **delist** — they leave the tape, exactly like a real bankruptcy. "
            "Crucially, we plant **no real edge** in this market.\n\n"
            "Then we run *one* strategy — buy last year's losers — on two views of the same "
            "data:\n\n"
            "1. **FULL** — every company, including the ones that died. The honest tape.\n"
            "2. **SURVIVORS** — the same tape with the dead companies deleted (a "
            "current-membership universe projected back).\n\n"
            "**The mirage test:** if the survivors-only view shows a 'significant' edge that "
            "the full view doesn't, the edge was manufactured by the deletion. And the "
            "**control:** if we set the death rate to *zero*, the two views must be identical — "
            "or our harness is the thing that's broken."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First the control.** Turn deaths OFF. With nobody delisting, FULL and SURVIVORS "
            "are literally the same panel, so the strategy must score *identically*. Then turn "
            "deaths up and watch the two reads split apart."
        ),
        code(
            "rates = [0.0, 0.15, 0.30, 0.45]\n"
            "full_t, surv_t = [], []\n"
            "for dr in rates:\n"
            "    rf, alive, _ = data.synthetic_panel(n_assets=200, death_rate=dr,\n"
            "                                        signal_strength=0.0, seed=345)\n"
            "    surv = data.survivors_only(rf, alive)\n"
            "    res = st.compare_views(rf, surv, long_short=False, direction=-1, cost_bps=0.0)\n"
            "    full_t.append(res['test_full']['tstat'])\n"
            "    surv_t.append(res['test_survivors']['tstat'])\n"
            "x = [r*100 for r in rates]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.plot(x, surv_t, 'o-', c=RED, lw=2.5, label='SURVIVORS-only backtest')\n"
            "ax.plot(x, full_t, 's-', c=GREEN, lw=2.5, label='FULL (honest) backtest')\n"
            "ax.axhline(2, ls=':', c=GREY); ax.text(1, 2.06, 't = 2  (\"significant\")', c=GREY)\n"
            "ax.set_xlabel('% of companies that go bankrupt and delist')\n"
            "ax.set_ylabel('strategy t-stat')\n"
            "ax.set_title('Same strategy, same data — deleting the dead invents an edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('FULL t      :', [f'{t:+.2f}' for t in full_t])\n"
            "print('SURVIVORS t :', [f'{t:+.2f}' for t in surv_t])"
        ),
        md(
            f"At **0% deaths** both lines sit on top of each other at *t* ≈ {R['full_t'][0]:.1f} "
            "— the control passes, the harness invents nothing. But as companies start dying, "
            "the **honest (green) line collapses toward zero** (the strategy keeps buying names "
            f"that go bankrupt), while the **survivors-only (red) line stays up around *t* ≈ "
            f"{R['surv_t'][-1]:.1f}**. The deletion is holding the 'edge' up single-handedly."
        ),
        md(
            "**The headline, head-to-head.** At a realistic ~30% deletion rate, here are the "
            "two reads of the identical strategy on the identical (signal-free) data:"
        ),
        code(
            "rf, alive, _ = data.synthetic_panel(n_assets=200, death_rate=0.30,\n"
            "                                    signal_strength=0.0, seed=345)\n"
            "surv = data.survivors_only(rf, alive)\n"
            "res = st.compare_views(rf, surv, long_short=False, direction=-1, cost_bps=0.0)\n"
            "ft, stt = res['test_full']['tstat'], res['test_survivors']['tstat']\n"
            "fm, sm = res['test_full']['mean_ann']*100, res['test_survivors']['mean_ann']*100\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "b = ax.bar(['FULL\\n(honest)','SURVIVORS\\n(dead deleted)'], [ft, stt],\n"
            "           color=[GREEN, RED], width=.5)\n"
            "ax.axhline(2, ls=':', c=GREY); ax.set_ylabel('strategy t-stat')\n"
            "ax.set_title('No edge on the honest tape — a \"discovery\" on the survivors tape')\n"
            "for bar_, m in zip(b, [fm, sm]):\n"
            "    ax.annotate(f'{m:+.1f}%/yr', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'FULL: {fm:+.2f}%/yr  t={ft:+.2f}   SURVIVORS: {sm:+.2f}%/yr  t={stt:+.2f}')"
        ),
        md(
            f"There it is. On the **honest** tape the strategy earns {R['h_full_mean']:.1f}%/yr "
            f"at *t* = {R['h_full_t']:.1f} — **not significant**, no edge. Delete the dead "
            f"companies and the *exact same strategy* prints {R['h_surv_mean']:.1f}%/yr at "
            f"*t* = {R['h_surv_t']:.1f} — a textbook 'discovery.' Nothing about the strategy "
            "changed. We just stopped counting the companies that died."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On the honest tape the strategy has no edge. The only place a "
            "'significant' result appears is the survivors-only view, where the deletion put it "
            "there.\n"
            "- **Tradability — Mirage.** You can't trade an edge that exists only because you "
            "forgot to include the bankruptcies. Put them back and it's gone.\n"
            "- **Does survivorship bias manufacture an edge? — Confirmed.** It turns a "
            f"*t* = {R['h_full_t']:.1f} nothing into a *t* = {R['h_surv_t']:.1f} 'discovery,' "
            "grows with the death rate, and correctly vanishes when nobody dies."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and that's the whole point. A strategy whose edge lives entirely in the "
            "*deletion of dead companies* has no edge at all: in real life you'd have held the "
            "losers that went to zero, because you couldn't have known in advance which ones "
            "would survive. The honest tape *is* the live experience, and on the honest tape "
            "there's nothing. The trading lesson isn't 'buy losers' — it's **build your "
            "universe from point-in-time membership, with delisting returns booked**, or your "
            "backtest is a fairy tale starring only the heroes who lived."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A real, biased tape.** The companion notebook runs the same rule on a live "
            "large-cap universe — which is *itself* survivors-only — and shows the inflated "
            "*t* you'd naively report (and why you shouldn't trust it).\n"
            "- **Flip the strategy.** Try momentum (buy winners) instead of contrarian. The "
            "bias still points up, but the mechanism differs — fork it and see.\n"
            "- **Dial the wipe-out.** Make delistings only −50% instead of −80%, or merge some "
            "companies away (no loss) instead of bankrupting them. Survivorship bias has many "
            "flavours; this is the harshest one.\n\n"
            "*Think your favourite backtest is clean? Fork this, plant your strategy on the "
            "FULL tape, then delete the dead names and see how much of the 'alpha' was just the "
            "graveyard you left out.*"
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
            "# Survivorship-Bias — a quantitative teardown 🔬\n"
            "### One strategy, two tapes · death-rate sweep · HAC *t* + block bootstrap · the "
            "zero-death control · the delisting-loss mechanism · the opt-in guard\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Manufactures_an_edge%3F: Confirmed](https://img.shields.io/badge/Manufactures_an_edge%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We run one "
            "contrarian buy-losers book on a panel in two states (FULL with delistings, "
            "SURVIVORS without) and measure the bias as the **delta** between the two reads.\n\n"
            "> ⚠️ **Not investment advice.** The headline run is a deterministic synthetic panel "
            "(the dead names toggle on/off); the real cell uses a live large-cap tape that is "
            "*itself* survivors-only (as-of 2026-05-31, opt-in guard — see beat 3). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Honest FULL tape (30% deletion): mean "
            f"{R['h_full_mean']:.2f}%/yr, HAC *t* = **{R['h_full_t']:.2f}** (< 2). No real "
            "edge; the synthetic control never backs a stamp, and the only 'significant' read "
            "is the biased one. |\n"
            f"| **Tradability** | `MIRAGE` | The survivors-only edge ({R['h_surv_mean']:.2f}%/yr, "
            f"*t* = {R['h_surv_t']:.2f}) is gone the instant the delisted names are restored. |\n"
            f"| **Manufactures an edge?** | `CONFIRMED` | Δ Sharpe is **+0.00** at 0% deaths "
            f"(control) and rises monotonically to **+{R['delta_sharpe'][-1]:.2f}** at 45%; the "
            f"FULL *t* decays {R['full_t'][0]:.2f} → {R['full_t'][-1]:.2f} while SURVIVORS holds "
            f"~{R['surv_t'][-1]:.1f}. |\n\n"
            "> 💡 In plain words: with **no true signal planted**, deleting the firms that "
            "delisted is sufficient — on its own — to print a 'significant' strategy. The bias "
            "doesn't just inflate a real edge; here it manufactures one from zero."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the strategy be a contrarian cross-sectional book: each month, long the bottom "
            "quintile by trailing 12-month return (the 'losers'), equal-weight, one execution "
            "lag. Run it on returns $r^{full}$ (all firms, delistings booked as a terminal "
            "loss) and on $r^{surv}$ (only firms alive at the sample end).\n\n"
            "- **H₁ (the strategy has a real edge).** $\\mathbb{E}[r^{full}_t] > 0$ with HAC "
            "*t* ≥ 2.\n"
            "- **H₂ (survivorship inflates it).** $\\text{Sharpe}(r^{surv}) > "
            "\\text{Sharpe}(r^{full})$.\n"
            "- **H₃ (survivorship can *manufacture* it).** With **no** planted signal, "
            "$r^{full}$ is insignificant *yet* $r^{surv}$ clears *t* = 2.\n"
            "- **Control (the harness is honest).** At death rate 0, $r^{full} \\equiv r^{surv}$ "
            "and every statistic matches exactly.\n\n"
            "We find **H₁ rejected** on the honest tape, **H₂ and H₃ confirmed**, and the "
            "**control passes** to the bit."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Survivorship bias is the most-cited and least-controlled-for backtest pitfall "
            "(López de Prado 2018). The fund literature pins the average overstatement at "
            "~1–1.5%/yr (Brown et al. 1992; Malkiel 1995); Shumway (1997) shows that *dropping* "
            "delisted CRSP names rather than booking their delisting return biases returns up, "
            "worst for distressed stocks — exactly the names a contrarian rule buys. The desk's "
            "contribution is to make the bias a *controlled experiment*: one knob (the death "
            "rate), a true null (no planted signal), and a zero-death control, so we can show "
            "the bias is not a nuisance term but, at realistic deletion rates, **the entire "
            "result**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Panel.** 200 names, 240 monthly periods, seed 345. Hidden quality scores drive "
            "both expected return *and* who dies — so deletion preferentially removes the left "
            "tail (the realistic coupling). **No cross-sectional signal is planted** "
            "(`signal_strength = 0`): any edge is manufactured.\n"
            "- **Delisting.** A `death_rate` fraction (lowest quality) bleed, take a **−80% "
            "terminal bar**, then leave the tape (returns → NaN). Deaths are spread across the "
            "back two-thirds of the sample, so SURVIVORS is missing names throughout.\n"
            "- **Strategy.** Contrarian buy-losers, long-only bottom quintile, equal-weight, "
            "**one execution lag** (signal at close of *t* earns *t+1*), one-way costs × NAV.\n"
            "- **Two views.** FULL keeps the dead (delisting losses included); SURVIVORS keeps "
            "only the end-alive names — a current-membership universe projected back.\n"
            "- **Inference.** Newey-West HAC *t* on each view's monthly returns; circular "
            "block-bootstrap CI (Politis-Romano). Excess-of-zero (long-only, fully invested).\n"
            "- **Survivorship guard (the subject, named on Signal).** The real loader refuses "
            "to run without `allow_survivorship_bias=True`; the convenience universe is *itself* "
            "survivors-only, so its real-tape *t* is an explicit **upper bound**.\n"
            "- **The mirage line.** If SURVIVORS clears *t* = 2 while FULL does not — at a "
            "death rate of zero we'd see *no* gap — survivorship bias is confirmed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The control — zero deaths, zero gap\n\n"
            "Before anything, prove the harness manufactures nothing. With `death_rate = 0` the "
            "FULL and SURVIVORS panels are byte-identical, so every statistic must match."
        ),
        code(
            "rf, alive, _ = data.synthetic_panel(n_assets=200, death_rate=0.0,\n"
            "                                    signal_strength=0.0, seed=345)\n"
            "surv = data.survivors_only(rf, alive)\n"
            "res0 = st.compare_views(rf, surv, long_short=False, direction=-1, cost_bps=0.0)\n"
            "print(f\"survivors == full universe: {surv.shape[1]} of {rf.shape[1]} names kept\")\n"
            "print(f\"FULL  t={res0['test_full']['tstat']:+.4f}  \"\n"
            "      f\"SURVIVORS t={res0['test_survivors']['tstat']:+.4f}\")\n"
            "print(f\"delta Sharpe = {res0['delta_sharpe']:+.6f}  (must be ~0)\")\n"
            "assert abs(res0['delta_sharpe']) < 1e-9"
        ),
        md(
            "> 💡 In plain words: no firm delists, so 'survivors' *is* the whole universe, the "
            "two backtests are the same backtest, and the survivorship delta is exactly 0. The "
            "engine is honest — anything it finds later is the deletion, not a bug."
        ),
        md(
            "### 4b · The death-rate sweep — the bias is monotone\n\n"
            "Now turn deaths up. HAC *t* of the strategy on each view as the death rate climbs."
        ),
        code(
            "rates = [0.0, 0.15, 0.30, 0.45]\n"
            "ft, stt, dsh = [], [], []\n"
            "for dr in rates:\n"
            "    rf, alive, _ = data.synthetic_panel(n_assets=200, death_rate=dr,\n"
            "                                        signal_strength=0.0, seed=345)\n"
            "    surv = data.survivors_only(rf, alive)\n"
            "    res = st.compare_views(rf, surv, long_short=False, direction=-1, cost_bps=0.0)\n"
            "    ft.append(res['test_full']['tstat']); stt.append(res['test_survivors']['tstat'])\n"
            "    dsh.append(res['delta_sharpe'])\n"
            "x = [r*100 for r in rates]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(x, stt, 'o-', c=RED, lw=2.5, label='SURVIVORS')\n"
            "a1.plot(x, ft, 's-', c=GREEN, lw=2.5, label='FULL (honest)')\n"
            "a1.axhline(2, ls=':', c=GREY); a1.set_xlabel('% firms delisting')\n"
            "a1.set_ylabel('strategy HAC t'); a1.set_title('Honest t collapses, survivor t holds')\n"
            "a1.legend()\n"
            "a2.bar(x, dsh, width=6, color=GREY); a2.set_xlabel('% firms delisting')\n"
            "a2.set_ylabel('Sharpe(surv) - Sharpe(full)')\n"
            "a2.set_title('Survivorship delta grows with the death rate')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('FULL t     :', [f'{t:+.2f}' for t in ft])\n"
            "print('SURVIVORS t:', [f'{t:+.2f}' for t in stt])\n"
            "print('delta Sharpe:', [f'{d:+.2f}' for d in dsh])"
        ),
        md(
            f"> 💡 In plain words: the honest *t* decays {R['full_t'][0]:.2f} → "
            f"{R['full_t'][-1]:.2f} as more firms die (the strategy keeps buying bankruptcies), "
            f"while the survivors-only *t* stays near {R['surv_t'][-1]:.1f}. The Sharpe gap "
            f"climbs 0 → +{R['delta_sharpe'][-1]:.2f}. The bias is not a constant smudge — it "
            "scales with exactly how much of the truth you deleted."
        ),
        md(
            "### 4c · The headline + bootstrap — a manufactured 'discovery'\n\n"
            "At a realistic 30% deletion, the two reads with their block-bootstrap CIs."
        ),
        code(
            "rf, alive, _ = data.synthetic_panel(n_assets=200, death_rate=0.30,\n"
            "                                    signal_strength=0.0, seed=345)\n"
            "surv = data.survivors_only(rf, alive)\n"
            "res = st.compare_views(rf, surv, long_short=False, direction=-1, cost_bps=0.0)\n"
            "lof, hif = st.block_bootstrap_ci(res['full'], seed=345)\n"
            "los, his = st.block_bootstrap_ci(res['survivors'], seed=345)\n"
            "fm = res['test_full']['mean']; sm = res['test_survivors']['mean']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.errorbar([0], [fm*100], yerr=[[(fm-lof)*100],[(hif-fm)*100]], fmt='s', c=GREEN,\n"
            "            ms=10, capsize=6, label='FULL (honest)')\n"
            "ax.errorbar([1], [sm*100], yerr=[[(sm-los)*100],[(his-sm)*100]], fmt='o', c=RED,\n"
            "            ms=10, capsize=6, label='SURVIVORS')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks([0,1]); ax.set_xticklabels(['FULL','SURVIVORS'])\n"
            "ax.set_ylabel('mean monthly return (%)')\n"
            "ax.set_title('FULL CI straddles zero; SURVIVORS sits clear above it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"FULL : {res['test_full']['mean_ann']*100:+.2f}%/yr  t={res['test_full']['tstat']:+.2f}  \"\n"
            "      f\"95% CI/mo [{lof*100:+.2f}, {hif*100:+.2f}]%\")\n"
            "print(f\"SURV : {res['test_survivors']['mean_ann']*100:+.2f}%/yr  t={res['test_survivors']['tstat']:+.2f}  \"\n"
            "      f\"95% CI/mo [{los*100:+.2f}, {his*100:+.2f}]%\")"
        ),
        md(
            f"> 💡 In plain words: the honest FULL interval **straddles zero** "
            f"(*t* = {R['h_full_t']:.2f}); the survivors-only interval sits **clear above it** "
            f"(*t* = {R['h_surv_t']:.2f}). Same strategy, same underlying data — the only "
            "difference is whether the bankruptcies are in the sample. **H₃ confirmed:** the "
            "edge is manufactured, not inflated, from a true null."
        ),
        md(
            "### 4d · The real illustration — and why its *t* lies\n\n"
            "On a live large-cap universe (which is *itself* survivors-only — built from current "
            "membership), the same rule prints a juicy *t*. That number is the **symptom**, "
            "quoted to be distrusted."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = real_panel()\n"
            "    r = st.backtest(rets, lookback=12, frac=0.2, long_short=False, direction=-1)\n"
            "    s = st.summarize(r); t = st.hac_tstat(r)\n"
            "    sharpe, mean_ann, tstat, nobs = s['sharpe'], t['mean_ann']*100, t['tstat'], t['n']\n"
            "    fp = data.fingerprint(rets)\n"
            "else:\n"
            f"    sharpe, mean_ann, tstat, nobs, fp = ({R['real_sharpe']}, {R['real_mean']}, "
            f"{R['real_t']}, {R['real_months']}, '{R['real_fp']}')\n"
            "print(f'Real survivors-only large-cap tape (fingerprint {fp}, n={nobs}):')\n"
            "print(f'  buy-losers Sharpe={sharpe:+.2f}  mean={mean_ann:+.2f}%/yr  HAC t={tstat:+.2f}')\n"
            "print('  >> This universe has NO delisted names — the t is an UPPER BOUND, not evidence.')"
        ),
        md(
            f"> 💡 In plain words: the real tape shows *t* = {R['real_t']:.2f} — which, taken at "
            "face value, looks like a real edge. But this universe is exactly the trap: it "
            "contains only firms that survived to today, so the rule is rewarded for buying "
            "losers that *happened not to delist*. We can't put the dead names back here, so "
            "the synthetic core — where we can — is what carries the verdict. **A synthetic "
            "control proves the machinery; it never certifies a Signal**, and this real number "
            "is biased by construction. That is why Signal is `NONE`, not `WEAK`."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — honest FULL tape (30% deletion) earns {R['h_full_mean']:.2f}"
            f"%/yr at HAC *t* {R['h_full_t']:.2f} (< 2). No real edge. The synthetic positive "
            "control is a machinery proof, and the only real tape we have is itself "
            "survivorship-biased — neither can certify a signal.\n"
            f"- **Tradability `MIRAGE`** — the survivors-only edge ({R['h_surv_mean']:.2f}%/yr, "
            f"*t* {R['h_surv_t']:.2f}) is an artefact of deletion; restoring the dead names "
            "erases it.\n"
            f"- **Manufactures an edge? `CONFIRMED`** — Δ Sharpe +0.00 at 0% deaths (control), "
            f"rising to +{R['delta_sharpe'][-1]:.2f} at 45%; turns *t* {R['h_full_t']:.1f} into "
            f"*t* {R['h_surv_t']:.1f} from a true null. The bias is real, directional, and "
            "monotone in the deletion rate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "An edge that exists only because you deleted the bankruptcies is, by definition, "
            "not investable: the live experience *is* the FULL tape (you'd have held the losers "
            "that went to zero), and on the FULL tape the edge is absent. Costs don't even get "
            "a say — there's no gross edge for them to erode."
        ),
        code(
            "costs = [0.0, 10.0, 25.0, 50.0]\n"
            "rf, alive, _ = data.synthetic_panel(n_assets=200, death_rate=0.30,\n"
            "                                    signal_strength=0.0, seed=345)\n"
            "surv = data.survivors_only(rf, alive)\n"
            "ft = [st.hac_tstat(st.backtest(rf, long_short=False, direction=-1, cost_bps=c))['tstat']\n"
            "      for c in costs]\n"
            "stt = [st.hac_tstat(st.backtest(surv, long_short=False, direction=-1, cost_bps=c))['tstat']\n"
            "       for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, stt, 'o-', c=RED, lw=2, label='SURVIVORS')\n"
            "ax.plot(costs, ft, 's-', c=GREEN, lw=2, label='FULL (honest)')\n"
            "ax.axhline(2, ls=':', c=GREY); ax.set_xlabel('one-way cost (bps)')\n"
            "ax.set_ylabel('strategy HAC t'); ax.set_title('The honest tape never reaches t=2 at any cost')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('FULL t by cost     :', [f'{t:+.2f}' for t in ft])\n"
            "print('SURVIVORS t by cost:', [f'{t:+.2f}' for t in stt])"
        ),
        md(
            "> 💡 In plain words: the honest line sits below *t* = 2 at **every** cost level — "
            "there is no break-even to find because there is no edge. The only way to make this "
            "strategy 'work' is to keep deleting the dead, which is not a trade, it's a "
            "**measurement error**. The fix is the method: point-in-time universes, delisting "
            "returns booked."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Inflation vs manufacture.** Re-run with `signal_strength > 0` (a real planted "
            "edge): now survivorship *inflates* a true effect instead of inventing one. Both "
            "are dangerous; the inflation case is harder to catch because the honest tape also "
            "looks good.\n"
            "- **Gentler deaths.** Swap the −80% wipe for a −40% delisting loss, or merge firms "
            "away with no loss (acquisitions). The bias shrinks but never reverses for a "
            "long-biased rule.\n"
            "- **Point-in-time fix.** Fork this with a real PIT membership source (delisting "
            "returns booked à la Shumway 1997) and quantify how much of a published anomaly "
            "survives. That's the method this whole study is arguing for.\n\n"
            "*Every backtest on 'the index, projected back' is running this experiment without "
            "knowing it — on the SURVIVORS tape only. Fork this, plant your strategy, and find "
            "out how much of your 'alpha' is the graveyard you never opened.*"
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
