"""Generate the two narrative notebooks for Study 840 (Clustered Standard Errors).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

This is a synthetic-only method demo: every figure runs offline and deterministic on the
seed-840 null panel — there is NO real-tape cell (real free data can never certify "zero
slope"), so the study is capped at NONE. The dict ``R`` below mirrors the headline numbers in
docs/results.md (the full 2,000-rep run); the live cells recompute a fast reduced-rep version
(so execution stays well under a minute) and the two agree to sampling error.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; config fp a271c7ebce63;
# null-panel fp 607a6862117f; 2,000 reps x 50 periods x 50 firms; rho_x = rho_e = 0.5; seed 840).
R = dict(
    cfg_fp="a271c7ebce63", panel_fp="607a6862117f",
    n_reps=2000, T=50, N=50, rho_x=0.5, rho_e=0.5, beta=0.06,
    nominal=0.05, true_sd=0.07525, b_mean=-0.00302, true_sd_fm=0.02038, b_fm_mean=0.00041,
    moulton=3.640,
    ols_fp=0.604, ols_se_ratio=0.266, ols_t_sd=3.785,
    firm_fp=0.662, firm_se_ratio=0.230, firm_t_sd=4.445,
    time_fp=0.074, time_se_ratio=0.933, time_t_sd=1.116,
    fm_fp=0.053, fm_se_ratio=1.003, fm_t_sd=1.013,
    ctl_ols_fp=0.0495, ctl_firm_fp=0.054, ctl_time_fp=0.0475, ctl_fm_fp=0.053,
    ctl_naive_t_sd=0.991,
    # rho_e sweep: (rho_e, ols_fp, firm_fp, time_fp, fm_fp, naive_t_sd, moulton)
    sweep_rho=[(0.0, 0.046, 0.051, 0.057, 0.053, 0.997, 1.000),
               (0.2, 0.435, 0.464, 0.074, 0.053, 2.525, 2.429),
               (0.4, 0.559, 0.605, 0.073, 0.053, 3.415, 3.286),
               (0.6, 0.639, 0.695, 0.075, 0.053, 4.125, 3.962),
               (0.8, 0.679, 0.750, 0.081, 0.053, 4.744, 4.539)],
    # N sweep: (N, ols_fp, firm_fp, time_fp, fm_fp, naive_t_sd, moulton)
    sweep_N=[(2, 0.085, 0.398, 0.068, 0.025, 1.140, 1.118),
             (5, 0.158, 0.291, 0.059, 0.059, 1.376, 1.414),
             (10, 0.279, 0.387, 0.073, 0.060, 1.847, 1.803),
             (25, 0.464, 0.542, 0.078, 0.045, 2.675, 2.646),
             (50, 0.604, 0.661, 0.073, 0.053, 3.785, 3.640),
             (100, 0.696, 0.733, 0.065, 0.051, 5.100, 5.074)],
    fm_power=0.840, fm_t_mean=2.987, fm_sign=0.997, fm_b_recovered=0.0604,
    timer_gross=-3.142, timer_net=-10.169, timer_ann=-25.62,
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

from clustered_se import data, strategy as st

# The whole demo runs on the deterministic, offline seed-840 null panel (common time effect,
# TRUE slope = 0). This is a synthetic-only method demo — there is no real-tape cell by design.
# Live cells use a reduced-rep panel so execution is quick; the frozen R dict below carries the
# full 2,000-rep headline (docs/results.md). LIVE = fast reduced run; R = frozen headline.
LIVE_REPS, LIVE_T, LIVE_N = 600, 50, 50
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Clustered Standard Errors — when a 'significant' predictor is pure noise 🧷\n"
            "### Why firms in the same month aren't independent, and how a shared shock fakes a *t*-stat out of nothing\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_dependence_fake_significance%3F: Confirmed](https://img.shields.io/badge/Does_dependence_fake_significance%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Here is a trap that has fooled a *lot* of published finance research. You have a panel: "
            "hundreds of firms, dozens of months, one candidate predictor `x`. You pool every "
            "firm-month into one big regression of returns `y` on `x`, and out pops a *t*-statistic "
            "of 3. Publishable! Except `x` is **pure noise** — it has *zero* true relationship with "
            "`y`. What went wrong?\n\n"
            "The answer is **cross-sectional dependence**: in any given month, a common shock (the "
            "market, a macro surprise) moves *every* firm together, so your thousands of "
            "observations are nowhere near independent. The pooled regression doesn't know that, "
            "counts them as independent, and hands you a standard error that is far too small — and "
            "a *t*-stat far too big.\n\n"
            "> 📓 **This is the plain-language layer.** Want the four standard errors, the Moulton "
            "inflation formula and the Monte-Carlo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it, on a deterministic synthetic panel. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| On a *noise* predictor (true effect **zero**), how often does the naive pooled *t* "
            f"cry 'significant'? | **{R['ols_fp']*100:.0f}% of the time** — twelve times the 5% it "
            "advertises. |\n"
            f"| Does clustering by *firm* fix it? | **No — it's worse ({R['firm_fp']*100:.0f}%).** "
            "The dependence is *across firms in a month*, not within a firm over time; firm-"
            "clustering catches the wrong thing. |\n"
            f"| Does **Fama-MacBeth** fix it? | **Yes.** Its false-positive rate is "
            f"**{R['fm_fp']*100:.1f}%** — right at nominal. |\n"
            f"| Is the naive result real? | **No.** By construction there is *nothing there*; the "
            "*t*-stat is a standard-error illusion. |\n\n"
            "> A pooled *t* > 2 from a cross-sectional regression, without the right clustering, is "
            "nearly worthless — exactly the way a suspiciously smooth track record is."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Regress returns on a predictor across a big firm-month panel; a *t*-stat above 2 "
            "means the predictor works.\"*\n\n"
            "Petersen (2009) showed this is often false, because two kinds of correlation lurk in "
            "panel residuals: a **firm effect** (a firm is correlated with its own past) and a "
            "**time effect** (all firms move together in a given period). This study is about the "
            "**time effect** — the common shock — and how badly it fools the ordinary standard "
            "error."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because a huge fraction of the asset-pricing 'factor zoo' rests on exactly this kind of "
            "panel regression. If a shared monthly shock can manufacture a *t* of 3 from a predictor "
            "with **no** real content, then a naked *t*-stat proves nothing. The desk has shown the "
            "same lie told two other ways — by *trying many hypotheses* "
            "([346 multiple-testing](../../346-multiple-testing/)) and by *autocorrelation over time* "
            "in a single series ([838 hac-necessity](../../838-hac-necessity/)). This is the "
            "**cross-sectional** version."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We build a panel we *know* is empty, so any 'significance' must be fake. Each period we "
            "draw one **shared** shock for the predictor and one **shared** shock for the outcome "
            "(they are drawn *independently*, so `x` truly says nothing about `y`), and add "
            "firm-specific noise on top. Set the true slope to **zero**. Then we do the pooled "
            "regression 600 times and count how often each method's *t*-stat clears 2.\n\n"
            "If a method rejects far more than 5% of the time on this empty panel, its standard error "
            "is broken."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually run it\n\n"
            "Four ways to put a standard error on the *same* pooled slope: **naive** (pretend "
            "independence), **cluster by firm**, **cluster by time**, and **Fama-MacBeth** (a "
            "separate cross-sectional regression each period, then average). Watch which ones cry "
            "wolf."
        ),
        code(
            "X, Y = data.panel(LIVE_REPS, LIVE_T, LIVE_N, rho_x=0.5, rho_e=0.5, beta=0.0, seed=840)\n"
            "c = st.calibration(X, Y, crit=1.96)\n"
            "labels = ['naive\\nOLS', 'cluster\\nby FIRM', 'cluster\\nby TIME', 'Fama-\\nMacBeth']\n"
            "keys = ['ols', 'firm', 'time', 'fm']\n"
            "fp = [c[k+'_fp'] for k in keys]\n"
            "colors = [RED, RED, GREEN, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.6))\n"
            "ax.bar(labels, [x*100 for x in fp], color=colors, width=.6)\n"
            "ax.axhline(5, ls='--', c=GREY, lw=1.5, label='nominal 5%')\n"
            "ax.set_ylabel('false-positive rate (%)')\n"
            "ax.set_title('On a NOISE predictor: naive & firm-clustering cry wolf; time & Fama-MacBeth do not')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for lab, k in zip(['naive OLS','firm cluster','time cluster','Fama-MacBeth'], keys):\n"
            "    print(f'{lab:14s} false-positive rate {c[k+\"_fp\"]*100:5.1f}%  (SE is {c[k+\"_se_ratio\"]:.2f}x the truth)')"
        ),
        md(
            f"There it is. The **naive** pooled *t* fires on ~{R['ols_fp']*100:.0f}% of empty panels, "
            f"and clustering by **firm** is no better (~{R['firm_fp']*100:.0f}%) — it's guarding the "
            "wrong door. Cluster by **time**, or use **Fama-MacBeth**, and the false-positive rate "
            f"drops back to ~5%. The naive standard error is only about a **quarter** the size of the "
            "estimate's real swing — so of course the *t*-stat looks huge."
        ),
        md(
            "**How bad does it get?** The more of each period's variation comes from the shared shock "
            "(higher ρ), and the more firms per period (bigger N), the worse the naive test — while "
            "Fama-MacBeth stays flat at 5%:"
        ),
        code(
            "sweep = R_SWEEP  # frozen full-run rho_e sweep (docs/results.md)\n"
            "rho = [s[0] for s in sweep]; ols = [s[1]*100 for s in sweep]; fm = [s[4]*100 for s in sweep]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.plot(rho, ols, 'o-', c=RED, lw=2, label='naive OLS false-positive rate')\n"
            "ax.plot(rho, fm, 's-', c=GREEN, lw=2, label='Fama-MacBeth false-positive rate')\n"
            "ax.axhline(5, ls='--', c=GREY, lw=1)\n"
            "ax.set_xlabel('how correlated firms are within a period (rho_e ->)')\n"
            "ax.set_ylabel('false-positive rate (%)')\n"
            "ax.set_title('More shared shock = a bigger naive lie; Fama-MacBeth is immune')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for s in sweep:\n"
            "    print(f'rho_e {s[0]:.1f}: naive {s[1]*100:5.1f}%   Fama-MacBeth {s[4]*100:4.1f}%')",
        ),
        md(
            f"At a realistic ρ = 0.5 with 50 firms a month, the naive test is wrong "
            f"~{R['ols_fp']*100:.0f}% of the time. This is exactly why cross-sectional asset-pricing "
            "studies report **Fama-MacBeth** *t*-stats, not pooled OLS ones."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The panel has a true slope of zero — there is nothing to find. "
            "Fama-MacBeth agrees (5% rejection).\n"
            "- **Tradability — Mirage.** You can't trade a standard-error artefact; costed, the null "
            "loses money.\n"
            "- **Does cross-sectional dependence fake significance? — Confirmed.** Yes — the naive "
            f"*t* cries wolf {R['ols_fp']*100:.0f}% of the time, and clustering on the wrong "
            "dimension makes it worse. Only time-clustering / Fama-MacBeth tell the truth."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually 'trade' it?\n\n"
            "No. There is no predictor here — `x` is noise. If you built the dollar-neutral "
            "long-short the naive *t*-stat tempts you into, its gross return is indistinguishable "
            "from zero, and once you pay to trade it every period it bleeds ~25%/yr. A broken "
            "standard error is not a strategy."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The two cousins.** [838 HAC Necessity](../../838-hac-necessity/) is the same disease "
            "along the *time* axis (a single serially-correlated series, fixed by Newey-West); "
            "[346 multiple-testing](../../346-multiple-testing/) is the *how-many-hypotheses* haircut. "
            "This study is the *cross-sectional* axis.\n"
            "- **The fix is standard.** Fama-MacBeth (1973) and two-way clustering "
            "(Cameron-Gelbach-Miller 2011) — see the quants notebook for the mechanics.\n\n"
            "*Think your panel regression's *t* > 2 is real? Fork this, drop in your own predictor, "
            "and compare the naive *t* to the Fama-MacBeth one — the gap is the lie.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    # inject the frozen sweep as a module-level variable the sweep cell reads
    _inject_R(nb)
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Clustered Standard Errors — a quantitative teardown 🔬\n"
            "### four SEs for one slope · the Monte-Carlo false-positive experiment · the Moulton √(1+(N−1)ρ_xρ_e) identity · why firm-clustering fails · the no-dependence control · the planted-effect power check\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_dependence_fake_significance%3F: Confirmed](https://img.shields.io/badge/Does_dependence_fake_significance%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a null panel with a "
            "common time effect and **zero slope**, attach four standard errors to the pooled OLS "
            "estimate, and measure which ones are calibrated.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the panel is built to have "
            f"zero slope (config fp `{R['cfg_fp']}`, null-panel fp `{R['panel_fp']}`), so real free "
            "data can never certify it and the study is capped at `NONE`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Constructed null (β = 0, pooled estimate mean {R['b_mean']:+.4f}); "
            f"Fama-MacBeth FP {R['fm_fp']*100:.1f}% ≈ nominal. Synthetic-only demo (no real tape). |\n"
            f"| **Tradability** | `MIRAGE` | Notional long-short on the noise predictor: gross "
            f"{R['timer_gross']:+.1f} bps → net {R['timer_net']:+.1f} bps ({R['timer_ann']:+.1f}%/yr). |\n"
            f"| **Does dependence fake significance?** | `CONFIRMED` | Naive OLS FP "
            f"**{R['ols_fp']*100:.0f}%** (SE only {R['ols_se_ratio']:.2f}× the truth); firm-cluster "
            f"**{R['firm_fp']*100:.0f}%** (worse); naive-*t* SD {R['ols_t_sd']:.2f} ≈ Moulton "
            f"{R['moulton']:.2f}; Fama-MacBeth restores {R['fm_fp']*100:.1f}%. |\n\n"
            "> 💡 In plain words: the pooled point estimate is fine; the *standard error* is the lie, "
            "and clustering must match the dimension of the dependence."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Pool a panel and regress $y_{it} = a + b\\,x_{it} + e_{it}$. The OLS slope variance is "
            "$\\widehat{\\mathrm{Var}}(\\hat b) = s^2 / S_{xx}$, which is correct **only if the "
            "$e_{it}$ are independent**. Introduce a common time factor so "
            "$x_{it} = \\sqrt{\\rho_x}\\,f_t + \\sqrt{1-\\rho_x}\\,u_{it}$ and "
            "$e_{it} = \\sqrt{\\rho_e}\\,g_t + \\sqrt{1-\\rho_e}\\,v_{it}$ (with $f_t \\perp g_t$, so "
            "$b = 0$ in truth). Now:\n\n"
            "- **H₁ (the bias).** The true $\\mathrm{Var}(\\hat b)$ exceeds the naive one by the "
            "**Moulton factor** $\\tau = 1 + (N-1)\\rho_x\\rho_e$, so the naive *t* is inflated by "
            "$\\sqrt{\\tau}$.\n"
            "- **H₂ (wrong-dimension clustering).** Clustering by **firm** addresses "
            "*within-firm-over-time* correlation — which this design does **not** have — so it does "
            "not shrink the bias.\n"
            "- **H₃ (the fix).** Fama-MacBeth (a cross-sectional regression each period, then the "
            "time-series SE of the $T$ slopes) and **time** clustering are robust to arbitrary "
            "within-period correlation.\n\n"
            "We **confirm H₁** (naive-*t* SD lands on $\\sqrt{\\tau}$), **confirm H₂** (firm FP stays "
            "high), and **confirm H₃** (FM / time FP ≈ 5%)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "The naive SE divides by the *count* of observations $NT$, as if each were fresh "
            "information. But when $N$ firms share one shock $g_t$, a period contributes far less "
            "than $N$ independent draws — closer to *one*. The Moulton factor "
            "$\\sqrt{1+(N-1)\\rho_x\\rho_e}$ is exactly the ratio of the honest information content to "
            "the naive count. This is the **cross-sectional** sibling of "
            "[838](../../838-hac-necessity/)'s $\\sqrt{K}$ overlap inflation (there: correlation "
            "across *time*; here: across *firms*) — same sandwich-estimator cure, different clustering "
            "dimension. Petersen (2009) is the canonical demonstration; Fama-MacBeth (1973) the "
            "canonical fix."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Panel.** 2,000 reps × 50 periods × 50 firms; regressor and residual each split "
            "50/50 between a shared per-period factor and idiosyncratic noise (ρ_x = ρ_e = 0.5); "
            "**true slope zero**. The two factors are independent, so the estimate is unbiased — the "
            "damage is pure standard error.\n"
            "- **Four SEs, one slope.** naive i.i.d. OLS; one-way firm cluster; one-way time cluster; "
            "Fama-MacBeth (with the finite-sample cluster scaling $G/(G-1)\\cdot(NT-1)/(NT-2)$).\n"
            "- **The truth.** The Monte-Carlo SD of the pooled slope across reps — a calibrated SE "
            "must equal it (ratio ≈ 1).\n"
            "- **Inference.** False-positive rate at |*t*| > 1.96 with a Wilson band; the naive-*t* SD "
            "matched to the closed-form Moulton factor; a ρ = 0 no-dependence control.\n"
            "- **Positive control.** Plant β = 0.06 and confirm Fama-MacBeth's power and sign.\n\n"
            "Live cells recompute a reduced-rep version; the frozen `R` dict carries the full "
            "2,000-rep headline (they agree to sampling error)."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Four standard errors for one slope — H₁ and H₂\n\n"
            "The centrepiece: on the null panel, how big is each SE relative to the estimate's *true* "
            "sampling variability, and how often does each *t* reject?"
        ),
        code(
            "X, Y = data.panel(LIVE_REPS, LIVE_T, LIVE_N, rho_x=0.5, rho_e=0.5, beta=0.0, seed=840)\n"
            "c = st.calibration(X, Y, crit=1.96)\n"
            "keys = ['ols', 'firm', 'time', 'fm']\n"
            "names = ['naive OLS', 'firm cluster', 'time cluster', 'Fama-MacBeth']\n"
            "print(f\"pooled slope: mean {c['b_mean']:+.4f} (unbiased), TRUE SD {c['true_sd']:.4f}\\n\")\n"
            "print(f\"{'estimator':14s}{'FP rate':>10s}{'SE/true':>10s}{'t SD':>8s}\")\n"
            "for nm, k in zip(names, keys):\n"
            "    print(f'{nm:14s}{c[k+\"_fp\"]*100:9.1f}%{c[k+\"_se_ratio\"]:10.3f}{c[k+\"_t_sd\"]:8.2f}')\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "cols = [RED, RED, GREEN, GREEN]\n"
            "a1.bar(names, [c[k+'_fp']*100 for k in keys], color=cols, width=.6)\n"
            "a1.axhline(5, ls='--', c=GREY, lw=1.5); a1.set_ylabel('false-positive rate (%)')\n"
            "a1.set_title('FP rate (nominal 5%)'); a1.tick_params(axis='x', rotation=20)\n"
            "a2.bar(names, [c[k+'_se_ratio'] for k in keys], color=cols, width=.6)\n"
            "a2.axhline(1.0, ls='--', c=GREY, lw=1.5); a2.set_ylabel('mean SE / true SD')\n"
            "a2.set_title('SE calibration (1.0 = honest)'); a2.tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **H₁ confirmed** — the naive SE is only ~{R['ols_se_ratio']:.2f}× "
            f"the estimate's true swing, so it over-rejects ~{R['ols_fp']*100:.0f}% of the time. "
            f"**H₂ confirmed** — firm clustering (SE ratio ~{R['firm_se_ratio']:.2f}) is no better "
            "and here slightly worse; it guards the wrong dimension. Time clustering and "
            f"Fama-MacBeth sit at SE ratio ~1 and FP ~5%."
        ),
        md(
            "### 4b · The inflation identity — H₁ against the Moulton closed form\n\n"
            "Under the null the naive-*t* has SD $\\sqrt{\\tau} = \\sqrt{1+(N-1)\\rho_x\\rho_e}$ "
            "instead of 1. Sweep the residual correlation ρ_e and the cross-section size N and match "
            "the empirical naive-*t* SD to the formula."
        ),
        code(
            "sr = np.array(R_SWEEP)   # frozen rho_e sweep: cols = rho_e, ols_fp, firm_fp, time_fp, fm_fp, naive_t_sd, moulton\n"
            "sn = np.array(R_SWEEPN)  # frozen N sweep\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(sr[:,0], sr[:,5], 'o-', c=RED, lw=2, label='empirical naive-t SD')\n"
            "a1.plot(sr[:,0], sr[:,6], 'k--', lw=1.5, label='Moulton sqrt(1+(N-1)rho_x rho_e)')\n"
            "a1.set_xlabel('residual intra-period corr rho_e'); a1.set_ylabel('naive-t SD (inflation)')\n"
            "a1.set_title('Inflation vs rho_e (N=50)'); a1.legend()\n"
            "a2.plot(sn[:,0], sn[:,5], 'o-', c=RED, lw=2, label='empirical naive-t SD')\n"
            "a2.plot(sn[:,0], sn[:,6], 'k--', lw=1.5, label='Moulton closed form')\n"
            "a2.set_xlabel('firms per period N'); a2.set_ylabel('naive-t SD (inflation)')\n"
            "a2.set_title('Inflation vs N (rho=0.5)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('rho_e:  empirical vs Moulton'); [print(f'  {r[0]:.1f}: {r[5]:.2f} vs {r[6]:.2f}') for r in sr]\n"
            "print('N:      empirical vs Moulton'); [print(f'  {int(r[0]):3d}: {r[5]:.2f} vs {r[6]:.2f}') for r in sn]"
        ),
        md(
            f"> 💡 In plain words: **H₁ confirmed to two figures.** The naive-*t* SD rides the Moulton "
            f"curve as both dials turn — at the headline (N=50, ρ=0.5) it is {R['ols_t_sd']:.2f} "
            f"against the closed form {R['moulton']:.2f}. The small consistent excess shrinks as T "
            "grows (Moulton is a large-T approximation)."
        ),
        md(
            "### 4c · The control — switch the common factor OFF (H₁'s cause)\n\n"
            "Set ρ_x = ρ_e = 0: no shared shock, an i.i.d. panel. Same generator, same estimators — "
            "the pitfall must vanish."
        ),
        code(
            "Xc, Yc = data.panel(LIVE_REPS, LIVE_T, LIVE_N, rho_x=0.0, rho_e=0.0, beta=0.0, seed=840)\n"
            "cc = st.calibration(Xc, Yc, crit=1.96)\n"
            "for nm, k in zip(['naive OLS','firm','time','Fama-MacBeth'], ['ols','firm','time','fm']):\n"
            "    print(f'{nm:14s} FP {cc[k+\"_fp\"]*100:4.1f}%')\n"
            "print(f'naive-t SD {cc[\"ols_t_sd\"]:.3f} (~1: no inflation)')"
        ),
        md(
            f"> 💡 In plain words: with no dependence *all four* estimators are calibrated (~5%) and "
            f"the naive-*t* SD is ~1 (frozen full run: {R['ctl_naive_t_sd']:.2f}). The pitfall is "
            "*caused by* the common time factor and nothing else. (Row-shuffling an already-drawn "
            "panel is not a valid placebo — the pooled slope and its naive SE are invariant to "
            "relabelling; the cause must be removed at generation.)"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — constructed null (β = 0), Fama-MacBeth FP {R['fm_fp']*100:.1f}% ≈ "
            "nominal. Synthetic-only demo — no real tape, so never `REAL`.\n"
            f"- **Tradability `MIRAGE`** — a notional long-short on the noise predictor nets "
            f"{R['timer_net']:+.1f} bps/period ({R['timer_ann']:+.1f}%/yr).\n"
            f"- **Does cross-sectional dependence fake significance? `CONFIRMED`** — naive FP "
            f"{R['ols_fp']*100:.0f}%, firm-cluster worse, inflation = √(1+(N−1)ρ_xρ_e), Fama-MacBeth "
            "restores calibration."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the costed null\n\n"
            "The naive *t*-stat tempts you into a dollar-neutral long-short on `x`. Cost it."
        ),
        code(
            "Xo, Yo = data.one_panel(50, 50, rho_x=0.5, rho_e=0.5, beta=0.0, seed=840)\n"
            "tm = st.timer_stats(Xo, Yo, ret_scale=0.01, cost_bps=5.0, borrow_bps_yr=50.0)\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net (5bps+borrow)'], [tm['gross_bps'], tm['net_bps']], color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('bps per period')\n"
            "ax.set_title('A null minus costs is a guaranteed loser')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {tm['gross_bps']:+.2f} bps -> net {tm['net_bps']:+.2f} bps/period \"\n"
            "      f\"({tm['ann_net_pct']:+.1f}%/yr, net t {tm['t_net_naive']:+.1f})\")"
        ),
        md(
            "> 💡 In plain words: gross is indistinguishable from zero (there is no predictor), and "
            "costs turn it firmly negative. `MIRAGE` by construction."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control (does the fix have power?)\n\n"
            "Is Fama-MacBeth just numb, or does it reward a **real** slope? Plant β = 0.06 and check "
            "it fires, recovers the magnitude, and gets the sign right."
        ),
        code(
            "Xe, Ye = data.panel(LIVE_REPS, LIVE_T, LIVE_N, rho_x=0.5, rho_e=0.5, beta=0.06, seed=840)\n"
            "inf = st.panel_inference(Xe, Ye)\n"
            "power = float(np.mean(np.abs(inf['t_fm']) > 1.96))\n"
            "print(f\"Fama-MacBeth power        : {power*100:.1f}%\")\n"
            "print(f\"mean FM t                 : {inf['t_fm'].mean():+.2f}\")\n"
            "print(f\"share FM t > 0 (right sign): {np.mean(inf['t_fm']>0)*100:.1f}%\")\n"
            "print(f\"mean FM slope (beta=0.06) : {inf['b_fm'].mean():+.4f}\")\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.hist(inf['t_fm'], bins=40, color=GREEN, alpha=.8)\n"
            "ax.axvline(1.96, ls='--', c=GREY); ax.axvline(-1.96, ls='--', c=GREY)\n"
            "ax.set_xlabel('Fama-MacBeth t (planted beta = 0.06)'); ax.set_ylabel('count')\n"
            "ax.set_title('Hand it a real effect and Fama-MacBeth fires — unbiased, not merely conservative')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The Fama-MacBeth test rejects the null ~{R['fm_power']*100:.0f}% of the time on a "
            f"genuinely planted slope, recovers the magnitude ({R['fm_b_recovered']:.4f} vs 0.06), and "
            f"gets the sign right {R['fm_sign']*100:.1f}% of the time. So the correction removes the "
            "*false* positives without killing the *true* ones. For the time-axis version of this "
            "disease see [838 HAC Necessity](../../838-hac-necessity/); for the trial-count version, "
            "[346 multiple-testing](../../346-multiple-testing/)."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _inject_R(nb)
    _write(nb, "02_for_the_quants.ipynb")


def _inject_R(nb):
    """Append the frozen sweep arrays into the BOOT cell so the plotting cells can read them."""
    extra = (
        f"\nR_SWEEP = {R['sweep_rho']!r}\n"
        f"R_SWEEPN = {R['sweep_N']!r}\n"
    )
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and cell["source"].startswith("import sys, os"):
            cell["source"] = cell["source"] + extra
            break


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
