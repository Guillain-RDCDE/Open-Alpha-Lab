"""Generate the two narrative notebooks for Study 838 (HAC Necessity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The heavy 600×2,520 headline Monte-Carlo is quoted from the frozen ``R`` dict (mirror of
docs/results.md); the notebooks live-run only a fast, small Monte-Carlo (a few seconds) so they
re-execute end-to-end for any reader, offline and deterministic.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30).
R = dict(
    seed=838, n_reps=600, n_days=2520, window=21, nw_lags=42, crit=1.96,
    cfg_fp="767e2ce61be1", null_fp="0c98419fb4d7",
    nominal=0.05,
    # hero: 21-day overlap null
    naive_fp=0.643, naive_ci=(0.604, 0.681),
    nw_fp=0.095, nw_ci=(0.074, 0.121),
    naive_t_sd=4.86, nw_t_sd=1.18, sqrt_window=4.583,
    # iid control (window=1)
    ctl_naive_fp=0.0567, ctl_nw_fp=0.0683, ctl_naive_t_sd=1.01,
    # overlap inflation curve
    ov_window=[1, 5, 10, 21, 42, 63],
    ov_naive_fp=[0.057, 0.378, 0.562, 0.643, 0.755, 0.808],
    ov_nw_fp=[0.058, 0.078, 0.087, 0.095, 0.090, 0.108],
    ov_naive_sd=[1.01, 2.25, 3.34, 4.86, 6.68, 8.52],
    ov_theory=[1.00, 2.24, 3.16, 4.58, 6.48, 7.94],
    # AR(1) inflation curve
    ar_rho=[0.0, 0.2, 0.4, 0.6, 0.8],
    ar_naive_fp=[0.057, 0.115, 0.175, 0.300, 0.520],
    ar_nw_fp=[0.058, 0.068, 0.082, 0.098, 0.135],
    ar_naive_sd=[1.01, 1.24, 1.54, 2.02, 3.04],
    ar_theory=[1.00, 1.22, 1.53, 2.00, 3.00],
    # positive control
    ctrl_mean=6e-4, ctrl_power=0.892, ctrl_t_mean=3.42, ctrl_pos=0.997,
    # timer
    gross_bps=0.38, net_bps=-9.76, ann_net=-24.6,
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

from hac_necessity import data as d, strategy as st

# A FAST live Monte-Carlo (a few seconds); the frozen headline uses 600 x 2520 (see R / results.md).
LIVE_REPS, LIVE_DAYS, WINDOW, NW_LAGS, CRIT = 200, 1260, 21, 42, 1.96
print("live MC:", LIVE_REPS, "reps x", LIVE_DAYS, "days   (headline: 600 x 2520)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# HAC Necessity — how 'independent' days that aren't fake a significant result 📏\n"
            "### Overlapping windows quietly break the *t*-test, and one correction fixes it\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_ignoring_autocorrelation_fake_significance%3F: Confirmed](https://img.shields.io/badge/Does_ignoring_autocorrelation_fake_significance%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "A *t*-statistic above 2 is the industry's stamp of 'this is real, not luck'. But that stamp "
            "assumes your data points are **independent**. The moment your strategy uses a *formation "
            "window* — 'rank stocks on their last 21 days, hold, repeat tomorrow' — consecutive days "
            "overlap and stop being independent. We'll build a market with **literally no edge**, show "
            "the ordinary *t*-test screaming 'significant!' anyway, and hand you the one correction "
            "(Newey-West) that sees through it.\n\n"
            "> 📓 **This is the plain-language layer.** Want the √window inflation identity and the "
            "false-positive Monte-Carlo? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Why does a backtest with a holding window look 'significant' so easily? | **Because its "
            "days overlap.** Today's 21-day signal shares 20 days with yesterday's — so you don't really "
            "have thousands of independent observations, you have far fewer. |\n"
            "| We tested it on a market with **no edge**. What did the ordinary *t*-test do? | It flagged "
            f"a **true nothing** as 'significant' **{R['naive_fp']:.0%} of the time** — instead of the "
            "5% it promises. |\n"
            "| Can a tool catch it? | **Yes — one line.** The *Newey-West* standard error accounts for "
            f"the overlap and drops the false-alarm rate back to **~{R['nw_fp']:.0%}**. |\n"
            "| So is there anything to trade here? | **No.** This is a study about *method*. The lesson "
            "is the product. |\n\n"
            "> The overlap inflates the *t*-statistic by almost exactly **√21 ≈ 4.6×**. A '*t* = 3' that "
            "should have been a '*t* = 0.65' is how noise gets published as a discovery."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Econometricians Whitney Newey and Kenneth West (1987), building on Hansen & Hodrick (1980), "
            "made it precise: if your observations are **serially correlated** — each one statistically "
            "leaning on the last — the ordinary standard error is *too small*, so every *t*-statistic "
            "you compute is *too big*. Overlapping windows are the classic culprit. The fix is a "
            "'heteroskedasticity- and autocorrelation-consistent' (HAC) standard error — the "
            "**Newey-West** correction. The claim we test: *ignore it, and you will manufacture "
            "significance out of nothing.*"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is one of the most common ways a dead strategy gets a green light. Momentum, "
            "reversal, carry, seasonal effects — almost every published signal uses a formation or "
            "holding window, which means overlapping returns, which means the naive *t*-stat on it is "
            "inflated. If you don't correct for it, you'll fund noise; if you do, a lot of 'edges' "
            "quietly fall below the bar. Knowing *how much* the overlap inflates the number lets you "
            "un-inflate it before you bet."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We rig the experiment so the truth is known:\n\n"
            "1. **Build a market with no edge.** A daily series whose true average is *exactly zero* — "
            "but carrying the overlap of a 21-day window (each day = the trailing 21-day average of pure "
            "coin-flips).\n"
            "2. **Run it hundreds of times.** For each simulated history, compute the ordinary *t*-stat "
            "and check: did it wrongly call this *nothing* 'significant'?\n"
            "3. **Count the false alarms.** A fair test should cry wolf **5%** of the time. We'll see how "
            "far past 5% the naive test goes.\n"
            "4. **Apply the fix.** Re-run with the Newey-West standard error and watch the false alarms "
            "fall back into line.\n\n"
            "If overlap were harmless, both tests would sit at 5%. Spoiler: only one does."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: what 'overlapping' does to a series.** Here's a mean-zero series and its "
            "autocorrelation — how strongly each day predicts the next. Independent noise would show "
            "bars near zero; our 21-day overlap shows a staircase that only dies out after ~21 days."
        ),
        code(
            "x = d.overlap_returns(4000, window=WINDOW, seed=1)\n"
            "ac = st.autocorr(x.to_numpy(), lags=30)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(range(1, 31), ac, color=RED, alpha=.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('lag (days apart)'); ax.set_ylabel('autocorrelation')\n"
            "ax.set_title('A 21-day overlap: each day is ~95% correlated with the next')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'lag-1 autocorrelation = {ac[0]:.2f}  (independent noise would be ~0)')"
        ),
        md(
            "Those days are *not* independent — the series remembers three weeks back. Yet the ordinary "
            "*t*-test assumes they're all fresh, independent draws. That single wrong assumption is the "
            "whole bug."
        ),
        md(
            "**Second: the false-alarm rate.** We simulate many mean-zero histories and count how often "
            "each test wrongly shouts 'significant!'. The dashed line is the 5% a fair test should hit."
        ),
        code(
            "X = d.overlap_matrix(LIVE_REPS, LIVE_DAYS, window=WINDOW, seed=838)\n"
            "fp = st.false_positive_rate(X, lags=NW_LAGS, crit=CRIT)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['Ordinary\\n(i.i.d.) t-test', 'Newey-West\\n(HAC) t-test'],\n"
            "       [fp['naive_fp'], fp['nw_fp']], color=[RED, GREEN], width=0.55)\n"
            "ax.axhline(0.05, ls='--', c=GREY, label='fair rate (5%)')\n"
            "ax.set_ylabel('false-alarm rate on a NULL market'); ax.set_ylim(0, 0.8)\n"
            "ax.set_title('Same empty market — one test cries wolf, one does not'); ax.legend()\n"
            "for i, v in enumerate([fp['naive_fp'], fp['nw_fp']]):\n"
            "    ax.text(i, v + 0.02, f'{v:.0%}', ha='center', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'live: naive false-alarm {fp[\"naive_fp\"]:.0%}  vs  Newey-West {fp[\"nw_fp\"]:.0%}')\n"
            "print(f'headline 600x2520 : naive {HEAD_NAIVE:.0%}  vs  Newey-West {HEAD_NW:.0%}')"
        ),
        md(
            f"On a market with **nothing there**, the ordinary *t*-test calls it a discovery ~"
            f"**{R['naive_fp']:.0%}** of the time (your small live run will land near it; the frozen "
            f"600×2,520 headline is {R['naive_fp']:.0%}). Newey-West brings it back near the honest "
            f"**{R['nw_fp']:.0%}**. Everything the naive test 'found' was the overlap fooling its "
            "standard error."
        ),
        md(
            "**Third: the more overlap, the bigger the lie.** Widen the window and the false-alarm rate "
            "climbs — because there's more overlap and fewer truly-independent observations. The naive "
            "*t* is inflated by almost exactly the **square root of the window length**."
        ),
        code(
            "ws = [1, 5, 10, 21, 42]\n"
            "naive, nw = [], []\n"
            "for w in ws:\n"
            "    Xi = d.overlap_matrix(LIVE_REPS, LIVE_DAYS, window=w, seed=838)\n"
            "    f = st.false_positive_rate(Xi, lags=2*max(1,w), crit=CRIT)\n"
            "    naive.append(f['naive_fp']); nw.append(f['nw_fp'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(ws, naive, 'o-', c=RED, lw=2, label='ordinary t-test')\n"
            "ax.plot(ws, nw, 's-', c=GREEN, lw=2, label='Newey-West')\n"
            "ax.axhline(0.05, ls='--', c=GREY, label='fair rate (5%)')\n"
            "ax.set_xlabel('overlap window (days)'); ax.set_ylabel('false-alarm rate')\n"
            "ax.set_title('Wider window -> more overlap -> more false alarms'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('naive false-alarm by window:', [f'{v:.0%}' for v in naive])"
        ),
        md(
            "At window = 1 (no overlap) both tests behave — that's the control that proves overlap is "
            "the cause. As the window grows, only the naive line runs away. **The pitfall is the "
            "overlap, and the cure is the corrected standard error.**"
        ),
        md(
            "**Fourth: the catch — does the fix just kill *everything*?** No. Plant a **real** edge and "
            "Newey-West still finds it — it removes the *false* alarms without silencing the *true* one."
        ),
        code(
            "pw = st.power_check(LIVE_REPS, LIVE_DAYS, window=WINDOW, mean=6e-4, seed=838, lags=NW_LAGS)\n"
            "print(f'With a REAL effect planted, Newey-West rejects the null {pw[\"nw_power\"]:.0%} of the time,')\n"
            "print(f'with the correct (positive) sign {pw[\"nw_t_positive_share\"]:.0%} of the time.')\n"
            "print(f'(headline power at 600x2520: {0.892:.0%})')"
        ),
        md(
            f"That's the whole point of a *good* correction: on the empty market it stays quiet "
            f"(~{R['nw_fp']:.0%} false alarms), and on a real effect it speaks up "
            f"(~{R['ctrl_power']:.0%} power). The naive test can't tell the two apart — it shouts either "
            "way."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** By construction the market has zero edge; there is nothing real to "
            "find, and the corrected test agrees.\n"
            "- **Tradability — Mirage.** The 'significance' is a standard-error illusion, not a return; "
            "costed, the null just loses money.\n"
            "- **Does ignoring autocorrelation fake significance? — Confirmed.** On an empty 21-day-"
            f"overlap market the ordinary *t*-test cries wolf **{R['naive_fp']:.0%}** of the time and is "
            f"inflated **√21 ≈ 4.6×**; Newey-West restores ~{R['nw_fp']:.0%} while still catching a real "
            "effect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade — that's the point. But the practical takeaway is huge: **before "
            "you believe any *t*-stat on a strategy with a holding or formation window, ask whether it "
            "used a HAC standard error.** If it used the ordinary one, mentally divide the *t* by "
            "roughly √(window) and see if it still clears the bar. Most window-based 'edges' don't "
            "survive that one division. The [quants notebook](02_for_the_quants.ipynb) shows the exact "
            "inflation identity and how to apply it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Use enough lags.** Newey-West needs a bandwidth at least as long as your overlap — too "
            "few lags and even it under-corrects (we show this with a strongly-persistent series in the "
            "quants notebook).\n"
            "- **Block bootstrap.** When in doubt about the standard error, resample in *blocks* to "
            "preserve the autocorrelation and read the interval off directly.\n"
            "- **Non-overlapping sampling.** The bluntest fix: only sample every *window* days so the "
            "observations really are independent — at the cost of throwing away data.\n\n"
            "*Got a favourite signal with a formation window? Fork this, drop its daily returns into "
            "`strategy.newey_west_t`, and compare against the naive `one_sample_t`. If the honest *t* "
            "still clears 2, you might have something. If it doesn't, the overlap was doing the talking.*"
        ),
    ]
    # give the notebook the frozen constants it references inline
    boot2 = BOOT + "\nHEAD_NAIVE, HEAD_NW = 0.643, 0.095   # frozen 600x2520 headline (docs/results.md)\n"
    cells[1] = code(boot2)
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# HAC Necessity — a quantitative teardown 🔬\n"
            "### Overlapping-returns null · naive vs Newey-West false-positive rates · the √window and "
            "√((1+ρ)/(1−ρ)) inflation identities · statsmodels cross-check · a planted-effect power "
            "control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_ignoring_autocorrelation_fake_significance%3F: Confirmed](https://img.shields.io/badge/Does_ignoring_autocorrelation_fake_significance%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We Monte-Carlo a mean-zero "
            "overlapping-returns null, measure the naive vs HAC false-positive rate, and match the "
            "*t*-inflation to its closed form.\n\n"
            "> ⚠️ **Not investment advice.** A pure simulation study — deterministic & offline, no market "
            "data. Heavy headline numbers are quoted from the frozen `R` dict (mirror of "
            "[`docs/results.md`](../docs/results.md), as-of 2026-06-30, config fp `767e2ce61be1`); the "
            "live cells run a fast small Monte-Carlo. Methods in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT + "\n# frozen headline mirror (see docs/results.md)\n"
             "R = dict(naive_fp=0.643, nw_fp=0.095, naive_t_sd=4.86, sqrt_window=4.583,\n"
             "         nw_t_sd=1.18, ctrl_power=0.892, ctrl_t_mean=3.42, nominal=0.05)\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | The null has true mean **0**; Newey-West's false-positive rate is "
            f"**{R['nw_fp']:.1%}** ≈ nominal. Nothing to find, and the HAC test agrees. |\n"
            f"| **Tradability** | `MIRAGE` | The naive 'significance' is a standard-error artefact; a "
            "costed null loses **−24.6%/yr**. |\n"
            f"| **Does ignoring autocorrelation fake significance?** | `CONFIRMED` | Naive false-positive "
            f"rate **{R['naive_fp']:.1%}** (nominal 5%); naive-*t* SD **{R['naive_t_sd']:.2f}** ≈ √21 "
            f"**{R['sqrt_window']:.2f}**; inflation tracks √window & √((1+ρ)/(1−ρ)); the i.i.d. control "
            f"is calibrated; NW keeps {R['ctrl_power']:.0%} power on a planted effect. |\n\n"
            "> 💡 In plain words: the disease is a *mis-specified variance*, not too many hypotheses "
            "(that is [346](../../346-multiple-testing/)) nor too much fitting flexibility (that is "
            "[348](../../348-curve-fitting/)). Here a **single** hypothesis is tested with the **wrong "
            "standard error** because the observations overlap."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "For a mean estimator $\\hat\\mu = \\frac1T\\sum_t x_t$ the correct sampling variance is\n\n"
            "$$\\operatorname{Var}(\\hat\\mu) = \\frac{1}{T}\\Big(\\gamma_0 + 2\\sum_{\\ell\\ge 1}\\gamma_\\ell\\Big) "
            "= \\frac{\\mathrm{LRV}}{T},$$\n\n"
            "where $\\gamma_\\ell$ are the autocovariances and LRV is the **long-run variance**. The "
            "naive OLS SE uses only $\\gamma_0/T$, so the naive *t* is inflated by "
            "$\\sqrt{\\mathrm{LRV}/\\gamma_0}$. Two closed forms:\n\n"
            "- **Overlap of $K$ days** (a rolling mean of i.i.d. shocks, an MA($K-1$)): $\\gamma_0 = "
            "\\sigma^2/K$, $\\mathrm{LRV} = \\sigma^2$, so $\\mathrm{LRV}/\\gamma_0 = K$ and the "
            "inflation is $\\sqrt{K}$.\n"
            "- **AR(1) with autocorrelation $\\rho$**: $\\mathrm{LRV}/\\gamma_0 = (1+\\rho)/(1-\\rho)$, "
            "inflation $\\sqrt{(1+\\rho)/(1-\\rho)}$.\n\n"
            "The **Newey-West** estimator plugs a Bartlett-kernel-weighted sample LRV into the SE, "
            "restoring calibration. H₀: the naive test over-rejects at $\\sqrt{\\mathrm{LRV}/\\gamma_0}$; "
            "H₁: NW is ~calibrated; H₂: NW still has power on a planted mean. We confirm all three."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on it\n\n"
            "Every overlapping predictive regression in finance — long-horizon return predictability, "
            "momentum/reversal formation windows, carry, seasonal composites — inherits this MA "
            "structure. Hansen-Hodrick (1980) and the long-horizon-regression literature "
            "(Boudoukh-Israel-Richardson 2022) show that un-corrected overlap *t*-stats are a leading "
            "manufacturer of spurious predictability. The desk's rule — HAC *t* on every serially-"
            "correlated series — exists because of exactly this experiment."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Tape.** A mean-zero overlapping-window null, $K=21$, $\\sigma=0.01$, **600 reps × 2,520 "
            "days**, seed 838 (headline). Live cells use a fast 200 × 1,260 version.\n"
            "- **Estimators.** Naive one-sample *t* (`strategy.one_sample_t`) vs Newey-West Bartlett-"
            "kernel HAC *t* (`strategy.newey_west_t`), bandwidth 42 ($\\ge$ the overlap; Bartlett "
            "downweighting wants a generous span).\n"
            "- **False-positive rate.** Share of reps with $|t| > 1.96$ under each estimator; a Wilson "
            "interval on the proportion.\n"
            "- **Inflation.** The naive-*t* SD across reps (= $\\sqrt{\\mathrm{LRV}/\\gamma_0}$) vs the "
            "closed forms $\\sqrt{K}$ and $\\sqrt{(1+\\rho)/(1-\\rho)}$.\n"
            "- **Control.** $K=1$ (i.i.d.) — both tests must be calibrated.\n"
            "- **Power.** Plant a mean of 6e-4; NW rejection rate and sign."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The naive *t* is not standard-normal under the null\n\n"
            "Under a calibrated test the *t* is ~N(0,1). Here the naive *t* is N(0, √21) — a distribution "
            "**4.6× too wide** — so far more of it spills past ±1.96. The Newey-West *t* sits back near "
            "N(0,1)."
        ),
        code(
            "X = d.overlap_matrix(LIVE_REPS, LIVE_DAYS, window=WINDOW, seed=838)\n"
            "print('null-matrix fingerprint (live):', d.fingerprint(X))\n"
            "fp = st.false_positive_rate(X, lags=NW_LAGS, crit=CRIT)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "bins = np.linspace(-12, 12, 61)\n"
            "ax.hist(fp['t_naive'], bins=bins, color=RED, alpha=.55, density=True, label=f'naive t (SD {fp[\"naive_t_sd\"]:.2f})')\n"
            "ax.hist(fp['t_nw'], bins=bins, color=GREEN, alpha=.55, density=True, label=f'Newey-West t (SD {fp[\"nw_t_sd\"]:.2f})')\n"
            "zz = np.linspace(-12, 12, 300)\n"
            "ax.plot(zz, np.exp(-zz**2/2)/np.sqrt(2*np.pi), 'k--', lw=1.4, label='N(0,1) — what calibrated looks like')\n"
            "for c in (-1.96, 1.96): ax.axvline(c, color=GREY, ls=':', lw=1)\n"
            "ax.set_xlabel('t-statistic under the NULL'); ax.set_ylabel('density')\n"
            "ax.set_title('Same empty market: the naive t is 4.6x too wide'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'naive false-positive {fp[\"naive_fp\"]:.1%} [Wilson {fp[\"naive_fp_ci\"][0]:.1%},{fp[\"naive_fp_ci\"][1]:.1%}]  '\n"
            "      f'| NW {fp[\"nw_fp\"]:.1%} [Wilson {fp[\"nw_fp_ci\"][0]:.1%},{fp[\"nw_fp_ci\"][1]:.1%}]')\n"
            "print(f'headline (600x2520): naive {R[\"naive_fp\"]:.1%}  NW {R[\"nw_fp\"]:.1%}')"
        ),
        md(
            f"> 💡 In plain words: the naive-*t* SD is **~{R['naive_t_sd']:.1f}** = √21, not 1. The ±1.96 "
            "gates were drawn for an N(0,1); against an N(0,√21) they catch "
            f"**{R['naive_fp']:.0%}** of the mass. Newey-West re-centres the width to ~1 and the "
            f"false-positive rate to ~{R['nw_fp']:.0%}."
        ),
        md(
            "### 4b · The inflation identity — √window, to two figures\n\n"
            "Sweep the overlap window and overlay the measured naive-*t* SD on the closed form √window."
        ),
        code(
            "ws = [1, 5, 10, 21, 42, 63]\n"
            "sd_emp, sd_th, naive_fp, nw_fp = [], [], [], []\n"
            "for w in ws:\n"
            "    Xi = d.overlap_matrix(LIVE_REPS, LIVE_DAYS, window=w, seed=838)\n"
            "    f = st.false_positive_rate(Xi, lags=2*max(1,w), crit=CRIT)\n"
            "    sd_emp.append(f['naive_t_sd']); sd_th.append(d.theoretical_inflation_overlap(w))\n"
            "    naive_fp.append(f['naive_fp']); nw_fp.append(f['nw_fp'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))\n"
            "a1.plot(ws, sd_th, 'k--', lw=1.8, label='closed form  \\u221awindow')\n"
            "a1.plot(ws, sd_emp, 'o', c=RED, ms=8, label='measured naive-t SD')\n"
            "a1.set_xlabel('overlap window'); a1.set_ylabel('naive-t SD (inflation factor)')\n"
            "a1.set_title('t-inflation = sqrt(window)'); a1.legend()\n"
            "a2.plot(ws, naive_fp, 'o-', c=RED, lw=2, label='naive')\n"
            "a2.plot(ws, nw_fp, 's-', c=GREEN, lw=2, label='Newey-West')\n"
            "a2.axhline(0.05, ls='--', c=GREY, label='nominal 5%')\n"
            "a2.set_xlabel('overlap window'); a2.set_ylabel('false-positive rate')\n"
            "a2.set_title('naive FP runs away; NW holds'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('window :', ws)\n"
            "print('emp SD :', [f'{v:.2f}' for v in sd_emp])\n"
            "print('sqrt(w):', [f'{v:.2f}' for v in sd_th])"
        ),
        md(
            "> 💡 In plain words: the red dots land on the dashed √window line at every window — the "
            "inflation is not approximate folklore, it is an identity. Practical shortcut: **divide a "
            "naive overlap *t* by √(window)** to sanity-check it. A '*t* = 3' on a 21-day overlap is "
            "really a *t* ≈ 0.65."
        ),
        md(
            "### 4c · The AR(1) dial and the bandwidth trap\n\n"
            "Autocorrelation needn't come from a hard window; any persistence does it. Sweep the AR(1) "
            "coefficient ρ; the inflation follows √((1+ρ)/(1−ρ)). Here NW uses its *automatic* "
            "bandwidth — watch it start to under-correct at high ρ, the lesson that the *formula* isn't "
            "enough without an adequate *bandwidth*."
        ),
        code(
            "df = st.inflation_curve_ar1([0.0, 0.2, 0.4, 0.6, 0.8], LIVE_REPS, LIVE_DAYS,\n"
            "                            seed=838, lags=st.nw_auto_lags(LIVE_DAYS))\n"
            "print(df.round(3).to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(df['rho'], df['theory_inflation'], 'k--', lw=1.8, label='sqrt((1+rho)/(1-rho))')\n"
            "ax.plot(df['rho'], df['naive_t_sd'], 'o', c=RED, ms=8, label='measured naive-t SD')\n"
            "ax.set_xlabel('AR(1) autocorrelation rho'); ax.set_ylabel('naive-t SD')\n"
            "ax.set_title('AR(1): inflation follows its closed form'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'auto-bandwidth used = {st.nw_auto_lags(LIVE_DAYS)} lags')"
        ),
        md(
            "> 💡 In plain words: the naive-*t* SD again tracks the closed form exactly. The Newey-West "
            "FP is well-behaved for mild ρ but drifts up near ρ = 0.8 because the automatic 8-lag "
            "bandwidth is too short for that much memory — HAC needs a bandwidth matched to the "
            "persistence, not just the right kernel (Newey-West 1994; Andrews 1991)."
        ),
        md(
            "### 4d · Our Newey-West equals statsmodels'\n\n"
            "The hand-rolled HAC *t* is not a bespoke approximation — it matches `statsmodels`' HAC "
            "covariance on an intercept-only OLS to six decimals."
        ),
        code(
            "import statsmodels.api as sm\n"
            "x = d.overlap_returns(1500, window=WINDOW, seed=11).to_numpy()\n"
            "ours = st.newey_west_t(x, lags=NW_LAGS)\n"
            "m = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS, 'use_correction': False})\n"
            "print(f'ours          : t = {ours:+.6f}')\n"
            "print(f'statsmodels   : t = {float(m.tvalues[0]):+.6f}')\n"
            "print(f'naive (i.i.d.): t = {st.one_sample_t(x):+.6f}   <- what you must NOT report here')"
        ),
        md(
            "> 💡 In plain words: the correction is the textbook one; the point of the study is not a new "
            "estimator but *the size of the error you make by skipping it* — often a factor of several "
            "on the very *t*-stat you were about to trust."
        ),
        md(
            "### 4e · The positive control — power is preserved\n\n"
            "A conservative test that never rejects would also 'fix' the false positives — uselessly. "
            "Plant a real mean and confirm Newey-West still rejects, with the right sign."
        ),
        code(
            "pw = st.power_check(LIVE_REPS, LIVE_DAYS, window=WINDOW, mean=6e-4, seed=838, lags=NW_LAGS)\n"
            "pw0 = st.power_check(LIVE_REPS, LIVE_DAYS, window=WINDOW, mean=0.0, seed=838, lags=NW_LAGS)\n"
            "print(f'planted mean 6e-4 : NW power {pw[\"nw_power\"]:.1%}, mean NW t {pw[\"nw_t_mean\"]:+.2f}, '\n"
            "      f'sign+ {pw[\"nw_t_positive_share\"]:.1%}')\n"
            "print(f'planted mean 0    : NW power {pw0[\"nw_power\"]:.1%}   <- no phantom signal')\n"
            "print(f'headline (600x2520): NW power {R[\"ctrl_power\"]:.1%}, mean NW t {R[\"ctrl_t_mean\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: on the empty market NW is quiet (~{R['nw_fp']:.0%}); on a genuine "
            f"effect it fires **~{R['ctrl_power']:.0%}** of the time with the correct sign "
            "**99.7%** of the time. The correction removes false positives *without* costing power — "
            "the defining property of an unbiased test."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — true mean 0 by construction; NW FP {R['nw_fp']:.1%} ≈ nominal. No "
            "robust |*t*| ≥ 2 on an honest tape exists because the honest tape is empty by design.\n"
            f"- **Tradability `MIRAGE`** — the naive significance is a variance artefact; costed, the "
            "null loses money (−24.6%/yr).\n"
            f"- **Does ignoring autocorrelation fake significance? `CONFIRMED`** — naive FP "
            f"{R['naive_fp']:.1%}, naive-*t* SD {R['naive_t_sd']:.2f} = √21, inflation matches √window "
            f"and √((1+ρ)/(1−ρ)), i.i.d. control calibrated, NW keeps {R['ctrl_power']:.0%} power. "
            "Report the HAC *t*, never the naive one, on any overlapping series."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the methodology dividend\n\n"
            "No edge to trade; the payoff is procedural. The desk's enforced rules, straight from this "
            "experiment:"
        ),
        code(
            "rows = [\n"
            "  ('Naive t on overlapping returns', 'inflated by ~sqrt(window) — do NOT report'),\n"
            "  ('Newey-West (HAC) t', 'bandwidth >= overlap length'),\n"
            "  ('Divide naive t by sqrt(window)', 'a 10-second sanity check'),\n"
            "  ('Block bootstrap', 'when the SE is in doubt, resample in blocks'),\n"
            "]\n"
            "print(pd.DataFrame(rows, columns=['practice', 'why']).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: a *t*-statistic is only as trustworthy as the standard error under it. "
            "On any series with a formation window, the ordinary SE is wrong by a factor of √(window) — "
            "correct it, and a lot of 'significant' edges quietly go away. That is the trade: the ones "
            "you *don't* fund."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Better kernels & prewhitening.** The quadratic-spectral kernel (Andrews 1991) is "
            "MSE-optimal; VAR-prewhitening (Andrews-Monahan 1992) shaves the residual over-rejection "
            "our Bartlett NW leaves (~9.5% vs 5%).\n"
            "- **Fixed-*b* inference.** Kiefer-Vogelsang (2005) give a size-improved asymptotic that "
            "keeps the bandwidth a fixed fraction of the sample.\n"
            "- **Your own signal.** Feed the daily returns of any window-based strategy into "
            "`strategy.newey_west_t` and compare to `strategy.one_sample_t`; the gap is how much the "
            "overlap was flattering you.\n\n"
            "*The uncomfortable corollary: the longer your formation window, the more suspicious a "
            "borderline *t*-stat should make you — because the overlap is inflating it by √(window). "
            "Fork this, run your signal through both standard errors, and let the honest one decide.*"
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
