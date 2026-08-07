"""Generate the two narrative notebooks for Study 841 (Overlapping-Returns Inflation).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a synthetic-only
method demo: every figure runs offline and deterministic on the seed-841 world — there is NO real-tape
cell (real free data can never certify "zero predictability"), so the study is capped at NONE. The
dict ``R`` below mirrors the frozen headline numbers in docs/results.md (the 2,000-sim Monte Carlo);
the notebooks live-run only the fast pieces (a single world; a small Monte Carlo) so they execute in
well under two minutes, and quote ``R`` for the heavy 2,000-sim sweep.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; null-world fp 4111f0ae3f09;
# 600 monthly rows; seeds 841…; rho=0.95, delta=-0.9; 2,000 sims/horizon; NW lags=h-1; crit=1.96).
R = dict(
    fp="4111f0ae3f09", n=600, start="1970-01-31", end="2019-12-31", rho=0.95, delta=-0.9, n_sims=2000,
    # null size sweep: (h, reject_naive, reject_nw, reject_hodrick, mean_abs_t_naive, mean_r2)
    null=[(1, 0.0615, 0.0635, 0.0615, 0.83, 0.0018),
          (3, 0.2650, 0.1205, 0.0580, 1.41, 0.0053),
          (6, 0.4290, 0.1370, 0.0600, 1.97, 0.0100),
          (12, 0.5600, 0.1570, 0.0600, 2.69, 0.0182),
          (24, 0.6560, 0.1910, 0.0600, 3.54, 0.0312)],
    # power sweep (beta=0.005): (h, naive, nw, hodrick)
    power=[(1, 0.9445, 0.9460, 0.9425),
           (3, 0.9950, 0.9835, 0.9300),
           (6, 0.9975, 0.9795, 0.8920),
           (12, 0.9990, 0.9665, 0.8145),
           (24, 0.9970, 0.9370, 0.6255)],
    # one null world, up close (seed 841): (h, slope, r2, t_naive, t_nw, t_hodrick)
    world=[(1, 0.0042, 0.0067, 2.01, 1.95, 1.94),
           (6, 0.0225, 0.0364, 4.73, 2.37, 1.88),
           (12, 0.0294, 0.0385, 4.84, 1.79, 1.29),
           (24, 0.0501, 0.0679, 6.47, 1.62, 1.27)],
    naive_h1=0.062, naive_h12=0.560, naive_h24=0.656, hod_h24=0.060, nw_h24=0.191,
    r2_h1=0.0018, r2_h24=0.0312,
    w_tnaive_h12=4.84, w_thod_h12=1.29, w_r2_h12=0.0385,
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

from overlapping_returns import data, strategy as st

# The whole demo runs on the deterministic, offline seed-841 world. beta=0 => NO predictability,
# so any long-horizon t-stat / R² above the nominal 5% level is a pure overlap artefact.
DF, TRUTH = data.simulate_world(n_months=600, beta=0.0, rho=0.95, seed=841)
X, Rr = DF["x"].to_numpy(), DF["r"].to_numpy()
print("null world:", len(DF), "months,", DF.index[0].date(), "->", DF.index[-1].date(),
      "| beta %.0f (NO edge), rho %.2f (fp %s)" % (TRUTH.beta, TRUTH.rho, data.fingerprint(DF)))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Overlapping Returns — how to 'discover' predictability that isn't there 🔗\n"
            "### Why a long-horizon predictive regression can print a stunning t-stat from a driftless predictor, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_overlap_inflate_inference%3F: Confirmed](https://img.shields.io/badge/Does_overlap_inflate_inference%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "For decades the headline evidence that markets are predictable came from **long-horizon "
            "regressions**: take a valuation ratio (say the dividend yield) known *today*, and use it "
            "to forecast the **total return over the next 12 months** — then slide the window forward "
            "one month and do it again, and again. The R² climbs, the t-stats look huge, and it feels "
            "like a discovery.\n\n"
            "There's a catch hiding in plain sight. Your 12-month return starting in January and your "
            "12-month return starting in February **share eleven months** of the same data. Every "
            "neighbouring data point overlaps its neighbours almost completely — and that overlap "
            "quietly breaks the statistics.\n\n"
            "So we run the experiment on a world we **built to have no predictability at all** — the "
            "predictor is pure noise, it forecasts *nothing* — and watch the naive regression "
            "'discover' an edge anyway.\n\n"
            "> 📓 **This is the plain-language layer.** Want the MA(h−1) residual structure, the "
            "Newey-West and Hodrick standard errors, and the Monte Carlo? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on a deterministic synthetic world. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Can *overlapping* long-horizon returns fake a great forecaster? | **Yes — hugely.** On "
            f"a driftless predictor, the 12-month regression prints a naive **t = +{R['w_tnaive_h12']}** "
            f"and R² **{R['w_r2_h12']*100:.1f}%** — a '5-sigma discovery' of an edge that *doesn't "
            f"exist*. |\n"
            f"| How often does the naive 5% test cry wolf? | At 12 months, **{R['naive_h12']*100:.0f}%** "
            f"of the time (it should be 5%). At 24 months, **{R['naive_h24']*100:.0f}%**. |\n"
            f"| Is there a fix? | **Yes.** The Hodrick (1992) standard error keeps the false-alarm rate "
            f"at a correct **~{R['hod_h24']*100:.0f}%** at every horizon. |\n"
            "| Is any of the 'predictability' real? | **None of it.** The world has zero edge by "
            "construction — every bit is an artefact of the overlap. |\n\n"
            "> The prettier the long-horizon R², the more suspicious you should be — overlap alone "
            "manufactures it for free."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Valuation ratios predict long-horizon returns — look at the R² and t-stats from "
            "regressing multi-year returns on the dividend yield.\"*\n\n"
            "The recipe: line up each month's predictor against the **cumulative return over the "
            "following `h` months**, run one big regression, read off the t-stat. The problem is that "
            "consecutive `h`-month returns overlap by `h−1` months, so the regression's errors are "
            "**not independent** — and the ordinary formula for the t-stat assumes they are."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because a generation of 'the long run is predictable' conclusions rests on exactly this "
            "regression. If overlapping data can conjure a t-stat above 6 from a predictor with **zero** "
            "forecasting power, then those conclusions need re-examining with an honest standard "
            "error — which is precisely what Hodrick (1992) and the careful re-studies (Ang & Bekaert "
            "2007; Boudoukh, Richardson & Whitelaw 2008) did, and much of the evidence softened."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We need a predictor we *know* forecasts nothing, so any 'predictability' must be fake. So "
            "we **build** a world: 600 months (50 years) of returns, plus a slow-moving, persistent "
            "predictor (like a valuation ratio) that is wired to have **zero** true forecasting power.\n\n"
            "Then we run the long-horizon regression at several horizons and check the t-stat **two "
            "ways**:\n\n"
            "1. the **naive** way everyone uses (the textbook OLS t-stat), and\n"
            "2. an **honest** way (the Hodrick 1992 correction) that accounts for the fact that "
            "neighbouring long-horizon returns overlap.\n\n"
            "If only the naive t-stat lights up, the 'discovery' is an overlap artefact."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually run it\n\n"
            "**One world, up close.** Same regression, three standard errors, as we lengthen the "
            "horizon. Watch the naive t-stat balloon while the honest (Hodrick) one stays calm — on a "
            "predictor that forecasts *nothing*."
        ),
        code(
            "rows = []\n"
            "for h in (1, 3, 6, 12, 24):\n"
            "    o = st.predictive_regression(X, Rr, h)\n"
            "    rows.append((h, o['slope'], o['r2'], o['t_naive'], o['t_nw'], o['t_hodrick']))\n"
            "tab = pd.DataFrame(rows, columns=['h','slope','naive_R2','t_naive','t_NW','t_Hodrick']).set_index('h')\n"
            "print(tab.round(3).to_string())\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.plot(tab.index, tab['t_naive'], 'o-', c=RED, lw=2, label='naive OLS t (what everyone quotes)')\n"
            "ax.plot(tab.index, tab['t_Hodrick'], 'o-', c=GREEN, lw=2, label='Hodrick 1992 t (honest)')\n"
            "ax.axhline(1.96, ls='--', c=GREY, lw=1, label='5% significance (|t|=1.96)')\n"
            "ax.axhline(-1.96, ls='--', c=GREY, lw=1)\n"
            "ax.set_xlabel('forecast horizon h (months)'); ax.set_ylabel('t-statistic of the slope')\n"
            "ax.set_title('A driftless predictor \\'predicts\\' 12-month returns at t>4.8 — but only naively')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"There it is. At a 12-month horizon the naive regression prints **t = +{R['w_tnaive_h12']}** "
            f"— it looks like an overwhelming discovery. But this world has **no predictability at "
            f"all**: the honest Hodrick t-stat is **+{R['w_thod_h12']}**, comfortably *inside* the "
            "±1.96 band. The whole 'signal' is the overlap talking."
        ),
        md(
            "**Is this one world a fluke?** No. Repeat the experiment over many fresh worlds and count "
            "how often each test wrongly declares 'significant'. An honest 5% test should cry wolf 5% "
            "of the time. Here's a quick live Monte Carlo at the 12-month horizon:"
        ),
        code(
            "e = st.size_experiment(data, h=12, beta=0.0, rho=0.95, n_sims=200, base_seed=841)\n"
            "print(f\"12-month horizon, {e['n_sims']} fresh NULL worlds — false-alarm rate of a 5% test:\")\n"
            "print(f\"  naive OLS : {e['reject_naive']*100:5.1f}%   (should be 5%)\")\n"
            "print(f\"  Hodrick   : {e['reject_hodrick']*100:5.1f}%   (should be 5%)\")\n"
            "fig, ax = plt.subplots(figsize=(6.5, 4.2))\n"
            "ax.bar(['naive OLS','Hodrick 1992'], [e['reject_naive']*100, e['reject_hodrick']*100],\n"
            "       color=[RED, GREEN], width=.55)\n"
            "ax.axhline(5, ls='--', c=GREY, lw=1.5, label='honest 5% rate')\n"
            "ax.set_ylabel('% of NULL worlds wrongly called significant')\n"
            "ax.set_title('How often each test cries wolf (12-month horizon)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"The naive test flags a phantom edge in **more than half** of no-edge worlds; the Hodrick "
            f"test stays near the honest 5%. And it gets *worse* the longer the horizon — over the full "
            f"{R['n_sims']:,}-world run the naive false-alarm rate climbs from "
            f"**{R['naive_h1']*100:.0f}% at 1 month** to **{R['naive_h24']*100:.0f}% at 24 months**:"
        ),
        code(
            "null = " + repr([(h, rn, rw, rh) for (h, rn, rw, rh, _t, _r) in R["null"]]) + "\n"
            "hh = [r[0] for r in null]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.plot(hh, [r[1]*100 for r in null], 'o-', c=RED, lw=2, label='naive OLS')\n"
            "ax.plot(hh, [r[2]*100 for r in null], 's-', c=AMBER, lw=2, label='Newey-West (h-1 lags)')\n"
            "ax.plot(hh, [r[3]*100 for r in null], '^-', c=GREEN, lw=2, label='Hodrick 1992')\n"
            "ax.axhline(5, ls='--', c=GREY, lw=1.5, label='honest 5% rate')\n"
            "ax.set_xlabel('forecast horizon h (months)')\n"
            "ax.set_ylabel('% of NULL worlds wrongly called significant')\n"
            "ax.set_title('The naive false-alarm rate explodes with the horizon; Hodrick holds')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('frozen 2,000-sim run (docs/results.md):')\n"
            "for h, rn, rw, rh in null:\n"
            "    print(f'  h={h:2d}: naive {rn*100:4.1f}%  NW {rw*100:4.1f}%  Hodrick {rh*100:4.1f}%')"
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The world has zero predictability by construction; the long-horizon "
            "'edge' is entirely an overlap artefact. Nothing real to detect.\n"
            "- **Tradability — Mirage.** You can't trade an inflated t-stat — the apparent forecast has "
            "no out-of-sample value and evaporates under an honest standard error.\n"
            "- **Does overlap inflate inference? — Confirmed.** The naive false-alarm rate hits "
            f"{R['naive_h24']*100:.0f}% at 24 months; the Hodrick correction restores an honest ~6%."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — there is nothing to trade. The regression describes a relationship that **does not "
            "exist**; its out-of-sample forecast is worthless. The only thing the overlap inflated was "
            "your *confidence*, not your returns. The fix isn't a better strategy, it's a correct "
            "standard error."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The fix is public and old.** The Hodrick (1992) standard error and Newey-West (1987) "
            "HAC both defang the overlap — see the quants notebook for exactly how.\n"
            "- **The cousins on this desk.** [838 HAC-Necessity](../../838-hac-necessity/) is the same "
            "autocorrelation problem for a *strategy's own daily P&L*; [835 Spurious-Regression]"
            "(../../835-spurious-regression/) fakes a relationship from *trending* series; "
            "[346 Multiple-Testing](../../346-multiple-testing/) fakes significance from *many "
            "hypotheses*.\n\n"
            "*Think a long-horizon R² proves the long run is predictable? Fork this, feed in a real "
            "dividend-yield series, and watch the Hodrick correction cut the t-stat down to size.*"
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
            "# Overlapping-Returns Inflation — a quantitative teardown 🔬\n"
            "### the MA(h−1) residual structure · naive OLS vs Newey-West vs Hodrick 1992 (1B) · Monte-Carlo size under the null · R² inflation vs horizon · the seed-robust positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_overlap_inflate_inference%3F: Confirmed](https://img.shields.io/badge/Does_overlap_inflate_inference%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We regress overlapping "
            "`h`-month returns on a persistent predictor with **zero** forecasting power, and measure "
            "how badly the overlap inflates the naive t and R² — and which correction repairs it.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the world is built to have "
            f"zero predictability (null fp `{R['fp']}`), so real free data can never certify it and the "
            "study is capped at `NONE`. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | On a `beta=0` world, the 12-month regression prints naive "
            f"t = +{R['w_tnaive_h12']}, R² {R['w_r2_h12']*100:.1f}% — a phantom edge; synthetic-only "
            "demo (no real tape). |\n"
            f"| **Tradability** | `MIRAGE` | An inflated t/R² has zero out-of-sample value; it vanishes "
            "under a correct standard error. |\n"
            f"| **Does overlap inflate inference?** | `CONFIRMED` | Naive 5%-test size {R['naive_h1']*100:.0f}% "
            f"(h=1) → {R['naive_h24']*100:.0f}% (h=24); Hodrick 1B stays ~{R['hod_h24']*100:.0f}% at "
            f"every horizon; NW only partly ({R['nw_h24']*100:.0f}% at h=24). |\n\n"
            "> 💡 In plain words: overlapping the dependent variable induces MA(h−1) residual "
            "autocorrelation the OLS SE ignores; Hodrick's estimator builds its moments from "
            "non-overlapping one-period returns and is well-sized, and the control proves it still "
            "detects a *real* edge."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_t$ be a predictor and $y_t = \\sum_{j=1}^{h} r_{t+j}$ the overlapping `h`-period "
            "forward return. The predictive regression is $y_t = a + b\\,x_t + e_t$.\n\n"
            "- **The trap.** Sampling monthly, $y_t$ and $y_{t+1}$ share $h-1$ of their $h$ summands, "
            "so under the null ($b=0$) the residual $e_t$ is a **moving average of order $h-1$**. OLS "
            "standard errors assume $\\mathrm{Cov}(e_t,e_{t-k})=0$; here it is nonzero for "
            "$k<h$, so $\\widehat{\\mathrm{Var}}(\\hat b)$ is understated and the t-stat is inflated.\n"
            "- **The R² is inflated too**, and both distortions **grow with $h$**.\n"
            "- **Fix A — Newey-West (1987).** A HAC sandwich with Bartlett weights and $\\approx h-1$ "
            "lags absorbs the MA structure.\n"
            "- **Fix B — Hodrick (1992) 1B.** An exact reformulation: since "
            "$\\hat b = \\big[\\sum_s r_s\\,\\mathrm{XS}_s\\big]/\\sum_t \\tilde x_t^2$ with "
            "$\\mathrm{XS}_s=\\sum_{i=1}^{h}\\tilde x_{s-i}$, the numerator is built from "
            "**non-overlapping** one-period returns, so its variance needs no autocovariance terms — "
            "only a heteroskedastic core. Best finite-sample size.\n\n"
            "We **confirm the trap** (naive size ≫ 5%, rising in $h$), **confirm Fix A** (NW helps but "
            "stays over-sized), and **confirm Fix B** (Hodrick ≈ nominal at every $h$)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "The naive OLS variance is $\\hat\\sigma^2 (X'X)^{-1}$, valid only under spherical errors. "
            "With overlap the correct sandwich is $(X'X)^{-1} S (X'X)^{-1}$ where the 'meat' $S$ sums "
            "the score autocovariances $\\Gamma_k=\\mathbb E[g_t g_{t-k}']$, $g_t=X_t e_t$, out to "
            "$k=h-1$. Under positive persistence these $\\Gamma_k$ are positive, so the naive $S=\\Gamma_0$ "
            "**understates** the true variance — exactly the inflation we see. This is the same family "
            "of correction as [838 HAC-Necessity](../../838-hac-necessity/) (there: a strategy's own "
            "daily P&L; here: the *overlap* of cumulative returns). Hodrick's trick sidesteps the "
            "$\\Gamma_k$ estimation entirely."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **World.** 600 monthly returns; a persistent AR(1) predictor ($\\rho=0.95$) with "
            "Stambaugh feedback ($\\delta=-0.9$, the return/predictor innovation correlation of real "
            "valuation ratios); **$\\beta=0$** (zero predictability) for the null, $\\beta>0$ for the "
            "control.\n"
            "- **Three t-stats.** `ols_slope_t` (naive homoskedastic), `newey_west_slope_t` "
            "(HAC, lags $=h-1$), `hodrick_1b_slope_t` (non-overlapping moments).\n"
            "- **Monte Carlo.** `size_experiment` repeats over many worlds and reports the **rejection "
            "rate** at $|t|>1.96$ — the *size* under the null, the *power* under a planted $\\beta$.\n"
            "- **Execution honesty.** $x_t$ is known at the close of month $t$ and predicts month "
            "$t+1$ onward (a one-period lag baked into the alignment).\n\n"
            "The heavy 2,000-sim sweep is quoted from `docs/results.md`; the cells below live-run "
            "smaller Monte Carlos so the notebook executes fast."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · One world, three standard errors — the trap in a single regression\n\n"
            "On the seed-841 null world, lengthen the horizon and read all three t-stats."
        ),
        code(
            "rows = []\n"
            "for h in (1, 3, 6, 12, 24):\n"
            "    o = st.predictive_regression(X, Rr, h)\n"
            "    rows.append(o)\n"
            "tab = pd.DataFrame(rows).set_index('h')[['slope','r2','t_naive','t_nw','t_hodrick']]\n"
            "print(tab.round(4).to_string())\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.plot(tab.index, tab['t_naive'], 'o-', c=RED, lw=2, label='naive OLS')\n"
            "ax.plot(tab.index, tab['t_nw'], 's-', c=AMBER, lw=2, label='Newey-West (h-1)')\n"
            "ax.plot(tab.index, tab['t_hodrick'], '^-', c=GREEN, lw=2, label='Hodrick 1B')\n"
            "ax.axhline(1.96, ls='--', c=GREY, lw=1); ax.axhline(-1.96, ls='--', c=GREY, lw=1)\n"
            "ax.set_xlabel('horizon h (months)'); ax.set_ylabel('slope t-statistic')\n"
            "ax.set_title('Same beta=0 world: the naive t inflates with h; Hodrick stays inside +/-1.96')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: at $h=12$ the naive t is **+{R['w_tnaive_h12']}** (spuriously "
            f"'significant'), NW is +1.79, and Hodrick is **+{R['w_thod_h12']}** (correctly not "
            "significant). The overlap manufactured the naive t out of nothing."
        ),
        md(
            "### 4b · The size of the test vs the horizon — the headline (2,000 sims)\n\n"
            "The definitive plot: the rejection rate of a nominal 5% test **under the null**, by "
            "horizon and method. This is the frozen 2,000-sim run from `docs/results.md`; a smaller "
            "live Monte Carlo below reproduces the $h=12$ point."
        ),
        code(
            "null = " + repr(R["null"]) + "  # (h, naive, nw, hod, mean|t|_naive, mean_r2)\n"
            "hh = [r[0] for r in null]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "a1.plot(hh, [r[1]*100 for r in null], 'o-', c=RED, lw=2, label='naive OLS')\n"
            "a1.plot(hh, [r[2]*100 for r in null], 's-', c=AMBER, lw=2, label='Newey-West')\n"
            "a1.plot(hh, [r[3]*100 for r in null], '^-', c=GREEN, lw=2, label='Hodrick 1B')\n"
            "a1.axhline(5, ls='--', c=GREY, lw=1.5, label='nominal 5%')\n"
            "a1.set_xlabel('horizon h (months)'); a1.set_ylabel('rejection rate under the null (%)')\n"
            "a1.set_title('Test SIZE: naive explodes, Hodrick holds'); a1.legend()\n"
            "a2.plot(hh, [r[5]*100 for r in null], 'o-', c=RED, lw=2)\n"
            "a2.set_xlabel('horizon h (months)'); a2.set_ylabel('mean naive R² under the null (%)')\n"
            "a2.set_title('...and the naive R² is inflated from nothing too')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('frozen 2,000-sim run:')\n"
            "for h, rn, rw, rh, mt, mr in null:\n"
            "    print(f'  h={h:2d}: naive {rn*100:4.1f}%  NW {rw*100:4.1f}%  Hodrick {rh*100:4.1f}%  '\n"
            "          f'| mean|t|_naive {mt:.2f}  mean_R2 {mr*100:.2f}%')"
        ),
        code(
            "# Live check of the h=12 headline point (smaller Monte Carlo, same story):\n"
            "e = st.size_experiment(data, h=12, beta=0.0, rho=0.95, n_sims=300, base_seed=841)\n"
            "print(f\"live {e['n_sims']}-sim size at h=12:  naive {e['reject_naive']*100:.1f}%  \"\n"
            "      f\"NW {e['reject_nw']*100:.1f}%  Hodrick {e['reject_hodrick']*100:.1f}%\")\n"
            f"print(\"frozen 2,000-sim size  at h=12:  naive {R['naive_h12']*100:.1f}%  "
            f"NW 15.7%  Hodrick 6.0%\")"
        ),
        md(
            f"> 💡 In plain words: the naive 5% test rejects a *true* null "
            f"**{R['naive_h1']*100:.0f}% → {R['naive_h24']*100:.0f}%** of the time as $h$ grows, and "
            f"the mean R² inflates **{R['r2_h1']*100:.1f}% → {R['r2_h24']*100:.1f}%** — all from "
            f"overlap, on a predictor that forecasts nothing. Hodrick 1B stays at ~"
            f"{R['hod_h24']*100:.0f}% throughout."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — a `beta=0` world; the long-horizon t (+{R['w_tnaive_h12']} at "
            "h=12) and R² are pure overlap artefacts. Synthetic-only — no real tape, so never `REAL`.\n"
            f"- **Tradability `MIRAGE`** — an inflated t/R² has no out-of-sample value; correct it and "
            "the edge is gone.\n"
            f"- **Does overlap inflate inference? `CONFIRMED`** — naive size "
            f"{R['naive_h24']*100:.0f}% at h=24, R² up to {R['r2_h24']*100:.1f}%; Hodrick 1B repairs it."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — nothing to harvest\n\n"
            "There is no P&L here to cost: the regression describes a relationship that does not exist, "
            "and its out-of-sample forecast is worthless. Unlike a cross-sectional strategy, the "
            "'edge' is a pure inference artefact — the remedy is a correct standard error, not a "
            "cheaper broker. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control (is the fix just numb?)\n\n"
            "A good correction must still **detect a real edge**, not merely tame the null. Plant a "
            "genuine one-period edge ($\\beta=0.005$) and measure the rejection rate = **power**. Run "
            "it live (small Monte Carlo) and compare to the frozen 2,000-sim run."
        ),
        code(
            "hs = (1, 6, 12, 24)\n"
            "live = {h: st.size_experiment(data, h=h, beta=0.005, rho=0.95, n_sims=200, base_seed=841)\n"
            "        for h in hs}\n"
            "power = " + repr(R["power"]) + "  # frozen 2,000-sim (h, naive, nw, hod)\n"
            "pmap = {h: (n, w, hd) for (h, n, w, hd) in power}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "hh = [h for (h, *_ ) in power]\n"
            "ax.plot(hh, [pmap[h][2]*100 for h in hh], '^-', c=GREEN, lw=2, label='Hodrick 1B power (real edge)')\n"
            "ax.plot(hh, [pmap[h][1]*100 for h in hh], 's-', c=AMBER, lw=2, label='Newey-West power')\n"
            "null = " + repr([(h, rh) for (h, _n, _w, rh, _t, _r) in R["null"]]) + "\n"
            "nmap = {h: v for h, v in null}\n"
            "ax.plot(hh, [nmap[h]*100 for h in hh], '^:', c=GREY, lw=2, label='Hodrick size (NULL, ~5%)')\n"
            "ax.axhline(5, ls='--', c=GREY, lw=1)\n"
            "ax.set_xlabel('horizon h (months)'); ax.set_ylabel('rejection rate (%)')\n"
            "ax.set_title('The corrections are POWERED on a real edge, and SIZED on the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('live 200-sim power (planted beta=0.005):')\n"
            "for h in hs:\n"
            "    print(f\"  h={h:2d}: Hodrick {live[h]['reject_hodrick']*100:5.1f}%  \"\n"
            "          f\"NW {live[h]['reject_nw']*100:5.1f}%   (frozen Hodrick {pmap[h][2]*100:.1f}%)\")"
        ),
        md(
            "The Hodrick and Newey-West tests keep **high power** to detect a genuine edge (Hodrick "
            f"{R['power'][0][3]*100:.0f}% → {R['power'][-1][3]*100:.0f}% across horizons; it turns "
            "appropriately conservative as $h$ grows) while sitting at ~5% under the null. So the "
            "correction is a faithful detector: it refuses to be fooled by overlap *and* still banks a "
            "real edge. For the daily-P&L version of this autocorrelation problem see "
            "[838 HAC-Necessity](../../838-hac-necessity/); for false significance from *many "
            "hypotheses*, [346 Multiple-Testing](../../346-multiple-testing/)."
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
