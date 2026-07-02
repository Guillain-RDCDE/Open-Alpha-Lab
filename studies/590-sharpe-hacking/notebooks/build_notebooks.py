"""Generate the two narrative notebooks for Study 590 (Sharpe-Hacking).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a synthetic-only
method demo: every figure runs offline and deterministic on the seed-590 tape — there is NO real-tape
cell (real free data can never certify "zero alpha"), so the study is capped at NONE. The dict ``R``
below mirrors the headline numbers in docs/results.md; the notebooks recompute them live from the
engine so they always agree.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; null tape
# fp 807d5cddc33f; 3,000 daily rows; seed 590). Recomputed live in the notebooks.
R = dict(
    fp="807d5cddc33f", n=3000, start="2008-01-02", end="2019-07-02",
    base_honest=0.24,
    raw_naive=0.323, raw_honest=0.242,
    sm_naive=0.559, sm_honest=0.244, sm_naive_infl=0.236, sm_honest_infl=0.002,
    lev_naive=0.323, lev_honest=0.242,
    vt_naive=0.158, vt_honest=0.136, vt_net_naive=0.145, vt_net_honest=0.125,
    theta08_naive=0.979, ac1_raw=0.01, ac1_08=0.81,
    honest_ci=(-0.012, 0.013),
    # theta sweep: (theta, naive, honest, ac1)
    sweep=[(0.0, 0.323, 0.242, 0.01), (0.2, 0.394, 0.242, 0.21), (0.4, 0.492, 0.243, 0.42),
           (0.6, 0.647, 0.246, 0.61), (0.8, 0.979, 0.251, 0.81)],
    # seed control (25 seeds): (alpha, honest_raw, naive_infl, honest_infl)
    ctrl=[(0.0, 0.30, 0.218, 0.002), (0.05, 0.57, 0.396, 0.003),
          (0.10, 0.83, 0.573, 0.004), (0.20, 1.37, 0.927, 0.007)],
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

from sharpe_hacking import data, strategy as st

# The whole demo runs on the deterministic, offline seed-590 null tape (honest Sharpe ~0.24, ZERO
# alpha). This is a synthetic-only method demo — there is no real-tape cell by design.
RET, TRUTH = data.synthetic_series(n_days=3000, alpha=0.0, seed=590)
print("null tape:", len(RET), "days,", RET.index[0].date(), "->", RET.index[-1].date(),
      "| honest Sharpe %.2f, alpha %.0f (fp %s)" % (
          st.sharpe_honest_annual(RET.to_numpy()), TRUTH.alpha, data.fingerprint(RET)))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sharpe-Hacking — can you *fake* a great risk-adjusted return? 🎛️\n"
            "### Juicing a strategy's headline Sharpe with pure financial engineering, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Can_engineering_fake_a_Sharpe%3F: Confirmed](https://img.shields.io/badge/Can_engineering_fake_a_Sharpe%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The **Sharpe ratio** is the single most quoted number in investing: return per unit of "
            "risk. A Sharpe of 2 with a smooth, steadily-rising equity curve is catnip to allocators. "
            "It is also, it turns out, one of the *easiest numbers in finance to fake* — not by lying, "
            "but by pure **financial engineering**: how you *report* and *reshape* your returns.\n\n"
            "So we run the experiment on a stream of returns we **built to have no real edge** — a "
            "modest, honest Sharpe and zero skill — and try three classic tricks: **smoothing** the "
            "returns (like an illiquid fund with stale prices), piling on **leverage**, and "
            "**volatility targeting**. Which ones actually 'work'? And do any of them add real "
            "money?\n\n"
            "> 📓 **This is the plain-language layer.** Want the autocorrelation-corrected Sharpe, the "
            "bootstrap bands and the maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on a deterministic synthetic tape. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Can *smoothing* your returns fake a great Sharpe? | **Yes — hugely.** It lifts the "
            f"*reported* Sharpe from **{R['raw_naive']}** to **{R['sm_naive']}** (+73%), and up to "
            f"**{R['theta08_naive']}** (3× !) if you smooth harder — with **zero** added return. |\n"
            f"| Does piling on *leverage* help? | **No — not at all.** 3× leverage leaves the Sharpe "
            f"*exactly* unchanged ({R['lev_naive']}). It buys you a wilder ride, not a better ratio. |\n"
            f"| Is *volatility targeting* magic? | **No.** On this tape it *lowered* the Sharpe (to "
            f"**{R['vt_naive']}**). It's a real technique, but no free lunch. |\n"
            "| Is any of the 'gain' real? | **None of it.** An honest Sharpe that can't be fooled by "
            f"smoothing stays put at **~{R['base_honest']}** through all three tricks. |\n\n"
            "> The smooth, high-Sharpe track record everyone loves is often the *warning sign*, not "
            "the boast — it's exactly what stale, illiquid marks produce for free."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"You can juice a strategy's Sharpe ratio with pure financial engineering — "
            "vol-targeting, leverage, smoothing — without adding any real edge.\"*\n\n"
            "The Sharpe ratio is (roughly) your average return divided by how much it wobbles, scaled "
            "to a year. Two ways to make it bigger: earn more (hard, requires real skill), or **wobble "
            "less** (easy, if you're willing to *report* your returns cleverly). This study is about "
            "the second path — the metric-gaming one — and whether the folk levers people reach for "
            "actually move the needle."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because allocators hand out billions on the strength of a Sharpe ratio. If a smooth, "
            "Sharpe-2 line can be *manufactured* from a mediocre, no-edge return stream, then the "
            "number is nearly worthless without knowing *how it was measured*. The desk has already "
            "shown you can fake a Sharpe by **searching** for a lucky backtest "
            "([344 Backtest-Overfitting](../../344-backtest-overfitting/), "
            "[589 Genetic-Algo-Overfit](../../589-genetic-algo-overfit/)); here we fake it a different "
            "way — by **reshaping how the returns are reported**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We need a return stream we *know* has no real edge, so any Sharpe 'improvement' must be "
            "fake. So we **build** one: 3,000 days of returns with a modest honest Sharpe (~0.24) and "
            "zero skill, but with realistic *volatility clustering* (calm stretches and stormy "
            "stretches) so the vol-targeting trick has something real to act on.\n\n"
            "Then we run three tricks and, crucially, measure the Sharpe **two ways**:\n\n"
            "1. the **naive** Sharpe everyone quotes (average ÷ wobble × √252), and\n"
            "2. an **honest** Sharpe that corrects for a trap: if your returns are artificially "
            "*smooth* (each day looks like yesterday), the naive formula over-counts your skill. The "
            "honest version undoes exactly that.\n\n"
            "If a trick only moves the *naive* number and not the *honest* one, it's a fake."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually try it\n\n"
            "**Trick #1 — smoothing.** Pretend you run an illiquid fund that marks its book slowly, so "
            "each reported return is a blend of today's and yesterday's. Watch the *reported* Sharpe "
            "jump while the honest one doesn't move."
        ),
        code(
            "r = RET.to_numpy()\n"
            "gt = st.gaming_table(r, theta=0.5, leverage=3.0, target_ann_vol=0.10)\n"
            "labels = ['raw', 'smoothed', 'levered x3', 'vol-targeted']\n"
            "naive = gt['naive_sharpe'].values; honest = gt['honest_sharpe'].values\n"
            "x = np.arange(len(labels)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.bar(x - w/2, naive, w, color=RED, label='REPORTED (naive) Sharpe')\n"
            "ax.bar(x + w/2, honest, w, color=GREEN, label='HONEST (corrected) Sharpe')\n"
            "ax.axhline(honest[0], ls='--', c=GREY, lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('Only smoothing moves the REPORTED Sharpe — the honest one stays put')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for lab, nn, hh in zip(labels, naive, honest):\n"
            "    print(f'{lab:14s} reported {nn:.3f} | honest {hh:.3f}')"
        ),
        md(
            f"There it is. **Smoothing** lifts the reported Sharpe from **{R['raw_naive']}** to "
            f"**{R['sm_naive']}** — a +73% 'improvement' that added *nothing* (the honest Sharpe stays "
            f"~{R['base_honest']}). **Leverage** does *literally nothing* to either number — the bars "
            "are identical to raw. **Vol-targeting** actually *lowered* the Sharpe here. The only "
            "'winner' is the pure accounting trick."
        ),
        md(
            "**How far can smoothing take you?** The more you smooth (higher θ = staler marks), the "
            "prettier the reported number — while the honest Sharpe flatlines:"
        ),
        code(
            "sw = st.theta_sweep(r, thetas=(0.0, 0.2, 0.4, 0.6, 0.8))\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.plot(sw.index, sw['naive_sharpe'], 'o-', c=RED, lw=2, label='REPORTED (naive) Sharpe')\n"
            "ax.plot(sw.index, sw['honest_sharpe'], 'o-', c=GREEN, lw=2, label='HONEST (corrected) Sharpe')\n"
            "ax.set_xlabel('smoothing intensity theta (staler marks ->)'); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('More smoothing = a bigger REPORTED Sharpe (up to 3x) — honest one flat')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(sw.round(3).to_string())"
        ),
        md(
            f"At heavy smoothing (θ 0.8) the reported Sharpe hits **{R['theta08_naive']}** — three "
            f"times the truth — purely because the returns now look artificially calm (each day "
            f"tracks the last, autocorrelation {R['ac1_raw']} → {R['ac1_08']}). This is exactly why a "
            "suspiciously *smooth* hedge-fund track record should raise an eyebrow, not a cheque."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On a stream with no real edge, the only Sharpe 'gain' is a "
            "measurement illusion (smoothing); leverage does nothing, vol-targeting hurts. Nothing "
            "real to detect.\n"
            "- **Tradability — Mirage.** A smoothed Sharpe is an accounting number you can't spend. "
            "There's no money in any of it.\n"
            "- **Can engineering fake a Sharpe? — Confirmed.** Yes — the *reported* one, by up to 3×. "
            "Only the honest, correction-proof Sharpe tells the truth."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually 'trade' it?\n\n"
            "You can't spend a Sharpe ratio. Smoothing just changes the *story* your returns tell — "
            "the actual dollars are unchanged (worse, the smoothing hides the real drawdowns until "
            "they arrive all at once). Leverage genuinely changes your dollars, but scales risk and "
            "reward together — the ratio is fixed. Vol-targeting trades every day and, here, loses "
            "after costs. There is no free Sharpe to harvest — only a prettier report."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The searching cousins.** [344 Backtest-Overfitting](../../344-backtest-overfitting/) "
            "and [589 Genetic-Algo-Overfit](../../589-genetic-algo-overfit/) fake a Sharpe by *trying "
            "many strategies* and keeping the luckiest; this study fakes it by *reshaping the "
            "returns*.\n"
            "- **The fix is public.** The autocorrelation-corrected Sharpe (Lo 2002; "
            "Getmansky-Lo-Makarov 2004) undoes the smoothing game — see the quants notebook.\n\n"
            "*Think a smooth, high-Sharpe track record proves skill? Fork this, feed in a real "
            "illiquid-fund return series, and watch the correction cut the Sharpe down to size.*"
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
            "# Sharpe-Hacking — a quantitative teardown 🔬\n"
            "### naive vs autocorrelation-corrected (Lo/GLM) Sharpe · AR(1) smoothing · θ-inflation sweep · leverage invariance · vol-targeting costs · block-bootstrap bands · seed-robust positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Can_engineering_fake_a_Sharpe%3F: Confirmed](https://img.shields.io/badge/Can_engineering_fake_a_Sharpe%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We take a synthetic return "
            "stream with an honest Sharpe of ~0.24 and **zero alpha**, apply three Sharpe-gaming "
            "transforms, and measure the damage with the metric that can't be gamed.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the tape is built to have "
            f"zero edge (null fp `{R['fp']}`), so real free data can never certify it and the study is "
            "capped at `NONE`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Smoothing (θ 0.5) inflates the **naive** Sharpe "
            f"{R['raw_naive']} → {R['sm_naive']} (+{R['sm_naive_infl']}) while the **honest** Sharpe "
            f"moves {R['raw_honest']} → {R['sm_honest']} (+{R['sm_honest_infl']}); leverage ×3 changes "
            "*neither*; synthetic-only demo (no real tape). |\n"
            f"| **Tradability** | `MIRAGE` | Vol-targeting gross naive {R['vt_naive']} → net "
            f"{R['vt_net_naive']} (2 bps); a smoothed Sharpe is unspendable; leverage buys risk, not "
            "ratio. |\n"
            f"| **Can engineering fake a Sharpe?** | `CONFIRMED` | Naive Sharpe scales with staleness "
            f"({R['raw_naive']} → {R['theta08_naive']} as θ → 0.8); honest-inflation 95% CI "
            f"`[{R['honest_ci'][0]}, {R['honest_ci'][1]}]` straddles 0. |\n\n"
            "> 💡 In plain words: the reported Sharpe is a smoothing artefact; the correction sees "
            "through it, and the control proves the honest metric still rewards a *real* edge."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be true returns and $\\widetilde r_t$ the *reported* series after a transform. "
            "The annualised Sharpe is $\\widehat{SR} = \\bar r / \\hat\\sigma \\cdot \\sqrt{252}$.\n\n"
            "- **H₁ (smoothing).** An AR(1) smoothing $\\widetilde r_t = (1-\\theta) r_t + \\theta "
            "\\widetilde r_{t-1}$ preserves $\\bar r$ but lowers $\\hat\\sigma$ and adds positive "
            "autocorrelation, so the **naive** $\\widehat{SR}$ rises with $\\theta$ — with no real "
            "return.\n"
            "- **H₂ (leverage).** $\\widetilde r_t = L\\,r_t$ scales $\\bar r$ and $\\hat\\sigma$ "
            "equally, so $\\widehat{SR}$ is *invariant* to $L$.\n"
            "- **H₃ (vol-targeting).** Rescaling to a constant vol changes $\\widehat{SR}$ only if the "
            "mean-vol relationship helps — not guaranteed, and not free of turnover.\n"
            "- **H₄ (the fix).** The Lo (2002) autocorrelation-corrected Sharpe is *invariant* to "
            "smoothing: it strips the $\\eta(q)$ that naive annualisation gets wrong.\n\n"
            "We **confirm H₁** (big naive inflation), **confirm H₂** (exact invariance), **reject the "
            "free-lunch reading of H₃** (Sharpe falls here), and **confirm H₄** (honest Sharpe "
            "unmoved by smoothing)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "The naive Sharpe annualises a daily Sharpe by $\\sqrt{252}$, which is **only correct for "
            "iid returns**. Smoothing injects serial correlation, so the right factor is Lo's "
            "$\\eta(q) = q / \\sqrt{q + 2\\sum_{k<q}(q-k)\\rho_k}$, which is *smaller* than "
            "$\\sqrt{q}$ under positive autocorrelation — exactly cancelling the inflation. This is "
            "the same family of correction as the Deflated Sharpe Ratio in "
            "[344](../../344-backtest-overfitting/) (there: haircut for *trials*; here: haircut for "
            "*autocorrelation*). Getmansky-Lo-Makarov (2004) showed illiquid hedge funds smooth "
            "exactly this way."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Tape.** 3,000 daily returns, honest Sharpe ≈ 0.24, **zero alpha**, AR(1) log-variance "
            "vol clock (so vol-targeting has time-varying vol to act on). No serial correlation in the "
            "*mean* — the raw series is honestly annualisable.\n"
            "- **Two metrics.** `sharpe_naive` (mean/std × √252) and `sharpe_honest_annual` (Lo/GLM "
            "$\\eta$-corrected). For iid returns they agree; smoothing splits them apart.\n"
            "- **Three transforms.** `smooth_returns(θ)` (stale marks), `lever_returns(L)` (constant "
            "scale), `vol_target_returns(tgt)` (trailing-vol rescale, no look-ahead).\n"
            "- **Inference.** A circular-block bootstrap (block 20d) on the smoothing inflation, read "
            "for *both* metrics.\n"
            "- **Costs.** A 2 bps turnover charge on the vol-targeting exposure changes.\n"
            "- **Positive control.** The same machinery on tapes with a *planted* edge "
            "(`alpha > 0`), averaged over 25 seeds (house rule).\n\n"
            "Execution honesty: the vol-targeting exposure uses a *trailing* realised-vol window known "
            "*before* the day (a one-day lag, no look-ahead); everything is gross **and** net labelled."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The three levers, naive vs honest — H₁, H₂, H₃\n\n"
            "The centrepiece table: how each transform moves the reported vs the honest Sharpe."
        ),
        code(
            "r = RET.to_numpy()\n"
            "gt = st.gaming_table(r, theta=0.5, leverage=3.0, target_ann_vol=0.10)\n"
            "print(gt.round(3).to_string())\n"
            "labels = ['raw', 'smoothed', 'levered x3', 'vol-target']\n"
            "x = np.arange(len(labels)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.bar(x - w/2, gt['naive_sharpe'], w, color=RED, label='naive (reported)')\n"
            "ax.bar(x + w/2, gt['honest_sharpe'], w, color=GREEN, label='honest (corrected)')\n"
            "ax.axhline(gt['honest_sharpe'].iloc[0], ls='--', c=GREY, lw=1, label='honest baseline')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('Smoothing inflates the naive Sharpe alone; leverage is invariant')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **H₁ confirmed** — smoothing lifts the naive Sharpe by "
            f"+{R['sm_naive_infl']} and the honest by +{R['sm_honest_infl']} (≈0). **H₂ confirmed** — "
            f"leverage ×3 leaves *both* Sharpes at {R['lev_naive']} / {R['lev_honest']}, to machine "
            f"precision. **H₃'s free-lunch reading rejected** — vol-targeting *lowered* the naive "
            f"Sharpe to {R['vt_naive']} on this tape."
        ),
        md(
            "### 4b · The smoothing dose-response — H₁ and H₄ together\n\n"
            "Sweep the smoothing intensity θ and watch the naive Sharpe balloon while the honest one "
            "flatlines. The lag-1 autocorrelation is the tell."
        ),
        code(
            "sw = st.theta_sweep(r, thetas=(0.0, 0.2, 0.4, 0.6, 0.8))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(sw.index, sw['naive_sharpe'], 'o-', c=RED, lw=2, label='naive')\n"
            "a1.plot(sw.index, sw['honest_sharpe'], 'o-', c=GREEN, lw=2, label='honest')\n"
            "a1.set_xlabel('smoothing theta'); a1.set_ylabel('annualised Sharpe')\n"
            "a1.set_title('naive Sharpe scales with theta; honest flat'); a1.legend()\n"
            "a2.plot(sw.index, sw['lag1_autocorr'], 'o-', c=AMBER, lw=2)\n"
            "a2.set_xlabel('smoothing theta'); a2.set_ylabel('lag-1 autocorrelation')\n"
            "a2.set_title('...because smoothing injects autocorrelation')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sw.round(3).to_string())"
        ),
        md(
            f"> 💡 In plain words: **H₁ + H₄ confirmed.** As θ → 0.8 the naive Sharpe reaches "
            f"**{R['theta08_naive']}** (a 3× inflation) and the lag-1 autocorrelation reaches "
            f"**{R['ac1_08']}** — the two move together. The honest Sharpe stays ~{R['base_honest']} "
            "throughout: the $\\eta$-correction cancels exactly the autocorrelation the naive formula "
            "mistakes for skill."
        ),
        md(
            "### 4c · The bootstrap bands — which metric is the artefact\n\n"
            "A circular-block bootstrap (block 20d) on the θ = 0.5 inflation, read for both metrics."
        ),
        code(
            "nl = st.smoothing_inflation_null(r, theta=0.5, n_boot=2000, seed=590)\n"
            "print('naive  inflation %.3f  CI [%.3f, %.3f]' % (nl['naive_inflation'], *nl['naive_ci']))\n"
            "print('honest inflation %.3f  CI [%.3f, %.3f]' % (nl['honest_inflation'], *nl['honest_ci']))\n"
            "print('frac honest >= naive point:', nl['frac_honest_ge_naive'])\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "pts = [nl['naive_inflation'], nl['honest_inflation']]\n"
            "err = [[nl['naive_inflation']-nl['naive_ci'][0], nl['honest_inflation']-nl['honest_ci'][0]],\n"
            "       [nl['naive_ci'][1]-nl['naive_inflation'], nl['honest_ci'][1]-nl['honest_inflation']]]\n"
            "ax.errorbar(['naive\\n(reported)','honest\\n(corrected)'], pts, yerr=err, fmt='o',\n"
            "            color=RED, ecolor=GREY, capsize=6, ms=9, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('smoothing inflation (Sharpe units)')\n"
            "ax.set_title('naive inflation firmly > 0; honest inflation straddles 0')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the **honest** inflation's 95% CI "
            f"`[{R['honest_ci'][0]}, {R['honest_ci'][1]}]` **contains zero** — the corrected metric is "
            f"not fooled — while the **naive** inflation is a firmly-positive +{R['sm_naive_infl']}. "
            "No bootstrap draw's honest inflation reaches the naive point estimate."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the only Sharpe gain is a smoothing artefact (naive "
            f"+{R['sm_naive_infl']}, honest +{R['sm_honest_infl']}); leverage invariant; vol-targeting "
            "negative. Synthetic-only demo — no real tape, so never `REAL`.\n"
            f"- **Tradability `MIRAGE`** — a smoothed Sharpe is unspendable; vol-targeting nets "
            f"{R['vt_net_naive']} after 2 bps; leverage buys risk, not ratio.\n"
            "- **Can engineering fake a Sharpe? `CONFIRMED`** — the *reported* one, by up to 3×."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the vol-targeting lever, net\n\n"
            "The one lever that *actually trades* (vol-targeting) is below baseline and loses to "
            "costs."
        ),
        code(
            "nv = st.net_vol_target(r, target_ann_vol=0.10, cost_bps=2.0)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "bars = [nv['gross_naive_sharpe'], nv['net_naive_sharpe'], st.sharpe_naive(r)]\n"
            "ax.bar(['vol-target\\ngross','vol-target\\nnet (2bps)','raw\\nbaseline'], bars,\n"
            "       color=[AMBER, RED, GREY], width=.55)\n"
            "ax.axhline(st.sharpe_naive(r), ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('naive Sharpe'); ax.set_title('Vol-targeting is below baseline, and costs make it worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {nv['gross_naive_sharpe']:.3f} -> net {nv['net_naive_sharpe']:.3f} \"\n"
            "      f\"(turnover {nv['turnover_per_day']:.3f}/day); raw baseline {st.sharpe_naive(r):.3f}\")"
        ),
        md(
            "> 💡 In plain words: vol-targeting genuinely trades (turnover ~0.03/day), so it pays "
            "costs — and here it starts *below* the raw Sharpe and only sinks. There is nothing to "
            "harvest. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the honest Sharpe just numb to everything, or does it reward a **real** edge? Plant a "
            "genuine alpha and check: the honest raw Sharpe must *rise with the edge*, while the "
            "smoothing game inflates the *naive* Sharpe by the same mechanical amount regardless — "
            "averaged over 25 seeds so no lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, 0.05, 0.10, 0.20]\n"
            "rows = [st.seed_robust_control(data, alpha=a, n_seeds=25) for a in alphas]\n"
            "hr = [x['mean_honest_raw'] for x in rows]\n"
            "ni = [x['mean_naive_inflation'] for x in rows]\n"
            "hi = [x['mean_honest_inflation'] for x in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.plot(alphas, hr, 'o-', c=GREEN, lw=2, label='honest Sharpe of RAW series (tracks real edge)')\n"
            "ax.plot(alphas, ni, 's--', c=RED, lw=2, label='naive inflation from smoothing (the game)')\n"
            "ax.plot(alphas, hi, '^:', c=GREY, lw=2, label='honest inflation from smoothing (~0)')\n"
            "ax.set_xlabel('planted alpha (real edge)'); ax.set_ylabel('Sharpe units')\n"
            "ax.set_title('Honest Sharpe tracks REAL edge; smoothing inflates the naive metric regardless')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, x in zip(alphas, rows):\n"
            "    print(f\"alpha {a:.2f}: honest_raw {x['mean_honest_raw']:.3f} | \"\n"
            "          f\"naive_infl {x['mean_naive_inflation']:.3f} | honest_infl {x['mean_honest_inflation']:.4f}\")"
        ),
        md(
            "The **honest** Sharpe of the raw series climbs 0.30 → 1.37 with the planted edge — the "
            "correction rewards genuine skill. The **naive** smoothing inflation stays large "
            "(~0.22 → 0.93) *whether or not* there's real edge — it's a pure measurement game — while "
            "the **honest** inflation is a negligible ~0.002–0.007 throughout. So the corrected Sharpe "
            "is a faithful detector: it sees through the smoothing game *and* still banks a real edge. "
            "For the *searching* way to fake a Sharpe, see "
            "[344 Backtest-Overfitting](../../344-backtest-overfitting/) and "
            "[589 Genetic-Algo-Overfit](../../589-genetic-algo-overfit/)."
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
