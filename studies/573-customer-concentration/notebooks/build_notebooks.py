"""Generate the two narrative notebooks for Study 573 (Customer-Concentration).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This study is
**synthetic-only** — there is no free customer-concentration tape — so every cell runs offline and
deterministic on the seeded synthetic panel. The dict ``R`` mirrors the headline numbers in
docs/results.md; the cells RECOMPUTE those numbers live from the same seed, so a reader re-running
the notebook reproduces the fingerprinted headline exactly.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 573; 400
# firms; panel fp bc1d5db4bfa5; effect planted vol_beta=0.35, ret_alpha=0.04).
R = dict(
    fp_panel="bc1d5db4bfa5", n=400, k=120, seed=573,
    vol_beta=0.35, ret_alpha=0.04,
    div_vol=21.2, conc_vol=24.5, vol_spread=3.3, vol_t=29.4,
    div_ret=7.2, conc_ret=12.1, ret_spread=4.9, ret_t=1.61, placebo_p=0.102,
    net=3.7, cost_bps=5.0, borrow_bps=100.0,
    vol_slope=0.076, vol_slope_t=53.5, vol_corr=0.94,
    ret_slope=0.087, ret_slope_t=1.38, ret_corr=0.07,
    # robustness: (frac_label, vol_spread_pp, vol_t, ret_spread_pct, ret_t)
    sweep=[("decile 0.10", 4.7, 30.1, -1.3, -0.23), ("quintile 0.20", 3.9, 30.0, 3.1, 0.85),
           ("tercile 0.30", 3.3, 29.4, 4.9, 1.61), ("broad 0.40", 2.8, 27.6, 5.3, 2.05)],
    # vol control: (vol_beta, mean vol slope-t over 25 seeds)
    ctrl_vol=[(0.0, 0.21), (0.15, 22.3), (0.35, 51.8), (0.60, 88.7)],
    # ret control: (ret_alpha, mean ret slope-t over 25 seeds)
    ctrl_ret=[(0.0, 0.15), (0.02, 0.46), (0.04, 0.78), (0.08, 1.41), (0.12, 2.05)],
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

from customer_concentration import data, strategy as st

# SYNTHETIC-ONLY: no real customer-concentration tape exists (paywalled 10-K / Compustat segments).
# The headline panel plants both effects at literature-plausible strengths; all cells are offline.
PANEL, TRUTH = data.synthetic_panel()   # seed 573, vol_beta=0.35, ret_alpha=0.04
print("synthetic panel fp:", data.fingerprint(PANEL), "| n =", len(PANEL),
      "| planted vol_beta =", TRUTH["vol_beta"], ", ret_alpha =", TRUTH["ret_alpha"])
print("NOTE: synthetic-only study -> Signal capped at WEAK (no real tape to certify against).")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Customer Concentration — is a firm that leans on a few big customers riskier? 🔗\n"
            "### And if it is, does the stock market actually pay you for that risk? In plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Imagine a small supplier that sells **80% of everything it makes to one giant customer** "
            "— say a parts maker whose whole business is one carmaker. If that customer walks, "
            "renegotiates, or goes bust, the supplier is in deep trouble. Its cash flows are *lumpy* "
            "and *fragile*. Accountants and academics call this **customer concentration**, and the "
            "claim is that concentrated firms are **riskier** — and, if markets are efficient, that "
            "you should be **paid a premium** for holding that extra risk.\n\n"
            "We test both halves: does concentration raise a firm's forward **risk**, and does it "
            "come with a return **premium**?\n\n"
            "> ⚠️ **Heads-up: this is a synthetic study.** There is no free, public dataset of who "
            "each company's big customers are (it's buried in paywalled filings). So we build a "
            "*realistic simulated world* where we can dial the effect up and down, and see what an "
            "honest test would find. That means the best this study can earn is **Weak** — never "
            "**Real** (which needs a real market tape).\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> House style in [METHODOLOGY.md](../../../METHODOLOGY.md). **Not investment advice.**"
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Are concentrated firms *riskier* (more volatile)? | **Yes, clearly.** The most "
            f"concentrated third had forward volatility **{R['conc_vol']}%** vs **{R['div_vol']}%** "
            "for the most diversified third. The fragility is real. |\n"
            f"| Did they *earn more* (a premium)? | **A little, but it's noisy.** The concentrated "
            f"third returned **+{R['conc_ret']}%** vs **+{R['div_ret']}%** — a **+{R['ret_spread']}%** "
            f"gap in the right direction, but a shuffle test says the odds it's just luck are "
            f"*p* = {R['placebo_p']}. Not convincing. |\n"
            "| Could you trade the premium? | **No.** There's no index of 'concentrated firms' to "
            "buy, and the return gap is too shaky to bet on — it even vanishes at some cut points. |\n\n"
            "> So: concentration looks like a genuine *risk*, but the *reward* for bearing it is hard "
            "to see — even in a world where we deliberately planted one. Riskier, yes; visibly paid "
            "for it, no."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms whose revenue depends on a few large customers bear more risk — and that risk "
            "should be priced.\"* — the customer-concentration literature (Patatoukas 2012; Dhaliwal, "
            "Judd, Serfling & Shaikh 2016; Hertzel et al. 2008).\n\n"
            "The intuition: one big customer is a single point of failure. Lose it and revenue "
            "collapses. Distress even travels *up* the supply chain — when a big buyer stumbles, its "
            "concentrated suppliers get hurt. So a concentrated firm's future is *lumpier*: bigger "
            "swings, fatter tails. The textbook says risk should be **rewarded**, so these fragile "
            "firms *ought* to earn a premium. (Some researchers argue the opposite — a *discount* — "
            "if investors under-price the tail. The sign is genuinely up for grabs.)"
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If concentration is a priced risk, it's a *fundamental* factor you could tilt toward "
            "(harvest the premium) or away from (avoid the fragility). It also matters for anyone "
            "analysing a single stock: a supplier riding on one customer deserves a fatter discount "
            "rate. The desk has looked at balance-sheet fragility "
            "([540 Distress-Risk](../../540-distress-risk-anomaly/)); this is a different, "
            "*demand-side* kind of fragility."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score concentration.** For each firm, how lopsided is its customer base? One "
            "dominant customer → a high score; many small ones → a low score. (Formally a Herfindahl "
            "index of customer revenue shares.)\n"
            "2. **Sort.** Split the firms into a *diversified* third and a *concentrated* third.\n"
            "3. **Compare the future.** Did the concentrated third have (a) higher forward "
            "**volatility** — the risk claim — and (b) higher forward **returns** — the premium "
            "claim?\n\n"
            "Because there's no free real customer data, we do this on a **simulated 400-firm world** "
            "where we control the truth. In the headline run we planted *both* effects (a real risk "
            "bump and a real premium) — then we ask whether an honest test can even *see* them.\n\n"
            "*Timing:* concentration is measured *today* (the last filing); risk and return are "
            "measured over the *following* period. No peeking ahead."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First the risk claim: were the concentrated firms more volatile?**"
        ),
        code(
            "b = st.bucket_stats(PANEL, frac=0.3)\n"
            "div_v, conc_v, vspr, vt = b['div_vol']*100, b['conc_vol']*100, b['vol_spread']*100, st.vol_tstat(b)['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['DIVERSIFIED third','CONCENTRATED third'], [div_v, conc_v], color=[GREEN, RED], width=.5)\n"
            "ax.set_ylabel('forward volatility %')\n"
            "ax.set_title(f'Concentrated firms were noticeably more volatile ({conc_v:.1f}% vs {div_v:.1f}%)')\n"
            "fig.suptitle('synthetic world (400 firms, effect planted)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Diversified vol {div_v:.1f}% | Concentrated vol {conc_v:.1f}% | spread +{vspr:.1f}pp (t {vt:+.1f})')"
        ),
        md(
            f"There's the risk story, loud and clear: the concentrated third's forward vol is "
            f"**{R['conc_vol']}%** against **{R['div_vol']}%** for the diversified third — a "
            f"**+{R['vol_spread']}pp** gap with a huge *t* ({R['vol_t']:+.0f}). Depending on a few big "
            "customers really does make the ride bumpier."
        ),
        md("**Now the payoff claim: did the riskier firms actually *earn* more?**"),
        code(
            "div_r, conc_r, rspr = b['div_ret']*100, b['conc_ret']*100, b['ret_spread']*100\n"
            "rt = st.return_tstat(b)['t']; p = st.placebo_pvalue(PANEL, n_perm=2000, seed=573)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['DIVERSIFIED third','CONCENTRATED third'], [div_r, conc_r], color=[GREEN, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'A premium in the right direction (+{conc_r:.1f}% vs +{div_r:.1f}%) — but is it real?')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Diversified +{div_r:.1f}% | Concentrated +{conc_r:.1f}% | spread +{rspr:.1f}% (t {rt:+.2f}, shuffle p {p:.3f})')"
        ),
        md(
            f"The premium is *there* in direction — **+{R['conc_ret']}%** vs **+{R['div_ret']}%** — but "
            f"look at the *t* ({R['ret_t']:+.2f}) and the shuffle-test *p* ({R['placebo_p']}). Those "
            "say: *could easily be luck.* Even though we **planted a real premium**, it's hard to see, "
            "because the same fragility that raises risk also makes the returns jump around — the "
            "signal drowns in its own noise."
        ),
        md("**Does the answer depend on how we cut the buckets?**"),
        code(
            "rows = st.robustness_sweep(PANEL)\n"
            "labs = ['decile','quintile','tercile','broad']\n"
            "rspr = [r['ret_spread']*100 for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in rspr]\n"
            "ax.bar(labs, rspr, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('concentrated - diversified return %')\n"
            "ax.set_title('The return premium wobbles — even negative at the extremes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('return spreads by cut:', [round(s,1) for s in rspr])"
        ),
        md(
            "At the extreme deciles the 'premium' actually goes **negative**, then turns positive for "
            "wider buckets. A payoff that flips sign depending on where you draw the line is not "
            "something you can lean on. (The *risk* gap, by contrast, is rock-solid at every cut — "
            "see the quant notebook.)"
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The *risk* half is strong and stable (concentrated firms are "
            "clearly more volatile). But the *return* half is soft even where we planted it "
            f"(*t* {R['ret_t']:+.2f}, shuffle *p* {R['placebo_p']}), and there's no real market tape "
            "to certify anything — so, capped at Weak.\n"
            "- **Tradability — Mirage.** No investable 'concentration' index exists, and the premium "
            "is too shaky and buried in noise to trade.\n\n"
            "> Riskier? Plausibly yes. Paid for it? Not visibly."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not really. You'd have to (a) build the customer-concentration data yourself from "
            "thousands of filings, (b) somehow buy a basket of 'concentrated' firms and short "
            "'diversified' ones, and (c) hope the premium — which our honest test can barely detect "
            "and which flips sign across cut points — actually shows up. Costs then nibble the "
            "already-soft gap (gross vs net in the quant notebook). No free lunch here."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The risk vs the reward.** The lesson generalises: a factor can be a genuine *risk* "
            "and still not deliver a *tradable premium* — the reward can be real but too small and "
            "noisy to harvest.\n"
            "- **Cousins on the desk.** [540 Distress-Risk](../../540-distress-risk-anomaly/) is "
            "balance-sheet fragility; [177 Megacap-Concentration](../../177-megacap-concentration/) "
            "is *index*-level concentration.\n\n"
            "*Have real customer-concentration data (Compustat segments / hand-collected 10-Ks)? "
            "Fork this, drop it in, and see whether the return premium clears t = 2 on a real tape.*"
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
            "# Customer Concentration — a quantitative teardown 🔬\n"
            "### Herfindahl concentration score · tercile sort · two-sample *t* on vol AND return · label-shuffle placebo · firm-level slopes · tail-fraction robustness · costs & borrow · two-leg synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test the "
            "customer-concentration fundamental-risk claim on **two** legs: does concentration raise "
            "forward **volatility** (the robust part of the literature), and does it carry a return "
            "**premium** (the priced-risk part)?\n\n"
            "> ⚠️ **Synthetic-only.** There is no free customer-concentration tape (the measure is in "
            "paywalled 10-K major-customer / Compustat segment data), so the Signal axis is capped at "
            f"`WEAK` by house rule. All cells run offline on the seeded panel (fp `{R['fp_panel']}`, "
            "effect planted). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | RISK leg strong: vol spread **+{R['vol_spread']}pp** (two-sample "
            f"*t* **+{R['vol_t']}**), firm slope-*t* **+{R['vol_slope_t']}** (corr +{R['vol_corr']}). "
            f"RETURN leg soft *even planted*: spread **+{R['ret_spread']}%**, *t* **+{R['ret_t']}**, "
            f"placebo *p* {R['placebo_p']}, slope-*t* **+{R['ret_slope_t']}**. No real tape ⇒ capped WEAK. |\n"
            f"| **Tradability** | `MIRAGE` | No investable concentration index; return spread gross "
            f"**+{R['ret_spread']}%** → net **+{R['net']}%** (5 bps/leg + {R['borrow_bps']:.0f} bps "
            "borrow), on placebo *p* 0.10 and sign-unstable across cuts. |\n\n"
            "> 💡 In plain words: the engine works (the two-leg control proves it), and concentration "
            "genuinely raises modeled risk — but the *return* premium is nearly undetectable even "
            "where it's planted, because concentration inflates the return's own variance."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $c_i$ be firm $i$'s customer-concentration score, $\\sigma_i$ its forward volatility, "
            "and $r_i$ its forward return.\n\n"
            "- **H_vol (risk).** $\\mathbb{E}[\\sigma \\mid \\text{conc}] - "
            "\\mathbb{E}[\\sigma \\mid \\text{div}] > 0$ at *t* ≥ 2 (concentrated firms are more "
            "volatile). The robust leg of the literature.\n"
            "- **H_ret (premium).** $\\mathbb{E}[r \\mid \\text{conc}] - "
            "\\mathbb{E}[r \\mid \\text{div}] > 0$ at *t* ≥ 2 (the fragility is *priced*).\n"
            "- **H_sign (firm-level).** slope of $\\sigma$ on $c$ > 0 (risk) and slope of $r$ on $c$ "
            "> 0 (premium; a negative slope would be the *discount* story).\n"
            "- **H_robust.** the signs are stable across tail fractions.\n"
            "- **H_net.** H_ret survives costs + the short-leg borrow.\n\n"
            "On the headline (effect-planted) synthetic panel we find **H_vol strongly supported**, "
            "**H_ret NOT supported** (right sign but *t* ≈ 1.6, placebo *p* 0.10), **H_sign** split "
            "(vol slope hugely positive, return slope insignificant), **H_robust** failed for the "
            "return leg (sign flips), and **H_net** moot (the gross return leg isn't significant to "
            "begin with)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why the return leg is hard**. In the model (and, we argue, in "
            "reality) the *same* customer-concentration that raises a firm's cash-flow volatility also "
            "widens the dispersion of its realised returns. So even a genuinely priced premium sits "
            "inside a much larger idiosyncratic cloud, and the cross-sectional slope barely clears "
            "noise. A factor can be a real *risk* and still not a harvestable *return* — the desk's "
            "recurring lesson, shared with [540 Distress-Risk](../../540-distress-risk-anomaly/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** $c$ = Herfindahl-style customer-concentration index (higher = more "
            "concentrated). On the synthetic tape, the ``concentration`` column.\n"
            "- **Sort.** Tercile tails (~120 names): diversified (low $c$) vs concentrated (high $c$).\n"
            "- **Inference.** Two-sample (Welch) *t* on the concentrated − diversified forward "
            "**vol** and forward **return**; a **label-shuffle placebo** null on the return spread "
            "(2000 perms).\n"
            "- **Firm-level.** OLS of forward vol on $c$ (slope > 0 = risk) and forward return on $c$ "
            "(sign = premium vs discount).\n"
            "- **Robustness.** Re-sort at decile / quintile / tercile / broad tails; read the signs.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the shorted (diversified) "
            "leg.\n"
            "- **Positive control.** Deterministic worlds with a planted risk effect (`vol_beta`) and "
            "a planted premium (`ret_alpha`), and a null — each averaged over 25 seeds (house rule).\n\n"
            "Timing: concentration is an as-of characteristic (last filing); vol and return are "
            "measured over the subsequent window — one documented execution lag, separate columns in "
            "the panel, so no look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The risk leg — H_vol\n\n"
            "Concentrated vs diversified forward volatility, with the two-sample *t*."
        ),
        code(
            "b = st.bucket_stats(PANEL, 0.3)\n"
            "div_v, conc_v, vspr, vt = b['div_vol']*100, b['conc_vol']*100, b['vol_spread']*100, st.vol_tstat(b)['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['DIVERSIFIED','CONCENTRATED'], [div_v, conc_v], color=[GREEN, RED], width=.5)\n"
            "ax.set_ylabel('forward volatility %')\n"
            "ax.set_title(f'Concentrated - diversified vol = +{vspr:.1f}pp  (two-sample t {vt:+.1f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'div vol {div_v:.1f}% | conc vol {conc_v:.1f}% | spread +{vspr:.1f}pp | t {vt:+.1f}')"
        ),
        md(
            f"> 💡 In plain words: H_vol **supported, emphatically.** The concentrated tercile's "
            f"forward vol is **{R['conc_vol']}%** vs **{R['div_vol']}%** (+{R['vol_spread']}pp, "
            f"*t* +{R['vol_t']}). Concentration is a genuine risk characteristic."
        ),
        md(
            "### 4b · The return leg — H_ret\n\n"
            "Concentrated vs diversified forward return, with the two-sample *t* and the label-shuffle "
            "placebo."
        ),
        code(
            "div_r, conc_r, rspr = b['div_ret']*100, b['conc_ret']*100, b['ret_spread']*100\n"
            "rt = st.return_tstat(b)['t']; p = st.placebo_pvalue(PANEL, n_perm=2000, seed=573)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['DIVERSIFIED','CONCENTRATED'], [div_r, conc_r], color=[GREEN, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Concentrated - diversified return = +{rspr:.1f}%  (t {rt:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'div ret +{div_r:.1f}% | conc ret +{conc_r:.1f}% | spread +{rspr:.1f}% | t {rt:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H_ret **not supported.** The premium is the right sign "
            f"(+{R['ret_spread']}%) but *t* is only **+{R['ret_t']}** and the placebo *p* = "
            f"{R['placebo_p']} — indistinguishable from noise. And this is the world where we "
            "*planted* a premium."
        ),
        md(
            "### 4c · Firm-level slopes — H_sign\n\n"
            "Regress forward vol on concentration (risk) and forward return on concentration "
            "(premium/discount). The scatter shows *why* the return leg is hard."
        ),
        code(
            "vr = st.vol_regression(PANEL); rr = st.return_regression(PANEL)\n"
            "c = st.concentration_score(PANEL)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.scatter(c, PANEL['forward_vol']*100, s=10, color=GREY)\n"
            "xs = np.linspace(c.min(), c.max(), 50)\n"
            "a1.plot(xs, (vr['slope']*xs + (PANEL['forward_vol'].mean() - vr['slope']*c.mean()))*100, c=RED, lw=2)\n"
            "a1.set_xlabel('concentration'); a1.set_ylabel('forward vol %')\n"
            "a1.set_title(f'RISK: slope-t {vr[\"slope_t\"]:+.1f}, corr {vr[\"corr\"]:+.2f}')\n"
            "a2.scatter(c, PANEL['forward_ret']*100, s=10, color=GREY)\n"
            "a2.plot(xs, (rr['slope']*xs + (PANEL['forward_ret'].mean() - rr['slope']*c.mean()))*100, c=AMBER, lw=2)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xlabel('concentration'); a2.set_ylabel('forward return %')\n"
            "a2.set_title(f'RETURN: slope-t {rr[\"slope_t\"]:+.2f}, corr {rr[\"corr\"]:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'vol  slope {vr[\"slope\"]:+.3f} t {vr[\"slope_t\"]:+.1f} corr {vr[\"corr\"]:+.2f}')\n"
            "print(f'ret  slope {rr[\"slope\"]:+.3f} t {rr[\"slope_t\"]:+.2f} corr {rr[\"corr\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the left scatter is a tight line (risk corr +{R['vol_corr']}); the "
            f"right is a fog (return corr +{R['ret_corr']}). The premium slope-*t* is only "
            f"**+{R['ret_slope_t']}** — the planted premium is buried in the return dispersion that "
            "concentration *itself* inflates. Risk you can see; reward you can't."
        ),
        md(
            "### 4d · Robustness — H_robust (do the signs hold across cuts?)\n\n"
            "Re-sort at four tail fractions."
        ),
        code(
            "rows = st.robustness_sweep(PANEL)\n"
            "tab = pd.DataFrame(rows)\n"
            "tab['vol_spread'] *= 100; tab['ret_spread'] *= 100\n"
            "labs = ['decile','quintile','tercile','broad']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(labs, tab['vol_t'], color=RED, width=.55); a1.axhline(2, ls='--', c=GREY)\n"
            "a1.set_ylabel('vol two-sample t'); a1.set_title('RISK t: huge and stable everywhere')\n"
            "cols = [GREEN if s>0 else RED for s in tab['ret_t']]\n"
            "a2.bar(labs, tab['ret_t'], color=cols, width=.55); a2.axhline(2, ls='--', c=GREY); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('return two-sample t'); a2.set_title('RETURN t: wobbles, sign-flips, ~2 at best')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab[['frac','vol_spread','vol_t','ret_spread','ret_t']].round(2).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H_robust **holds for risk, fails for return.** The vol *t* is ~28–30 "
            "at every cut; the return *t* swings from **−0.23** (deciles) to **+2.05** (broad) — "
            "sign-unstable and marginal at best. A premium that depends on where you draw the bucket "
            "line is not bankable."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H_vol strongly supported (spread +{R['vol_spread']}pp, "
            f"*t* +{R['vol_t']}); H_ret rejected (spread +{R['ret_spread']}%, *t* +{R['ret_t']}, "
            f"placebo *p* {R['placebo_p']}); H_robust fails for the return leg; and **no real tape "
            "exists** ⇒ capped at WEAK.\n"
            f"- **Tradability `MIRAGE`** — no investable concentration index; gross +{R['ret_spread']}% "
            f"→ net +{R['net']}%, on a non-significant, sign-unstable premium."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Charge frictions on the (already non-significant) return long-short."
        ),
        code(
            "gross = b['ret_spread']*100\n"
            "net = st.net_return_spread(b, cost_bps=5.0, borrow_ann_bps=100.0, holding_years=1.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('concentrated - diversified return %')\n"
            "ax.set_title(f'Soft to begin with ({gross:+.1f}%), softer after costs ({net:+.1f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 100 bps borrow, 1y hold)')"
        ),
        md(
            "> 💡 In plain words: costs shave the spread, but the binding problem is upstream — the "
            "gross premium isn't statistically there (placebo *p* 0.10). `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the two-leg synthetic positive control\n\n"
            "Is the engine a faithful detector? Plant each effect separately, at growing strength, "
            "averaged over 25 seeds so no lucky seed can fake it. The **risk** detector should light "
            "up fast; the **return** detector should need a *large* planted premium — proving the "
            "weak real-leg reading is about noise, not a broken engine."
        ),
        code(
            "vbetas = [0.0, 0.10, 0.15, 0.25, 0.35, 0.50, 0.60]\n"
            "vts = [st.synthetic_vol_mean_t(data, vol_beta=vb, ret_alpha=0.0, n_seeds=25) for vb in vbetas]\n"
            "ralphas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]\n"
            "rts = [st.synthetic_ret_mean_t(data, ret_alpha=ra, vol_beta=0.0, n_seeds=25) for ra in ralphas]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(vbetas, vts, 'o-', c=RED, lw=2); a1.axhline(2, ls='--', c=GREY, label='t=2')\n"
            "a1.set_xlabel('planted vol_beta'); a1.set_ylabel('mean vol slope-t (25 seeds)')\n"
            "a1.set_title('RISK detector: lights up immediately'); a1.legend()\n"
            "a2.plot(ralphas, rts, 'o-', c=AMBER, lw=2); a2.axhline(2, ls='--', c=GREY, label='t=2'); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xlabel('planted ret_alpha'); a2.set_ylabel('mean ret slope-t (25 seeds)')\n"
            "a2.set_title('RETURN detector: needs a LARGE premium'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for vb, t in zip(vbetas, vts): print(f'vol_beta {vb:+.2f} -> mean vol slope-t {t:+.2f}')\n"
            "print('---')\n"
            "for ra, t in zip(ralphas, rts): print(f'ret_alpha {ra:+.2f} -> mean ret slope-t {t:+.2f}')"
        ),
        md(
            "Both detectors sit at *t* ≈ 0 at the null (no false signal). The **risk** slope-*t* blows "
            f"past 2 almost immediately; the **return** slope-*t* only reaches ~2 near "
            f"`ret_alpha ≈ 0.12` — a *large* premium — under realistic dispersion. So the engine is "
            "faithful: the weak real-leg reading reflects a genuinely hard-to-detect premium (the "
            "fragility inflates return variance), not a broken test. Concentration is a real *risk*; "
            "its *reward* is, at best, `WEAK` and untradable. See "
            "[540 Distress-Risk](../../540-distress-risk-anomaly/) for the balance-sheet-fragility "
            "cousin."
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
