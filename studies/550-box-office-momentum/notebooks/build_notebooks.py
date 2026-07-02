"""Generate the two narrative notebooks for Study 550 (Box-Office-Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a
SYNTHETIC-ONLY study — there is no honest free box-office tape (see ../box_office_momentum/data.py)
— so every figure is drawn live from the deterministic, seeded synthetic engine (offline,
reproducible). There is NO real-tape banner: no cell claims a real-data result. The dict ``R``
below mirrors the frozen headline numbers in docs/results.md as a cross-check.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 550,
# 520 weekly rows; null panel fp 985a49d136c2, positive-control fp 4efecda3764a). Every one of
# these is RECOMPUTED live by the notebook cells; R is the cross-check.
R = dict(
    fp_null="985a49d136c2", fp_planted="4efecda3764a", n=520, seed=550,
    media_naive_t=1.53, media_r2=0.0045, media_ctrl_t=0.65, media_confound_t=6.20,
    media_placebo=0.139,
    spy_naive_t=3.42, spy_ctrl_t=3.11, spy_placebo=0.0005,
    gross_ann=8.76, net_ann=7.53, bh_ann=6.55, net_minus_bh=0.98,
    turnover=24.6, exposure=0.50, hit=0.544,
    # threshold sweep: (threshold, net-minus-bh pp/yr)
    sweep=[(-0.5, -2.0), (-0.25, -2.5), (0.0, 1.0), (0.25, 4.0), (0.5, 2.9)],
    # control curve: (planted beta, mean naive t, mean controlled t) over 25 seeds
    ctrl=[(0.0, 1.21, 0.84), (0.003, 3.29, 2.97), (0.006, 5.37, 5.09),
          (0.009, 7.45, 7.22), (0.012, 9.53, 9.34)],
    planted_beta=0.009, planted_naive_t=7.84, planted_ctrl_t=7.10,
    planted_placebo=0.0005, planted_net_minus_bh=19.5,
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

from box_office_momentum import data, strategy as st

SEED = 550
# The honest folklore case: box office & stocks co-move on a common factor, no TRUE lead planted.
PANEL, TRUTH = data.synthetic_panel(predictive_beta=0.0, seed=SEED)
# A world WHERE a genuine lead IS planted — the positive control.
PANEL_POS, _ = data.synthetic_panel(predictive_beta=0.009, seed=SEED)

# SYNTHETIC-ONLY: there is no real box-office tape a no-key stack can reach honestly.
HAVE_REAL = False
print("synthetic-only study — no real tape (HAVE_REAL =", HAVE_REAL, ")")
print("null panel:", len(PANEL), "weeks, fingerprint", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Box-Office Momentum — do blockbuster weekends foreshadow stock gains? 🎬\n"
            "### Using weekend cinema receipts as a 'consumer mood' signal, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's a tidy-sounding idea. When a movie opens huge and cinemas are packed, people are "
            "clearly feeling confident and spending money — so maybe a **big box-office weekend is an "
            "early signal that consumers are happy**, and media/studio stocks (or even the whole "
            "market) are about to rise. Alt-data desks pitch exactly this.\n\n"
            "We test it the honest way. Because there's no clean, free, decades-long box-office "
            "database a hobbyist can use — and the movie studios themselves keep getting bought and "
            "merged — we build a **realistic simulated world** where box office and stocks behave the "
            "way the real ones do, and ask: *does box office actually **lead** stocks, or do they just "
            "**rise together** for the same reason?*\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on a deterministic simulated world. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does box office **lead** media stocks? | **No.** The predictive link is weak (*t* "
            f"{R['media_naive_t']}) and it basically **vanishes** (*t* {R['media_ctrl_t']}) once you "
            "account for the market moving that week anyway. |\n"
            f"| But doesn't it 'predict' the whole market? | **That's a mirage.** A naive test says yes "
            f"(*t* {R['spy_naive_t']}), but it's just the market's own momentum leaking in — box "
            "office and stocks both ride the same wave of consumer confidence. |\n"
            f"| Could you trade it? | **No.** A signal-timed strategy beats buy-and-hold by only "
            f"~{R['net_minus_bh']:.0f} pp/yr, and that edge **flips to a loss** if you nudge the "
            "trigger. |\n"
            "| Is the test itself broken? | **No** — when we secretly plant a *real* box-office lead, "
            "the engine catches it loud and clear. It just isn't there in the honest case. |\n\n"
            "> The catch is subtle: busy cinemas and rising stocks **happen together** because both "
            "mean 'consumers feel good this week'. Happening together is not the same as one "
            "*predicting* the other — and that gap is the whole story."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A strong weekend box office is a leading indicator of consumer sentiment — so it "
            "foreshadows gains for media stocks and the market.\"*\n\n"
            "It *feels* right. Cinema tickets are discretionary spending; a packed opening weekend "
            "says wallets are open and moods are high. And moods drive markets. So a box-office "
            "**momentum** signal — this weekend's take versus its recent average — ought to point at "
            "next week's returns."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, you'd have a cheap, public, weekly signal that front-runs the market — the "
            "dream of every alt-data shop. The desk has kicked the tyres on lots of these "
            "'sentiment-leads-returns' stories ([257 AAII-Sentiment](../../257-aaii-sentiment/), "
            "[300 Sports-Sentiment](../../300-sports-sentiment/), "
            "[335 Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)). They almost never survive one "
            "simple question: *is the alt-data series **leading** the market, or just **riding** the "
            "same wave?*"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the signal.** Each week, how strong is the box office versus its recent norm? "
            "Call that *box-office momentum*.\n"
            "2. **Line it up with the FUTURE.** Compare this week's box-office momentum to *next* "
            "week's stock return — never the same week (that would be cheating).\n"
            "3. **The honest twist.** Also ask: does the box-office link survive once we account for "
            "the market simply moving that week? If the 'lead' is really just co-movement, it "
            "disappears here.\n\n"
            "*Why simulated?* There's no free, clean, survivorship-honest box-office history, and the "
            "studios you'd trade (Fox, Time Warner, Viacom…) have been merged away. So we simulate a "
            "world with the *same structure* — a shared 'consumer confidence' wave that lifts both "
            "cinemas and stocks — and test the claim there. A simulated-only study can never earn the "
            "top 'REAL' grade; the best it can reach is 'None' or 'Weak'."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Does this week's box office predict next week's media-stock return?** Each dot is a "
            "week; the line is the best-fit predictive relationship."
        ),
        code(
            "reg = st.predictive_regression(PANEL, 'media_fwd')\n"
            "x = PANEL['bo_mom']; y = PANEL['media_fwd']*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=14, color=GREY, alpha=.6)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (reg['intercept'] + reg['slope']*xs)*100, c=RED, lw=2)\n"
            "ax.set_xlabel('box-office momentum this week (standardised)')\n"
            "ax.set_ylabel('media-stock return NEXT week %')\n"
            "ax.set_title(f\"Barely a tilt: slope t = {reg['slope_t']:+.2f} (needs |t|>2 to matter), R2 = {reg['r2']:.4f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"predictive slope t = {reg['slope_t']:+.2f}, R2 = {reg['r2']:.4f} \"\n"
            "      f\"(box office explains ~{reg['r2']*100:.1f}% of next-week returns)\")"
        ),
        md(
            f"The cloud is basically round — the line is almost flat. The predictive *t* is "
            f"**{R['media_naive_t']}**, short of the |*t*| > 2 you'd need to take it seriously, and box "
            f"office explains under **1%** of next-week returns. Already unpromising."
        ),
        md(
            "**Now the honest twist.** A naive test can be fooled: box office and stocks both rise in "
            "confident weeks, and confident weeks cluster together. So let's compare the box-office "
            "'lead' *before* and *after* we account for the market simply moving that week."
        ),
        code(
            "naive = st.predictive_regression(PANEL, 'media_fwd')['slope_t']\n"
            "ctrl = st.controlled_regression(PANEL, 'media_fwd')['slope_t']\n"
            "fig, ax = plt.subplots(figsize=(7, 4.3))\n"
            "ax.bar(['naive\\n(looks like a lead)','after accounting\\nfor the market'], [naive, ctrl],\n"
            "       color=[AMBER, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t| = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('predictive t-stat'); ax.legend()\n"
            "ax.set_title('The tiny apparent lead collapses once you remove the market move')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'box-office lead: naive t {naive:+.2f}  ->  after market control t {ctrl:+.2f}')"
        ),
        md(
            f"> 🔬 For the quants: the box-office slope-*t* falls from **{R['media_naive_t']}** to "
            f"**{R['media_ctrl_t']}** once the contemporaneous market return is included — the "
            "co-movement *was* the market, not a lead. The quants notebook shows the placebo test and "
            "the broad-market 'mirage' too."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The media-stock lead is weak (*t* {R['media_naive_t']}) and "
            f"collapses (*t* {R['media_ctrl_t']}) once you account for the market; the 'whole-market' "
            "version is a co-movement mirage; and there's no honest real tape anyway.\n"
            f"- **Tradability — Mirage.** A signal-timer beats buy-and-hold by only ~"
            f"{R['net_minus_bh']:.0f} pp/yr and flips to a loss if you tweak the trigger. Nothing to "
            "bank."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You'd go long media stocks in weeks the box office looks strong, and sit in cash "
            "otherwise. Let's race that against just buying and holding."
        ),
        code(
            "s = st.timed_strategy(PANEL, 'media_fwd', threshold=0.0, cost_bps=5.0)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['buy & hold','box-office\\ntimer (net)'], [s['bh_ann']*100, s['net_ann']*100],\n"
            "       color=[GREY, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title(f\"A wafer-thin edge: +{ (s['net_ann']-s['bh_ann'])*100:.1f} pp/yr, and fragile\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"buy&hold {s['bh_ann']*100:+.2f}%/yr | timer net {s['net_ann']*100:+.2f}%/yr \"\n"
            "      f\"| edge {(s['net_ann']-s['bh_ann'])*100:+.2f} pp/yr | ~{s['turnover_ann']:.0f} trades/yr\")"
        ),
        md(
            f"The timer edges buy-and-hold by about **{R['net_minus_bh']:.0f} pp/yr** — and the quants "
            "notebook shows that edge is the *same* market-momentum artifact, and that it turns "
            "**negative** if you move the trigger. That's the definition of a mirage."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is the test rigged to fail?** No — see the quants notebook, where we plant a *real* "
            "box-office lead and watch the engine catch it (*t* well past 2) while staying flat in the "
            "honest case.\n"
            "- **The neighbours.** The same 'sentiment-leads-returns' trap in other clothes: "
            "[257 AAII-Sentiment](../../257-aaii-sentiment/), "
            "[300 Sports-Sentiment](../../300-sports-sentiment/), "
            "[271 Cardboard-Box](../../271-cardboard-box/).\n\n"
            "*Think box office really leads and we missed it? Fork this, wire in a genuine "
            "survivorship-honest box-office tape and a surviving studio basket, and show the lead "
            "clearing t = 2 **after** the market control.*"
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
            "# Box-Office Momentum — a quantitative teardown 🔬\n"
            "### Predictive regression · circular-shift placebo · confound-controlled slope · the SPY mirage · costs & threshold sweep · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether box-office "
            "momentum leads next-week returns of a media basket and SPY — and whether any apparent "
            "lead survives a control for the contemporaneous market.\n\n"
            "> ⚠️ **Not investment advice.** **Synthetic-only** study: there is no survivorship-honest "
            "free box-office tape, so every number is computed live on a deterministic, seeded "
            f"simulated world (null panel fp `{R['fp_null']}`, as-of 2026-06-30). A synthetic-only "
            "study is capped at `WEAK`/`NONE` by house rule. Methods in "
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
            f"| **Signal** | `NONE` | Media lead naive *t* **+{R['media_naive_t']}** (below bar), "
            f"placebo *p* {R['media_placebo']}; **controlled *t* +{R['media_ctrl_t']}** (collapses). "
            f"SPY 'signal' *t* **+{R['spy_naive_t']}** is a common-factor mirage (*t* +{R['spy_ctrl_t']} "
            "controlled). Synthetic-only. |\n"
            f"| **Tradability** | `MIRAGE` | Timer net **+{R['net_ann']}%** vs buy-and-hold "
            f"**+{R['bh_ann']}%** (edge +{R['net_minus_bh']} pp/yr), sign-unstable across thresholds. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but in the honest "
            "case box office does not lead returns — the apparent link is the shared consumer/market "
            "factor, and it does not survive being controlled for."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $m_t$ be box-office momentum at week $t$ and $r_{t+1}$ next-week return.\n\n"
            "- **H₁ (the lead).** $r^{media}_{t+1} = a + b\\, m_t + e$ with $b>0$ at *t* ≥ 2.\n"
            "- **H₂ (survives the confound).** $b$ stays significant after adding the contemporaneous "
            "market return $r^{mkt}_{t+1}$ as a control (a genuine lead is orthogonal to the "
            "same-week market move).\n"
            "- **H₃ (whole tape).** The lead also predicts SPY.\n"
            "- **H₄ (net).** A signal-timed strategy beats buy-and-hold after costs, stably.\n\n"
            "We find **H₁ not met** (*t* below 2), **H₂ rejected** (the slope collapses under the "
            "control), **H₃ a mirage** (a common-factor artifact, not a lead), and **H₄ rejected** "
            "(the edge is thin and sign-unstable)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the confound that kills alt-data leads\n\n"
            "The content is the **why**. Box office and equity returns both load on the same "
            "*contemporaneous* consumer/market factor, and that factor is **persistent** (confident "
            "weeks cluster). So this-week box office correlates with *next*-week returns purely "
            "through the factor's autocorrelation — a **spurious lead**. The honest fix is to control "
            "for the contemporaneous market: a real lead survives, a co-movement artifact does not. "
            "This is the same trap that sinks most 'sentiment-leads-returns' studies "
            "([Da-Engelberg-Gao 2015](../docs/references.md); Novy-Marx 2014)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $m_t$ = standardised box-office momentum, known at the close of week $t$.\n"
            "- **Predictive regression.** OLS of $r_{t+1}$ on $m_t$; the slope's *t* is H₁.\n"
            "- **Placebo.** Circular-shift $m$ against returns (preserving autocorrelation, breaking "
            "the lead alignment) 2000×; read the |slope| tail probability.\n"
            "- **Confound control.** Add the contemporaneous market return; re-read the box-office "
            "slope (H₂).\n"
            "- **Frictions.** Long/flat timer, one-way cost per switch; buy-and-hold benchmark.\n"
            "- **Robustness.** Sweep the signal threshold; check sign stability.\n"
            "- **Positive control.** Plant a genuine lead (`predictive_beta > 0`) and a null, averaged "
            "over 25 seeds (house rule).\n\n"
            "Timing: the signal at week $t$ uses only box office known by that close; the position is "
            "entered at that close and held over week $t+1$ (the `*_fwd` columns). One documented week "
            "of lag; no future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁\n\n"
            "Media next-week return on this-week box-office momentum, with the placebo null."
        ),
        code(
            "reg = st.predictive_regression(PANEL, 'media_fwd')\n"
            "p = st.placebo_pvalue(PANEL, 'media_fwd', n_perm=2000, seed=SEED)\n"
            "x = PANEL['bo_mom']; y = PANEL['media_fwd']*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=14, color=GREY, alpha=.6)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (reg['intercept'] + reg['slope']*xs)*100, c=RED, lw=2)\n"
            "ax.set_xlabel('box-office momentum (std)'); ax.set_ylabel('media return next week %')\n"
            "ax.set_title(f\"slope t {reg['slope_t']:+.2f}, R2 {reg['r2']:.4f}, placebo p {p:.3f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope {reg['slope']:+.5f} (t {reg['slope_t']:+.2f}), R2 {reg['r2']:.4f}, placebo p {p:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: H₁ **not met.** The slope-*t* is **{R['media_naive_t']}** (below 2) "
            f"and the placebo *p* = **{R['media_placebo']}** is consistent with noise. Box office "
            "explains ~0.5% of next-week media returns."
        ),
        md(
            "### 4b · The confound-controlled slope — H₂\n\n"
            "Add the contemporaneous market return. A *genuine* lead keeps its *t*; a co-movement "
            "artifact collapses."
        ),
        code(
            "naive = st.predictive_regression(PANEL, 'media_fwd')\n"
            "ctrl = st.controlled_regression(PANEL, 'media_fwd')\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['naive','+ market control'], [naive['slope_t'], ctrl['slope_t']],\n"
            "       color=[AMBER, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t| = 2'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('box-office slope t'); ax.legend()\n"
            "ax.set_title(f\"Lead collapses: t {naive['slope_t']:+.2f} -> {ctrl['slope_t']:+.2f} \"\n"
            "             f\"(market control t {ctrl['ctrl_t']:+.2f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"naive t {naive['slope_t']:+.2f} -> controlled t {ctrl['slope_t']:+.2f}; \"\n"
            "      f\"contemporaneous-market control t {ctrl['ctrl_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected.** The box-office slope-*t* falls from "
            f"**{R['media_naive_t']}** to **{R['media_ctrl_t']}** once the same-week market return "
            f"(control *t* **+{R['media_confound_t']}**) is in the regression. The co-movement *was* "
            "the market."
        ),
        md(
            "### 4c · The SPY mirage — H₃ (and why a placebo alone isn't enough)\n\n"
            "Now aim the same signal at the *broad tape* (SPY). Watch a naive test — and even the "
            "placebo — get fooled by the persistent common factor."
        ),
        code(
            "sreg = st.predictive_regression(PANEL, 'spy_fwd')\n"
            "sctrl = st.controlled_regression(PANEL, 'spy_fwd')\n"
            "sp = st.placebo_pvalue(PANEL, 'spy_fwd', n_perm=2000, seed=SEED)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['naive','+ market control'], [sreg['slope_t'], sctrl['slope_t']],\n"
            "       color=[RED, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t| = 2'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('SPY box-office slope t'); ax.legend()\n"
            "ax.set_title(f\"'Signal' t {sreg['slope_t']:+.2f} (placebo p {sp:.4f}) — a common-factor artifact\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"SPY naive t {sreg['slope_t']:+.2f}, placebo p {sp:.4f}, controlled t {sctrl['slope_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: H₃ is a **mirage.** The SPY 'signal' looks strong (naive *t* "
            f"**{R['spy_naive_t']}**) and even *passes* the placebo (*p* **{R['spy_placebo']}**) — yet "
            f"we planted **zero** true lead. It is pure persistent-common-factor co-movement, and an "
            f"imperfect market control leaves it inflated (*t* **{R['spy_ctrl_t']}**). Lesson: a "
            "single placebo is not proof; you need a structural control."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ not met (media *t* {R['media_naive_t']}, placebo "
            f"{R['media_placebo']}); H₂ rejected (controlled *t* {R['media_ctrl_t']}); H₃ a "
            "common-factor mirage; synthetic-only caps it at `NONE`.\n"
            f"- **Tradability `MIRAGE`** — net edge +{R['net_minus_bh']} pp/yr, sign-unstable across "
            "thresholds."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the threshold sweep\n\n"
            "The timer's edge is thin, and it flips sign when you nudge the trigger."
        ),
        code(
            "sw = st.robustness_sweep(PANEL, 'media_fwd')\n"
            "s0 = st.timed_strategy(PANEL, 'media_fwd', 0.0, 5.0)\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax1.bar(['buy&hold','timer net'], [s0['bh_ann']*100, s0['net_ann']*100], color=[GREY, AMBER], width=.5)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_ylabel('annualised %'); ax1.set_title(f\"edge +{(s0['net_ann']-s0['bh_ann'])*100:.1f} pp/yr\")\n"
            "nb = sw['net_minus_bh']*100\n"
            "ax2.bar([str(t) for t in sw.index], nb, color=[GREEN if v>0 else RED for v in nb], width=.6)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_xlabel('signal threshold'); ax2.set_ylabel('net - buy&hold pp/yr')\n"
            "ax2.set_title('Edge flips sign across thresholds')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net-minus-bh by threshold (pp/yr):', [round(v,1) for v in nb])"
        ),
        md(
            f"> 💡 In plain words: the ~{R['net_minus_bh']} pp/yr edge is the same factor-persistence "
            "artifact, and it turns **negative** at looser thresholds. No stable, tradable edge — "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a *genuine* "
            "box-office lead (`predictive_beta > 0`) of increasing strength and watch the predictive "
            "*t* climb — **and survive the market control** (a real lead is orthogonal to the "
            "same-week market) — averaged over 25 seeds so no lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.003, 0.006, 0.009, 0.012]\n"
            "naive_ts = [st.synthetic_mean_t(data, b, n_seeds=25, controlled=False) for b in betas]\n"
            "ctrl_ts  = [st.synthetic_mean_t(data, b, n_seeds=25, controlled=True)  for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.plot(betas, naive_ts, 'o-', c=AMBER, lw=2, label='naive')\n"
            "ax.plot(betas, ctrl_ts, 's--', c=GREEN, lw=2, label='after market control')\n"
            "ax.axhline(2, ls=':', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted predictive_beta (0 = null)'); ax.set_ylabel('mean predictive t (25 seeds)')\n"
            "ax.set_title('The engine banks a REAL lead (and it survives the control); flat at the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, n, c in zip(betas, naive_ts, ctrl_ts): print(f'beta {b:+.3f} -> naive t {n:+.2f} | controlled t {c:+.2f}')"
        ),
        md(
            "The key contrast with the honest case: a **planted** lead keeps its *t* under the market "
            "control (naive ≈ controlled), because a genuine lead is orthogonal to the same-week "
            "market — whereas the honest-case media 'lead' *collapsed* under that control. Let's also "
            "verify the full positive-control world (planted `β = 0.009`) end-to-end:"
        ),
        code(
            "reg = st.predictive_regression(PANEL_POS, 'media_fwd')\n"
            "ctrl = st.controlled_regression(PANEL_POS, 'media_fwd')\n"
            "pp = st.placebo_pvalue(PANEL_POS, 'media_fwd', n_perm=2000, seed=SEED)\n"
            "sp = st.timed_strategy(PANEL_POS, 'media_fwd', 0.0, 5.0)\n"
            "print(f\"PLANTED beta=0.009: naive t {reg['slope_t']:+.2f}, controlled t {ctrl['slope_t']:+.2f}, \"\n"
            "      f\"placebo p {pp:.4f}, timer edge {(sp['net_ann']-sp['bh_ann'])*100:+.1f} pp/yr\")\n"
            "assert reg['slope_t'] > 2 and ctrl['slope_t'] > 2, 'engine should catch a planted lead'\n"
            "print('OK — the detector fires on a real lead and stays flat at the null.')"
        ),
        md(
            f"The planted world gives naive *t* **+{R['planted_naive_t']}**, controlled *t* "
            f"**+{R['planted_ctrl_t']}**, placebo *p* **{R['planted_placebo']}**, and a "
            f"**+{R['planted_net_minus_bh']} pp/yr** timer edge — the engine is faithful. So the "
            "honest-case `NONE` is the *structure of the claim* talking (box office and stocks share a "
            "factor; that isn't a lead), not a broken detector. And with no survivorship-honest free "
            "box-office tape, the study can never rise above `NONE`. Neighbours: "
            "[257 AAII-Sentiment](../../257-aaii-sentiment/), "
            "[300 Sports-Sentiment](../../300-sports-sentiment/), "
            "[271 Cardboard-Box](../../271-cardboard-box/)."
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
