"""Generate the two narrative notebooks for Study 555 (OpenTable-Reservations).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a **synthetic-only**
study: there is no free real seated-diners tape, so every figure runs on the deterministic, seeded
synthetic world in ../opentable_reservations/data.py — offline and reproducible for any reader. The
dict ``R`` mirrors the frozen headline numbers in docs/results.md (the demonstrator world, nowcast
planted at nowcast_beta = 0.35, seed 555).
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


# Frozen headline numbers — mirror of docs/results.md (synthetic demonstrator, seed 555, 312 weeks,
# nowcast planted at nowcast_beta = 0.35; panel fp 2491c27d3bc8; as-of 2026-06-30).
R = dict(
    fp_panel="2491c27d3bc8", fp_null="02902c940703",
    n_weeks=312, aligned_n=259, beta=0.35,
    slope=0.0050, slope_t=4.49, placebo_p=0.0005, r2=0.55,
    lf_gross=10.3, lf_net=9.3, buyhold=0.4, lf_excess=8.9, lf_ir=0.65, turnover=20.5,
    ls_net=17.6, ls_excess=17.2, ls_ir=0.63,
    cost_bps=5.0, borrow_bps=100.0,
    # window sweep: (label, slope_t, n)
    win=[("2020-01 -> 2021-06", 0.12, 25), ("2021-07 -> 2022-12", 3.00, 77),
         ("2023-01 -> 2024-06", 2.04, 77), ("2024-06 -> 2025-12", 2.72, 77)],
    # synthetic control: (nowcast_beta, mean slope-t over 25 seeds)
    ctrl=[(0.0, -0.01), (0.15, 1.30), (0.35, 2.91), (0.60, 4.55)],
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

from opentable_reservations import data, strategy as st

# THERE IS NO REAL TAPE (see data.py): the free seated-diners history does not exist.
REAL = data.fetch_series(fetch=False)
HAVE_REAL = not REAL.empty
print("real seated-diners tape present:", HAVE_REAL, "(synthetic-only study by design)")

# The demonstrator world: a deterministic, seeded synthetic panel with the nowcast PLANTED.
PANEL, TRUTH = data.synthetic_series(nowcast_beta=0.35, seed=555)
print("demonstrator panel:", len(PANEL), "weeks,", PANEL.index[0].date(), "->", PANEL.index[-1].date(),
      "| fp", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Reservations Nowcast — do diners coming back predict restaurant stocks? 🍽️\n"
            "### Using an OpenTable-style 'seated-diners' trend to time a restaurant basket, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's an intuitive idea. During the pandemic, OpenTable published a public dashboard of "
            "**seated diners** — how many people were being seated at restaurants each day versus 2019. "
            "Everyone watched it to see how fast dining was recovering. The natural trade: if diners are "
            "flooding back, restaurant revenue should follow, so **restaurant stocks** (and, loosely, the "
            "consumer-discretionary ETF **XLY**) should rise. Read this week's reservations trend, position "
            "the restaurant basket for next week.\n\n"
            "It's a classic **alt-data nowcast** — use a real-world behaviour signal to get ahead of the "
            "numbers. So does it work?\n\n"
            "> 🍽️ **The honest catch, up front.** There is **no free, clean, historical seated-diners feed** "
            "you can download and test on. OpenTable's dashboard was a moving web widget, kept getting "
            "restated, and was eventually taken down — and it only covered the pandemic. So we can't check "
            "the claim on real data. Instead we build a **synthetic world** where we *know* the answer, and "
            "prove our measuring machine works. That's why the Signal stamp is capped at **Weak** (plausible + "
            "machinery-proven) and can never be **Real** here.\n\n"
            "> 📓 **This is the plain-language layer.** Want the regressions, the *t*-stats and the cost "
            "maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the code "
            "beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is 'reservations predict restaurant stocks' a sensible idea? | **Yes** — it's a real, "
            "literature-backed alt-data nowcast. Diners returning *should* precede revenue. |\n"
            "| Does our engine catch the effect when it's really there? | **Yes** — on a world where we "
            f"*plant* the nowcast, it fires cleanly (predictive *t* **+{R['slope_t']}**), and stays flat when "
            "there's nothing (control **−0.01**). The measuring machine is honest. |\n"
            "| Can we confirm it on real data? | **No** — there is no free, stable seated-diners history to "
            "test on. The dashboard was transient and discontinued. |\n"
            "| So could you trade it? | **No.** No real, license-able, reproducible signal → nothing to trade. "
            "The edge you'll see below lives only in the synthetic demo. |\n\n"
            "> The idea is good and our tools work — but without a real tape, 'reservations nowcast restaurant "
            "stocks' stays a **Weak** (unconfirmed) signal and a **Mirage** trade."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When restaurant reservations recover, restaurant stocks follow — so the reservations trend "
            "nowcasts the sector.\"*\n\n"
            "The intuition is clean. Reservations are a **real-time** read on dining demand; company revenue "
            "and stock prices react with a lag. If you can see the demand signal first, you're ahead of the "
            "market. This is the same logic behind using Google-search trends, satellite parking-lot counts or "
            "credit-card panels to nowcast revenue — reservations are a dining-flavoured version."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked and you could get the data, you'd have a genuine edge on restaurant names and XLY. "
            "But alt-data edges are famously fragile: they decay once everyone watches the same feed, and — "
            "crucially here — they need **clean, historical, point-in-time data** to even test. The whole study "
            "hinges on whether that data exists. (Spoiler from Beat 3: it doesn't, for free.)"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the signal.** Take the reservations year-over-year trend and strip out two things: its "
            "own slow drift (we care about *changes*, not the level) and whatever the overall market did that "
            "week (so we're not just re-discovering 'stocks went up'). What's left is the **reservations "
            "surprise**.\n"
            "2. **Predict next week.** Does a positive surprise *this* week go with a higher restaurant-basket "
            "return *next* week? (We always use this week's info to bet on next week — a one-week lag, no "
            "peeking.)\n"
            "3. **Trade it.** Go long the basket when the surprise is positive, step aside when it's negative, "
            "and see if that beats just holding — after trading costs.\n\n"
            "The problem: step 1 needs a real seated-diners history, and there isn't a free one. So we run the "
            "whole pipeline on a **synthetic world** where we planted a known nowcast, to prove the machine "
            "measures correctly."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**On a world where the nowcast is really there, does a positive reservations surprise this week "
            "predict a higher basket return next week?** (Synthetic demonstrator — the nowcast is planted.)"
        ),
        code(
            "af = st.aligned_frame(PANEL)      # (this-week surprise, next-week return), one-week lag\n"
            "reg = st.predictive_regression(PANEL)\n"
            "# bucket next-week returns by the sign of this week's surprise\n"
            "pos = af[af['signal'] > 0]['fwd_ret'] * 100\n"
            "neg = af[af['signal'] <= 0]['fwd_ret'] * 100\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['surprise > 0','surprise <= 0'], [pos.mean(), neg.mean()], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg NEXT-week basket return %')\n"
            "ax.set_title(f'High-reservations weeks are followed by higher returns (slope t {reg[\"slope_t\"]:+.2f})')\n"
            "fig.suptitle('synthetic demonstrator — nowcast PLANTED (no real tape exists)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'after positive surprise: {pos.mean():+.2f}%/wk | after negative: {neg.mean():+.2f}%/wk')\n"
            "print(f'predictive slope t (Newey-West) = {reg[\"slope_t\"]:+.2f}')"
        ),
        md(
            f"There's the nowcast, exactly as planted: weeks *after* a positive reservations surprise earn more "
            f"than weeks after a negative one, and the predictive *t* is **+{R['slope_t']}**. This is what a "
            "working engine looks like **when the effect is real** — which, on this synthetic world, it is by "
            "construction."
        ),
        md(
            "**Would a simple timing overlay have beaten buy-and-hold?** Go long when the surprise is positive, "
            "flat when negative — after 5 bps of trading cost per switch."
        ),
        code(
            "ov = st.overlay_backtest(PANEL, allow_short=False, cost_bps=5.0)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['buy & hold','overlay (net)'], [ov['buyhold_ann']*100, ov['overlay_net_ann']*100],\n"
            "       color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title(f'Timing overlay net +{ov[\"overlay_net_ann\"]*100:.1f}%/yr vs buy-hold +{ov[\"buyhold_ann\"]*100:.1f}%/yr')\n"
            "fig.suptitle('synthetic demonstrator only', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'overlay net {ov[\"overlay_net_ann\"]*100:+.1f}%/yr vs buy-hold {ov[\"buyhold_ann\"]*100:+.1f}%/yr '\n"
            "      f'(net excess {ov[\"net_excess_ann\"]*100:+.1f}%/yr, IR {ov[\"net_excess_ir\"]:.2f})')"
        ),
        md(
            f"On the planted world the overlay adds about **+{R['lf_excess']}%/yr** net of costs. Nice — but "
            "remember, **this is the synthetic demo**. There is no real seated-diners tape, so this edge is a "
            "picture of *what the machine can detect*, not a tradable result."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The idea is plausible and the engine catches a real nowcast cleanly (*t* "
            f"+{R['slope_t']} on the planted world, flat at the null). But with **no real tape** it can't clear "
            "the bar for a confirmed (**Real**) signal — capped at **Weak**.\n"
            "- **Tradability — Mirage.** No free, license-able, reproducible seated-diners feed → nothing to "
            "actually trade. The edge above lives only in the synthetic world."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and the reason is data, not maths. OpenTable's seated-diners dashboard was a **moving web "
            "widget** that got restated and then taken down, and it only ever covered the pandemic recovery — "
            "a single, weird, non-repeating episode. You can't download a clean multi-year history, so you "
            "can't backtest it honestly, let alone run it live. A signal you can't reproduce is a signal you "
            "can't trade."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Same pattern, other collectibles/alt-data.** The desk's [273 Lego-Returns](../../273-lego-returns/), "
            "[275 Whisky-Cask](../../275-whisky-cask/) and [276 Sneaker-Resale](../../276-sneaker-resale/) are "
            "the same 'no free real data → prove the machine on a synthetic world' move.\n"
            "- **If you have the data.** Got a licensed, point-in-time reservations or foot-traffic panel? "
            "Fork this, feed it a real tape, and see whether the predictive *t* clears 2 out of sample — the "
            "only way this becomes a **Real** signal.\n\n"
            "*The engine is ready; it just needs a real tape that, for free, doesn't exist.*"
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
            "# The Reservations Nowcast — a quantitative teardown 🔬\n"
            "### surprise construction · predictive regression w/ market control · Newey-West *t* · label-shuffle placebo · timing overlay + costs/borrow · four-window sweep · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* We build an OpenTable-style reservations "
            "**surprise**, regress next-week restaurant-basket returns on it (controlling for the market), and "
            "test the nowcast — on a **synthetic demonstrator**, because no free real seated-diners tape "
            "exists.\n\n"
            "> ⚠️ **Not investment advice.** **Synthetic-only study**: `data.fetch_series` returns empty by "
            "design, `HAVE_REAL` is always False. All numbers below are the machinery proof on a deterministic, "
            f"seeded world (nowcast planted at `nowcast_beta = {R['beta']}`, seed 555, panel fp `{R['fp_panel']}`). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Predictive slope-*t* (Newey-West) **+{R['slope_t']}**, placebo *p* "
            f"**{R['placebo_p']}** on the *planted* world; control flat at the null (mean slope-*t* "
            f"**{R['ctrl'][0][1]}**). Plausible + machinery-proven — but **no real tape** → cannot be `REAL`. |\n"
            f"| **Tradability** | `MIRAGE` | Overlay net **+{R['lf_net']}%/yr** vs buy-hold **+{R['buyhold']}%/yr** "
            "(synthetic demo only). No license-able, reproducible seated-diners feed exists. |\n\n"
            "> 💡 In plain words: the detector is faithful (it fires on a planted nowcast and stays silent at the "
            "null), but there is nothing real to point it at — so the claim is unconfirmed, not investable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t$ be the reservations **surprise** at week $t$ (YoY, de-trended, market-orthogonalised) "
            "and $r_{t+1}$ the restaurant-basket return over week $t+1$.\n\n"
            "- **H₁ (nowcast).** In $r_{t+1} = a + b_1 s_t + b_2 m_{t+1} + \\varepsilon$, the nowcast loading "
            "$b_1 > 0$ at Newey-West *t* ≥ 2 (the surprise adds info *beyond* the market $m$).\n"
            "- **H₂ (not luck).** A label-shuffle placebo puts the observed *t* in the tail.\n"
            "- **H₃ (robustness).** $b_1$ keeps its sign across sub-periods.\n"
            "- **H₄ (tradable).** A sign-of-surprise overlay beats buy-and-hold **net** of costs.\n\n"
            "On the **planted** world we find H₁–H₄ all satisfied — that is the point of a positive control. "
            "The binding constraint is external: **there is no real tape** (H₀: the data exists), so none of "
            "this is a real-tape result and the Signal axis is capped at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Alt-data nowcasting is a genuine research program (Bok et al. 2018; Da, Engelberg & Gao 2011 on "
            "search-attention). Two things decide whether *this* instance is real: (a) whether the surprise "
            "carries information **beyond the market** — which is why we orthogonalise before testing; and (b) "
            "whether a **clean historical tape** exists to test on. Here (a) is demonstrable on synthetic data "
            "but (b) fails: OpenTable's seated-diners feed was transient, restated and discontinued. Same "
            "synthetic-only footing as [273 Lego](../../273-lego-returns/) / [275 Whisky](../../275-whisky-cask/) "
            "/ [276 Sneaker](../../276-sneaker-resale/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Surprise.** $s_t = z\\big(\\text{resv YoY} - \\text{trend} \\perp m_t\\big)$ — YoY minus a "
            "~quarter moving-average trend, residualised on the contemporaneous market, then z-scored. Built in "
            "`data.py` so it never peeks at the latent effect.\n"
            "- **Predictive regression.** OLS $r_{t+1} = a + b_1 s_t + b_2 m_{t+1} + \\varepsilon$ with "
            "**Newey-West (4-lag) HAC** *t* on $b_1$ (overlapping weekly forwards ⇒ plain *t* too optimistic).\n"
            "- **Placebo.** Shuffle $s$ vs $r_{t+1}$ 2000× and read the slope-*t* tail probability.\n"
            "- **Overlay + frictions.** Long/flat (or long/short) on $\\text{sign}(s_t)$; one-way cost on every "
            "position change; short weeks pay a pro-rated borrow. Gross AND net.\n"
            "- **Robustness.** Four contiguous sub-periods; re-estimate the slope-*t*.\n"
            "- **Positive control.** A deterministic synthetic world with a planted nowcast and a null, averaged "
            "over 25 seeds (house rule ≥ 20).\n\n"
            "Timing: $s_t$ uses only week-$t$ information; the position it implies earns week $t+1$. That "
            "one-week execution lag is the single documented convention."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁ & H₂\n\n"
            "Regress next-week basket return on this week's surprise, **controlling for next-week market**, "
            "with a Newey-West *t* and a label-shuffle placebo."
        ),
        code(
            "reg = st.predictive_regression(PANEL)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=555)\n"
            "af = st.aligned_frame(PANEL)\n"
            "# scatter: this-week surprise vs next-week return, with the fitted (market-partialled) slope\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(af['signal'], af['fwd_ret']*100, s=16, color=GREY, alpha=.6)\n"
            "xs = np.linspace(af['signal'].min(), af['signal'].max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs + af['fwd_ret'].mean())*100, c=GREEN, lw=2)\n"
            "ax.set_xlabel('reservations surprise this week (z)'); ax.set_ylabel('NEXT-week basket return %')\n"
            "ax.set_title(f'Nowcast slope t = {reg[\"slope_t\"]:+.2f}  (placebo p {p:.4f}, R2 {reg[\"r2\"]:.2f})')\n"
            "fig.suptitle('synthetic demonstrator — nowcast planted', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'b1 slope {reg[\"slope\"]:+.4f} | Newey-West t {reg[\"slope_t\"]:+.2f} | placebo p {p:.4f} '\n"
            "      f'| R2 {reg[\"r2\"]:.2f} | n {reg[\"n\"]}')"
        ),
        md(
            f"> 💡 In plain words: on the planted world H₁ & H₂ **hold** — the surprise predicts next week's "
            f"return at *t* **+{R['slope_t']}** with placebo *p* **{R['placebo_p']}**, and it survives the market "
            "control (so it's not just beta). This is the detector working, not a real-tape claim."
        ),
        md(
            "### 4b · The timing overlay — H₄ (net of costs & borrow)\n\n"
            "Long/flat and long/short overlays on the sign of the surprise, both net of a one-way cost and (for "
            "the short leg) a borrow."
        ),
        code(
            "lf = st.overlay_backtest(PANEL, allow_short=False, cost_bps=5.0)\n"
            "ls = st.overlay_backtest(PANEL, allow_short=True, cost_bps=5.0, borrow_ann_bps=100.0)\n"
            "labels = ['buy & hold','long/flat\\n(net)','long/short\\n(net)']\n"
            "vals = [lf['buyhold_ann']*100, lf['overlay_net_ann']*100, ls['overlay_net_ann']*100]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(labels, vals, color=[GREY, GREEN, AMBER], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title('Overlay beats buy-hold NET on the planted world (synthetic demo)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long/flat : gross {lf[\"overlay_gross_ann\"]*100:+.1f}% -> net {lf[\"overlay_net_ann\"]*100:+.1f}%/yr '\n"
            "      f'(excess {lf[\"net_excess_ann\"]*100:+.1f}%, IR {lf[\"net_excess_ir\"]:.2f}, turn {lf[\"turnover_per_yr\"]:.1f}/yr)')\n"
            "print(f'long/short: gross {ls[\"overlay_gross_ann\"]*100:+.1f}% -> net {ls[\"overlay_net_ann\"]*100:+.1f}%/yr '\n"
            "      f'(excess {ls[\"net_excess_ann\"]*100:+.1f}%, IR {ls[\"net_excess_ir\"]:.2f})')\n"
            "print(f'buy & hold: {lf[\"buyhold_ann\"]*100:+.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: H₄ **holds on the planted world** — long/flat nets **+{R['lf_net']}%/yr** "
            f"(excess +{R['lf_excess']}%, IR {R['lf_ir']}) vs buy-hold **+{R['buyhold']}%/yr**, and costs barely "
            "dent it. But this is the synthetic demo; **there is no real tape to run it on**, which is exactly "
            "why Tradability is `MIRAGE`."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold across sub-periods?)\n\n"
            "Four contiguous blocks; re-estimate the nowcast slope-*t* in each."
        ),
        code(
            "rows = st.window_sweep(PANEL, n_windows=4)\n"
            "tab = pd.DataFrame(rows)[['window','slope_t','n']].set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if t>0 else RED for t in tab['slope_t']]\n"
            "ax.bar(range(len(tab)), tab['slope_t'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels([w.split(' -> ')[0] for w in tab.index], rotation=15)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('nowcast slope-t'); ax.legend()\n"
            "ax.set_title('Sign stable & positive in the populated blocks (first block sparse: year-1 YoY drop)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **holds** where there's data — slope-*t* **+3.0 / +2.0 / +2.7** in the "
            "three full blocks (the first block is only ~25 weeks because the first year is lost to the YoY "
            "definition, so its **+0.1** is uninformative, not a sign flip)."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — on the planted world H₁ (slope-*t* +{R['slope_t']}), H₂ (placebo "
            f"{R['placebo_p']}) and H₃ (sign stable) all hold, and the control is flat at the null "
            f"({R['ctrl'][0][1]}). But **H₀ fails**: no free real tape ⇒ cannot be `REAL`, capped at `WEAK`.\n"
            f"- **Tradability `MIRAGE`** — overlay nets +{R['lf_net']}%/yr on the demo, but there is no "
            "license-able, reproducible seated-diners feed to trade; the edge exists only in the synthetic world."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the data wall\n\n"
            "The blocker is data, not economics. OpenTable's seated-diners dashboard was a transient HTML "
            "widget, repeatedly restated and then discontinued, covering only the pandemic recovery. There is "
            "no cache-first retail fetch that reproduces a clean multi-cycle history — so `fetch_series` returns "
            "empty by design and every number here is synthetic. No reproducible tape ⇒ no honest backtest ⇒ no "
            "trade."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it print a big *t* on anything? Plant a nowcast of "
            "increasing strength and watch the mean slope-*t* climb from ~0 — averaged over 25 seeds so no "
            "single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.10, 0.20, 0.35, 0.50, 0.60]\n"
            "ts = [st.synthetic_mean_t(data, nowcast_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted nowcast_beta (stronger →)')\n"
            "ax.set_ylabel('mean predictive slope-t (25 seeds)')\n"
            "ax.set_title('Faithful detector: flat at the null, past +2 as the planted nowcast grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'nowcast_beta {b:+.2f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "The mean slope-*t* is ≈ 0 at the null and rises monotonically past +2 as the planted nowcast grows "
            "— so the engine neither misses a real effect nor invents one. The real-tape verdict is therefore "
            "**unavailable, not negative**: point this engine at a licensed reservations/foot-traffic panel and "
            "it will give an honest read. Until then, `WEAK` × `MIRAGE`. Same synthetic-only footing as "
            "[273 Lego-Returns](../../273-lego-returns/), [275 Whisky-Cask](../../275-whisky-cask/) and "
            "[276 Sneaker-Resale](../../276-sneaker-resale/)."
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
