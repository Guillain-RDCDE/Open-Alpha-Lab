"""Generate the two narrative notebooks for Study 559 (Dark-Pool-Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a
**synthetic-only** study — there is no free real dark-pool-ratio tape (FINRA ATS is weekly &
lagged, off-ATS internalised flow is excluded, clean feeds are paywalled). So every cell runs on
the deterministic synthetic world in ``dark_pool_ratio/data.py`` (seeded on 559), fully offline and
reproducible; the frozen headline numbers in ``R`` mirror docs/results.md.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic, seed 559,
# n=300). Null-world base-seed cross-section + 25-seed seed-robust stats + the positive control.
R = dict(
    fp_null="11f28bf40b8c", fp_plant="fbdeeb6093c5",
    n=300, seed=559,
    # primary null-world cross-section (base seed 559):
    ic=0.026, ic_t=0.44, spread=1.28, ls_t=0.75, placebo_p=0.65,
    slope_t_raw=0.47, slope_t_ctrl=-0.46, net=0.58,
    cost_bps=5.0, borrow_bps=50.0,
    # seed-robust (25 seeds): the confound story
    null_raw_ic_t=0.77, null_ctrl_slope_t=-0.19,
    plant_raw_ic_t=2.32, plant_ctrl_slope_t=1.39,
    # sweep: (alpha, mean_ic, mean_ic_t) over 25 seeds
    sweep=[(0.0, 0.044, 0.77), (0.03, 0.089, 1.54), (0.06, 0.134, 2.32),
           (0.09, 0.178, 3.11), (0.12, 0.221, 3.88)],
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

from dark_pool_ratio import data, strategy as st

SEED = 559
# The study is SYNTHETIC-ONLY: no free real dark-pool-ratio tape exists (FINRA ATS is weekly &
# lagged, off-ATS internalised flow is excluded, clean feeds are paywalled). Everything below runs
# on the deterministic synthetic world, offline and reproducible.
REAL = data.fetch_panel(fetch=False)     # empty by construction — documented in data.py
HAVE_REAL = not REAL.empty
NULL_PANEL, _ = data.synthetic_null_panel(seed=SEED)   # the primary cross-section (dpr_alpha = 0)
print("real DPR tape available:", HAVE_REAL, "(none exists on a free stack — synthetic-only study)")
print("null-world synthetic panel:", len(NULL_PANEL), "names, seed", SEED)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Dark-Pool Ratio — is 'more volume in the dark' smart money? 🌑\n"
            "### When a stock trades more off-exchange, is that institutions quietly loading up?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Informed_accumulation_in_the_dark%3F: Unproven](https://img.shields.io/badge/Informed_accumulation_in_the_dark%3F-Unproven-8b949e?style=flat-square)\n\n"
            "Not every share trade happens on a stock exchange you can see. A big chunk prints "
            "**off-exchange** — in *dark pools*, private venues where institutions swap large blocks "
            "without showing their hand, and in *internalisers* (the wholesalers that fill most retail "
            "orders). The **dark-pool ratio** is simply: what fraction of a stock's volume printed in "
            "the dark rather than on the lit tape?\n\n"
            "A popular trading story says a *rising* dark-pool ratio is **smart money accumulating** — "
            "patient funds building a position quietly — so the stock should drift up next. We put "
            "that to the test.\n\n"
            "> ⚠️ **A catch you should know up front.** There is **no free daily dark-pool-ratio feed** "
            "a retail researcher can reach: the official FINRA numbers are *weekly*, come out *weeks "
            "late*, and miss the wholesaler flow entirely. So we can't run this on the real tape — we "
            "run it on a *synthetic world* we built, to test the **method** honestly, and we say so.\n"
            ">\n"
            "> 📓 **This is the plain-language layer.** Want the IC, the placebo test and the cost "
            "maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Not investment advice."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can we even test this on real data? | **No.** The free official data (FINRA) is weekly, "
            "lagged by weeks, and misses the wholesaler flow. A clean daily panel doesn't exist for "
            "free — so the strongest possible verdict is capped below `REAL`. |\n"
            f"| In our synthetic *null* world, does the ratio predict returns? | **No** — the rank "
            f"correlation is ≈ 0 (IC-*t* **+{R['ic_t']}**), and a shuffle test says *p* = {R['placebo_p']}. Flat. |\n"
            f"| But doesn't a naive sort show *something*? | **Yes — a mirage.** Averaged over 25 "
            f"worlds the raw signal looks mildly positive (IC-*t* +{R['null_raw_ic_t']}) — but that's "
            "purely because dark-pool trading rises with **size**, and big stocks drift. Control for "
            f"size and it vanishes (**{R['null_ctrl_slope_t']}**). |\n"
            f"| Does the engine work at all? | **Yes.** Plant a *real* effect and it's caught cleanly "
            f"(IC-*t* +{R['plant_raw_ic_t']}). The problem is the data, not the detector. |\n\n"
            "> The 'dark = smart money' story is untestable on a free stack, and the version you *can* "
            "probe turns out to be a **size confound** in disguise."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When more of a stock's volume prints off-exchange, informed institutions are "
            "accumulating — and the stock follows.\"* — the dark-pool-ratio folklore.\n\n"
            "The intuition: a fund that wants to buy a million shares can't just slam the lit "
            "exchange — it would move the price against itself. So it works the order quietly in dark "
            "pools. If that's what's happening, a high (and rising) dark-pool ratio is a footprint of "
            "**patient, informed buying**, and the price should catch up later.\n\n"
            "It's a seductive story. But microstructure theory is *not* on its side: a well-known "
            "result (Zhu 2014) argues dark pools skim the *less* informed orders, and the effect of "
            "dark trading on information is **non-monotone** (Comerton-Forde & Putniņš 2015) — so the "
            "sign isn't even clear in theory."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the folklore were true, it would be an easy edge: watch each stock's off-exchange "
            "share, buy the ones going dark, and ride the institutions. The desk has looked at other "
            "'flow' ideas — [376 MOC-Imbalance](../../376-moc-imbalance/) (the closing-auction "
            "imbalance), [418 Money-Flow-Index](../../418-money-flow-index/) and "
            "[419 Chaikin-Money-Flow](../../419-chaikin-money-flow/) (price×volume oscillators). This "
            "one is different: it's about **where** the volume prints — lit vs dark — not its price "
            "impact."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the ratio.** For each stock: what share of its volume printed off-exchange?\n"
            "2. **Sort.** Line the stocks up from most-lit to darkest; take the darkest fifth and the "
            "most-lit fifth.\n"
            "3. **Compare forward returns.** Did the *dark* fifth beat the *lit* fifth over the next "
            "period? The folklore says yes.\n\n"
            "**The catch that decides this study:** we can't get a clean daily dark-pool ratio for "
            "free. So we build a *synthetic* world where we control the truth — sometimes the ratio "
            "genuinely predicts returns, sometimes it's pure noise — and check whether our test tells "
            "the two apart. That's an honest test of the *method*, with the tape gap stated openly.\n\n"
            "*Timing:* the ratio is measured over the formation window (fully known at that close); we "
            "enter there and hold forward. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**In the null world (where the ratio carries NO real information), does the dark fifth "
            "still beat the lit fifth?**"
        ),
        code(
            "b = st.bucket_returns(NULL_PANEL, frac=0.2); ls = st.long_short_tstat(b)\n"
            "dark_m, lit_m, spr, t = b['dark_mean']*100, b['lit_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['DARKEST fifth','MOST-LIT fifth'], [dark_m, lit_m], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Null world: dark {dark_m:+.1f}% vs lit {lit_m:+.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')\n"
            "fig.suptitle('SYNTHETIC null world (no real edge planted)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'dark {dark_m:+.1f}% | lit {lit_m:+.1f}% | spread {spr:+.1f}% | t {t:+.2f}')"
        ),
        md(
            f"The gap is tiny (**{R['spread']}%**, *t* {R['ls_t']}) and not significant — exactly what "
            "you want to see when there's genuinely nothing there. Good: the test isn't inventing a "
            "signal from noise."
        ),
        md(
            "**But here's the trap.** In the real world the dark-pool ratio is *tangled up with size* "
            "— big, liquid stocks get internalised more. And big stocks drift. So a naive sort can "
            "show a fake 'signal' that is really just size. Watch what happens when we average over 25 "
            "worlds and then *control for size*:"
        ),
        code(
            "raw, ctrl = " + f"{R['null_raw_ic_t']}, {R['null_ctrl_slope_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['naive DPR sort','after controlling\\nfor size'], [raw, ctrl], color=[AMBER, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('mean IC t-stat (25 worlds, null)')\n"
            "ax.set_title('The apparent signal is a SIZE confound — it vanishes when you net size out')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('null raw IC-t +{R['null_raw_ic_t']} -> controlled slope-t {R['null_ctrl_slope_t']}')"
        ),
        md(
            f"There it is. The naive sort shows a mild positive (+{R['null_raw_ic_t']}) — tempting! — "
            f"but it's **entirely the size confound**: net size out and it collapses to "
            f"**{R['null_ctrl_slope_t']}**. On real data you'd have to strip size and liquidity out "
            "*first*, before believing any dark-pool signal."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No free real tape to earn `REAL`; on the synthetic null the ratio is "
            "flat, and the only apparent lift is a size confound.\n"
            "- **Tradability — Mirage.** A book whose edge is a confound, on a panel you can't build "
            "from free data. Nothing to harvest.\n"
            "- **Informed accumulation in the dark? — Unproven.** The engine would catch a real "
            "effect (see beat 7), but theory says the sign is ambiguous and no tape lets us check."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. First, you can't even *get* the daily numbers for free. Second, even in the "
            "synthetic world the apparent edge is size, not venue — and the tiny gross spread "
            f"(**{R['spread']}%**) is roughly halved by costs (net **{R['net']}%** after a per-leg fee "
            "and a short borrow). There is no trade here, only a story."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The engine is faithful.** Plant a *real* dark-pool effect and the test catches it "
            f"cleanly (IC-*t* +{R['plant_raw_ic_t']} over 25 worlds — see the quant notebook). So the "
            "`None` verdict is about the *data and the confound*, not a broken detector.\n"
            "- **The real test needs a real tape.** If you can buy a clean daily dark-pool-ratio feed, "
            "re-run this — but *control for size and liquidity first*, and remember theory says the "
            "sign may be the other way.\n\n"
            "*Think dark-pool prints really do lead? Fork this, plug in a paid daily DPR feed, net out "
            "size/liquidity, and show a residual IC clearing t = 2.*"
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
            "# The Dark-Pool Ratio — a quantitative teardown 🔬\n"
            "### Spearman IC (Fisher-z *t*) · quintile long-short · label-shuffle placebo · size-confound control · costs & borrow · monotone effect sweep · seed-robust positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Informed_accumulation_in_the_dark%3F: Unproven](https://img.shields.io/badge/Informed_accumulation_in_the_dark%3F-Unproven-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a stock's "
            "**dark-pool ratio** (off-exchange volume share) predicts forward returns.\n\n"
            "> ⚠️ **Synthetic-only, and here's why.** No free per-name daily dark-pool-ratio tape "
            "exists: FINRA ATS volume is weekly & lagged, off-ATS internalised flow is excluded, clean "
            "feeds are paywalled (see [`data.py`](../dark_pool_ratio/data.py)). So there is nothing to "
            "certify `REAL`; this is a **machinery proof** on a deterministic world (seed 559), and "
            "the tape gap is stated on the Signal axis. Numbers mirror "
            "[`docs/results.md`](../docs/results.md); methods in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | No free real tape (⇒ capped below `REAL`); synthetic null IC "
            f"**+{R['ic']}** (IC-*t* **+{R['ic_t']}**, placebo *p* {R['placebo_p']}); the only lift "
            f"(raw null IC-*t* +{R['null_raw_ic_t']}, 25 seeds) is a **size confound** → "
            f"**{R['null_ctrl_slope_t']}** controlled. |\n"
            f"| **Tradability** | `MIRAGE` | Edge is a confound, panel unbuildable from free data. "
            f"Null gross {R['spread']}% → net {R['net']}% ({R['cost_bps']:.0f} bps/leg + "
            f"{R['borrow_bps']:.0f} bps borrow). |\n"
            f"| **Informed accumulation in the dark?** | `UNPROVEN` | Engine banks a planted effect at "
            f"IC-*t* +{R['plant_raw_ic_t']} (still +{R['plant_ctrl_slope_t']} netting size) — faithful "
            "detector, no tape & ambiguous theory. |\n\n"
            "> 💡 In plain words: the detector works (the positive control proves it), but there is no "
            "real DPR tape to run it on, and the naive synthetic signal is size in disguise."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_i$ be firm $i$'s dark-pool ratio and $r_i$ its forward return.\n\n"
            "- **H₁ (IC).** $\\text{IC} = \\text{Spearman}(D, r) > 0$ at *t* ≥ 2 (darker names rank "
            "higher in forward return).\n"
            "- **H₂ (long-short).** $\\mathbb{E}[r \\mid \\text{dark}] - \\mathbb{E}[r \\mid \\text{lit}] > 0$.\n"
            "- **H₃ (not a confound).** The IC survives controlling for size/liquidity.\n"
            "- **H₄ (net).** H₁ survives costs and the short-leg borrow.\n\n"
            "The decisive obstacle to **H₁** is data: no free daily DPR panel exists, so a robust *t* "
            "≥ 2 on a *real* tape — the only thing that earns `REAL` — cannot be computed. On the "
            "synthetic null, **H₁ is not rejected in the folklore's favour** (IC ≈ 0), **H₃ fails for "
            "the naive sort** (the apparent lift is size), and **H₄ is moot** (no edge to net)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Two things sink the folklore before any tape: (a) **theory** "
            "— Zhu (2014) argues dark venues attract the *less* informed orders and Comerton-Forde & "
            "Putniņš (2015) find the effect of dark share on price discovery is *non-monotone*, so the "
            "sign isn't even predicted to be positive; and (b) **confounding** — dark share rises with "
            "size and liquidity (Buti-Rindi-Werner), both independent return drivers. This study "
            "reuses the IC + placebo + positive-control machinery of "
            "[540 Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **IC.** Spearman rank correlation of $D$ vs $r$; significance via Fisher-z, "
            "$t = \\tanh^{-1}(\\text{IC})\\sqrt{n-3}$.\n"
            "- **Sort.** Quintile tails: long the darkest fifth, short the most-lit fifth; two-sample "
            "(Welch) *t* on the spread.\n"
            "- **Placebo.** Label-shuffle null (2000 perms) on the IC.\n"
            "- **Confound control.** OLS of $r$ on $D$ *with a size regressor* — the marginal DPR "
            "slope beyond size.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the lit short.\n"
            "- **Effect sweep + positive control.** Dial the planted `dpr_alpha` from null up and "
            "watch the IC-*t*, averaged over 25 seeds (house rule ≥ 20).\n\n"
            "Timing: $D$ is measured over the formation window (public at that close); enter at that "
            "close, hold forward — one execution lag, no look-ahead (baked into the generator). The "
            "study is synthetic-only because the real daily DPR panel is not reachable for free."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The IC and the sort — H₁ / H₂ (null world)\n\n"
            "The Spearman IC with a Fisher-z *t*, the quintile spread with a Welch *t*, and the "
            "label-shuffle placebo — on the null world (no effect planted)."
        ),
        code(
            "ic = st.information_coefficient(NULL_PANEL)\n"
            "b = st.bucket_returns(NULL_PANEL, 0.2); ls = st.long_short_tstat(b)\n"
            "p = st.placebo_pvalue(NULL_PANEL, n_perm=2000, seed=SEED)\n"
            "icv, ict = ic['ic'], ic['ic_t']\n"
            "dark_m, lit_m, spr, t = b['dark_mean']*100, b['lit_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['DARK fifth','LIT fifth'], [dark_m, lit_m], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Null world: IC {icv:+.3f} (t {ict:+.2f}, placebo p {p:.2f}) | spread {spr:+.1f}% (t {t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'IC {icv:+.3f} | IC-t {ict:+.2f} | spread {spr:+.1f}% | LS-t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: on the null, **H₁/H₂ are flat** — IC **+{R['ic']}** (*t* "
            f"+{R['ic_t']}), spread **{R['spread']}%** (*t* {R['ls_t']}), placebo *p* = {R['placebo_p']}. "
            "The engine correctly reports *nothing* when nothing is planted."
        ),
        md(
            "### 4b · The size confound — H₃ (the crux)\n\n"
            "Dark share is correlated with size; size drives returns. Average the *raw* IC-*t* over 25 "
            "null worlds, then add a size control and re-measure the DPR slope-*t*."
        ),
        code(
            "def mean_stat(alpha, ctrl, n_seeds=25, base=SEED):\n"
            "    v=[]\n"
            "    for s in range(base, base+n_seeds):\n"
            "        pn,_=data.synthetic_panel(dpr_alpha=alpha, seed=s)\n"
            "        v.append(st.cross_section_regression(pn, control_size=True)['slope_t'] if ctrl\n"
            "                 else st.information_coefficient(pn)['ic_t'])\n"
            "    return float(np.nanmean(v))\n"
            "raw = mean_stat(0.0, False); ctrl = mean_stat(0.0, True)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['naive DPR IC-t','DPR slope-t\\n| size'], [raw, ctrl], color=[AMBER, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2')\n"
            "ax.set_ylabel('mean t-stat (25 null seeds)'); ax.legend()\n"
            "ax.set_title(f'Null: raw {raw:+.2f} is a SIZE confound -> {ctrl:+.2f} once netted out')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null raw IC-t {raw:+.2f} -> controlled slope-t {ctrl:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **H₃ fails for the naive sort.** The raw null IC-*t* averages "
            f"**+{R['null_raw_ic_t']}** — a *false positive* — but it is entirely the size confound: "
            f"net size out and the DPR slope-*t* is **{R['null_ctrl_slope_t']}**. Any real DPR study "
            "must control for size/liquidity before believing the IC."
        ),
        md(
            "### 4c · The effect sweep — the detector responds to truth, not seeds\n\n"
            "Dial the planted `dpr_alpha` from null up and watch the mean IC-*t* (25 seeds)."
        ),
        code(
            "alphas = [w[0] for w in " + repr(R["sweep"]) + "]\n"
            "sweep = st.effect_sweep(data, alphas, n_seeds=25, base_seed=SEED)\n"
            "xs = [s[0] for s in sweep]; ts = [s[2] for s in sweep]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(xs, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted dpr_alpha (0 = null, higher = stronger real effect)')\n"
            "ax.set_ylabel('mean IC t-stat (25 seeds)')\n"
            "ax.set_title('The IC-t rises monotonically with the planted truth — flat at null, past 2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, icv, t in sweep: print(f'alpha {a:+.2f} -> mean IC {icv:+.3f}, mean IC-t {t:+.2f}')"
        ),
        md(
            "> 💡 In plain words: at the null the IC-*t* is ≈ 0 (the residual +0.77 is the confound); "
            "as a *real* effect is planted the IC-*t* climbs monotonically past 2. The detector "
            "tracks truth."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no free real tape (⇒ below `REAL`); synthetic null IC ≈ 0 (*t* "
            f"+{R['ic_t']}); naive lift is a size confound (+{R['null_raw_ic_t']} → {R['null_ctrl_slope_t']}).\n"
            f"- **Tradability `MIRAGE`** — edge is a confound, panel unbuildable from free data; gross "
            f"{R['spread']}% → net {R['net']}%.\n"
            "- **Informed accumulation in the dark? `UNPROVEN`** — faithful engine, no tape, ambiguous "
            "theory."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "There is no edge to charge costs against; the frictions just deepen the noise-level "
            "spread."
        ),
        code(
            "b = st.bucket_returns(NULL_PANEL, 0.2)\n"
            "gross = b['spread']*100\n"
            "net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=50.0, holding_years=1.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('dark - lit spread % (null world)')\n"
            "ax.set_title(f'Noise-level gross ({gross:+.1f}%) roughly halved by costs ({net:+.1f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}% -> net {net:+.2f}% (5 bps/leg + 50 bps borrow, 1y hold)')"
        ),
        md(
            "> 💡 In plain words: at the null the long-dark/short-lit book is ~flat before costs and "
            f"positive-but-tiny after — no signal to monetise. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real DPR effect "
            "(`dpr_alpha > 0`) and confirm the IC-*t* clears 2 — and that the *marginal* effect "
            "survives the size control too. Averaged over 25 seeds so no lucky seed can fake it."
        ),
        code(
            "def mean_stat(alpha, ctrl, n_seeds=25, base=SEED):\n"
            "    v=[]\n"
            "    for s in range(base, base+n_seeds):\n"
            "        pn,_=data.synthetic_panel(dpr_alpha=alpha, seed=s)\n"
            "        v.append(st.cross_section_regression(pn, control_size=True)['slope_t'] if ctrl\n"
            "                 else st.information_coefficient(pn)['ic_t'])\n"
            "    return float(np.nanmean(v))\n"
            "rows = [('null (0.00)', mean_stat(0.0,False), mean_stat(0.0,True)),\n"
            "        ('planted (0.06)', mean_stat(0.06,False), mean_stat(0.06,True))]\n"
            "tab = pd.DataFrame(rows, columns=['world','raw IC-t','slope-t | size']).set_index('world')\n"
            "x = np.arange(len(tab)); w=0.38\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(x-w/2, tab['raw IC-t'], w, color=AMBER, label='raw IC-t')\n"
            "ax.bar(x+w/2, tab['slope-t | size'], w, color=GREEN, label='slope-t | size')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tab.index); ax.set_ylabel('mean t (25 seeds)'); ax.legend()\n"
            "ax.set_title('Flat at null (raw lift = confound); clears 2 when a real effect is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            f"The IC-*t* is ≈ 0 at the null (the +{R['null_raw_ic_t']} raw lift is the size confound, "
            f"which the size control removes to {R['null_ctrl_slope_t']}) and clears 2 "
            f"(+{R['plant_raw_ic_t']}, still +{R['plant_ctrl_slope_t']} controlled) when a real DPR "
            "effect is planted — so the engine is a faithful detector. The `NONE` verdict is therefore "
            "a statement about **data availability and confounding**, not a broken pipeline. For the "
            "close-auction cousin see [376 MOC-Imbalance](../../376-moc-imbalance/); for lit "
            "price×volume flow see [418](../../418-money-flow-index/) / [419](../../419-chaikin-money-flow/)."
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
