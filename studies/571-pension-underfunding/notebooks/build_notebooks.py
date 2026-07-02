"""Generate the two narrative notebooks for Study 571 (Pension-Underfunding).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). **This study is
synthetic-only**: the point-in-time pension-footnote data the real anomaly needs is not free, so
``HAVE_REAL`` is always ``False`` and there is NO real-tape banner anywhere. Every figure is drawn
from the deterministic, seeded synthetic world in ``pension_underfunding/data.py`` — the numbers in
the dict ``R`` below mirror those the code recomputes live (they match ``docs/results.md``), so the
notebook re-runs identically for any reader, offline.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic
# panel underfunding_alpha=-0.05, idio_vol=0.18, seed 571, 300 firms; panel fp b43234ffbd01).
# These are the DETERMINISTIC synthetic run's numbers — recomputed live in the cells below.
R = dict(
    fp_panel="b43234ffbd01", n=300, k=60, alpha=-0.05,
    well_mean=7.15, under_mean=-1.64, spread=8.79, ls_t=2.45, placebo_p=0.006,
    slope=-4.69, slope_t=-4.64, ic=-0.26,
    net=7.09, cost_bps=5.0, borrow_bps=150.0,
    # robustness designs: (label, spread%, ls_t)
    designs=[("quintile 20% eq", 8.79, 2.45), ("decile 10% eq", 14.86, 3.14),
             ("tercile 33% eq", 9.14, 3.49), ("quintile 20% cap", 5.93, 2.45)],
    # synthetic control: (alpha, mean slope-t, mean IC over 25 seeds)
    ctrl=[(0.0, 0.01, 0.001), (-0.02, -1.91, -0.109), (-0.05, -4.79, -0.266),
          (-0.08, -7.67, -0.404)],
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

from pension_underfunding import data, strategy as st

# The real point-in-time pension-footnote panel is NOT free (see data.py) -> synthetic-only.
REAL = data.real_panel(fetch=False)
HAVE_REAL = not REAL.empty
print("real pension panel present:", HAVE_REAL, "-> synthetic-only study (data caveat on SIGNAL axis)")

# The DETERMINISTIC synthetic world is the tape we test on (seed 571, planted alpha = -0.05).
PANEL, TRUTH = data.synthetic_panel()
print(f"synthetic panel: {len(PANEL)} firms, planted underfunding_alpha={TRUTH.underfunding_alpha}, "
      f"fp={data.fingerprint(PANEL)}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Pension Hole — do firms with big underfunded pensions earn *less*? 🕳️\n"
            "### Off-balance-sheet leverage the market might be ignoring, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Some companies promise their retirees a pension — and set aside a pot of money to pay "
            "for it. When that pot is smaller than the promises (the *projected benefit "
            "obligation*), the plan is **underfunded**: there's a hole. That hole is a real debt — "
            "the company will have to pour cash in to fill it — but it sits in a footnote, not on "
            "the main balance sheet. Easy to overlook.\n\n"
            "A famous 2006 study (Franzoni & Marin) found that the firms with the **biggest holes** "
            "went on to earn the **lowest** stock returns — about 7%/yr worse — as if the market "
            "keeps forgetting about the hidden debt until it bites. So we ask: *do big pension holes "
            "predict lower returns?*\n\n"
            "> ⚠️ **A catch we're upfront about.** The precise pension-hole numbers live in "
            "accounting footnotes that **cost money** to get in bulk (Compustat). A free retail data "
            "feed doesn't carry them. So we **cannot** test this on real stocks here — instead we "
            "build a realistic *synthetic* world where we control the truth, and check that our "
            "measuring tools actually work. That's an honest machinery demo, **not** a claim about "
            "real markets.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> Not investment advice. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the pension-hole effect real in the literature? | **Yes** — Franzoni-Marin (2006) "
            "document it: biggest holes → ~7%/yr *lower* returns. |\n"
            "| Can we prove it on real stocks here? | **No** — the pension-footnote data isn't free. "
            "So the Signal stamp is **None** (not evidenced on a real tape), by honest necessity. |\n"
            f"| Does our engine at least *work*? | **Yes** — on a synthetic world with a planted "
            f"hole-effect it cleanly finds it (the well-funded firms beat the underfunded ones by "
            f"**+{R['spread']}%**), and finds *nothing* when there's nothing to find. |\n"
            "| Could you trade it? | **No** — you'd short stressed, hard-to-borrow firms, and you "
            "can't even build the signal without paying for the data. |\n\n"
            "> The short version: a real anomaly the free retail stack simply **cannot reach**. We "
            "show the tool is sharp; we just don't have the tape to point it at."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The most underfunded pension firms earn anomalously low returns.\"* — the "
            "pension-underfunding anomaly (Franzoni & Marin 2006).\n\n"
            "Why it's a *puzzle*: a big pension hole is like hidden debt. More debt should mean more "
            "risk and — in theory — higher expected returns to compensate. Instead the data says the "
            "opposite: the most-underfunded firms *underperform*. The story is that the hole is "
            "buried in a footnote and slowly smoothed by pension accounting, so investors "
            "**under-react** — they don't mark the stock down enough, and it drifts lower as the "
            "truth leaks out through cash contributions and earnings hits."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it says the market is *inefficient* about a specific, knowable accounting item "
            "— a footnote everyone could read but few fully price. And it suggests a trade: **own the "
            "well-funded firms, avoid (or short) the underfunded ones.** The desk has looked at the "
            "*on*-balance-sheet [154 Leverage-Anomaly](../../154-leverage-anomaly/); this is its "
            "sneaky *off*-balance-sheet cousin — leverage hiding in the pension footnote."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the hole.** For each firm: (plan assets − pension promises) ÷ company "
            "value. Negative = underfunded. The more negative, the deeper the hole.\n"
            "2. **Sort.** Rank firms by hole depth; take the *well-funded* fifth and the "
            "*most-underfunded* fifth.\n"
            "3. **Compare forward returns.** Did the well-funded fifth beat the underfunded fifth "
            "over the next year? The puzzle says yes.\n\n"
            "**The honest catch:** step 1 needs the pension footnote, which isn't in any free feed. "
            "So we build a *synthetic* world where we know the true hole for 300 firms and can dial "
            "the effect on and off — then check our sorter finds it when it's there and stays quiet "
            "when it isn't.\n\n"
            "*Timing:* the hole is read from the last published report; we enter **after** it's "
            "public and hold forward. No peeking."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the synthetic world\n\n"
            "**In a world where the pension-hole effect is real, do the well-funded firms win?** "
            "(This is our synthetic tape — the truth is planted, so we *should* see it.)"
        ),
        code(
            "b = st.bucket_returns(PANEL, frac=0.2); ls = st.long_short_tstat(b)\n"
            "well_m, under_m, spr, t = b['well_mean']*100, b['under_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['WELL-FUNDED\\nfifth','UNDERFUNDED\\nfifth'], [well_m, under_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'The puzzle: well-funded win (+{well_m:.0f}%) vs underfunded ({under_m:+.0f}%)')\n"
            "fig.suptitle('SYNTHETIC world (planted effect) — NOT a real tape', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Well-funded +{well_m:.1f}%  vs  Underfunded {under_m:+.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"There's the puzzle, the right way round: the well-funded fifth earned **+{R['well_mean']}%** "
            f"and the deeply-underfunded fifth **{R['under_mean']}%** — a **+{R['spread']}%** gap "
            f"(*t* {R['ls_t']}). Our sorter finds the planted effect. Good — the tool works. "
            "**Remember: this is the synthetic world, not real stocks.**"
        ),
        md(
            "**And when there's *no* effect to find, does the tool stay quiet?** Turn the knob to "
            "zero (no pension-hole effect) and re-sort:"
        ),
        code(
            "null_panel, _ = data.synthetic_panel(underfunding_alpha=0.0)\n"
            "bn = st.bucket_returns(null_panel, 0.2); lsn = st.long_short_tstat(bn)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['effect ON\\n(planted)','effect OFF\\n(null)'],\n"
            "       [st.bucket_returns(PANEL,0.2)['spread']*100, bn['spread']*100],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('well - underfunded spread %')\n"
            "ax.set_title(f'Effect ON -> big gap; effect OFF -> ~0 ({bn[\"spread\"]*100:+.1f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null-world spread {bn[\"spread\"]*100:+.2f}% (t {lsn[\"t\"]:+.2f}) — flat, as it should be')"
        ),
        md(
            "With the effect switched off the gap collapses to roughly zero — the tool doesn't "
            "invent a signal out of noise. So the engine is honest in **both** directions."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Not because the anomaly is fake (the literature says it's real), "
            "but because **we have no real tape** — the pension-footnote data isn't free, and the "
            "desk only awards a real signal for a robust result on *real* data.\n"
            "- **Tradability — Mirage.** The short leg is stressed, hard-to-borrow firms, and you "
            "can't even source the signal cheaply."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — twice over. First, to *build* the signal you'd need to pay for the pension-footnote "
            "data. Second, the trade means *shorting* the most-underfunded firms — often stressed, "
            "old-economy names that are expensive and hard to borrow. Even in our synthetic world, "
            f"where the effect exists, a 150 bps borrow trims the **+{R['spread']}%** gross gap to "
            f"**+{R['net']}%** net."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The on-balance-sheet cousin.** [154 Leverage-Anomaly](../../154-leverage-anomaly/) "
            "tests *visible* debt; this is the *hidden*, pension kind.\n"
            "- **The under-reaction family.** [231 Sloan-Accruals](../../231-sloan-accruals/) — the "
            "market under-reacting to another opaque accounting item.\n"
            "- **The real test needs the footnotes.** With Compustat pension items (PBO, plan "
            "assets, funded status) you could run the true Franzoni-Marin sort on a real,"
            " survivorship-free panel — that's where the anomaly earns its keep.\n\n"
            "*Have the pension-footnote data? Fork this, drop in a real point-in-time funded-status "
            "panel, and show the well-minus-underfunded spread clearing t = 2 on the tape.*"
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
            "# The Pension-Underfunding Anomaly — a quantitative teardown 🔬\n"
            "### Funded-status signal · quintile sort · two-sample *t* · label-shuffle placebo · firm-level slope/IC · robustness sweep · costs & borrow · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the Franzoni-Marin "
            "funded-status measure, sort the cross-section, and test whether the most-underfunded "
            "firms earn the lowest returns (the puzzle).\n\n"
            "> ⚠️ **Synthetic-only, and we say so loudly.** The point-in-time pension-footnote panel "
            "(PBO, plan assets, funded status / market cap) is **not free** — no yfinance-class feed "
            "carries it. So `HAVE_REAL` is `False` by construction and there is **no real-tape "
            "banner** anywhere in this notebook: every number below is the deterministic synthetic "
            f"world (seed 571, panel fp `{R['fp_panel']}`). The Signal axis is capped at `NONE` "
            "accordingly. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | **No real tape** (pension footnotes not free). On the *synthetic* "
            f"world the engine recovers the planted puzzle: quintile spread **+{R['spread']}%** (two-sample "
            f"*t* **{R['ls_t']}**, placebo *p* {R['placebo_p']}), firm slope-*t* **{R['slope_t']}**, IC "
            f"**{R['ic']}**; flat at the null (slope-*t* +0.01). |\n"
            f"| **Tradability** | `MIRAGE` | Short leg = stressed, hard-to-borrow names; gross "
            f"**+{R['spread']}%** → net **+{R['net']}%** ({R['cost_bps']:.0f} bps/leg + {R['borrow_bps']:.0f} "
            "bps borrow) even in the synthetic world — and the signal itself is unsourceable for free. |\n\n"
            "> 💡 In plain words: the engine is a faithful detector (the control proves it), but with "
            "no real pension-footnote tape the desk cannot award a real signal — `NONE` by honesty, "
            "not by a failed test."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $H_i$ be firm $i$'s pension-hole depth (z-scored $-(\\text{assets}-\\text{PBO})/"
            "\\text{mktcap}$; higher = more underfunded) and $r_i$ its forward return.\n\n"
            "- **H₁ (the puzzle / long-short).** $\\mathbb{E}[r \\mid \\text{well-funded}] - "
            "\\mathbb{E}[r \\mid \\text{underfunded}] > 0$ at *t* ≥ 2.\n"
            "- **H₂ (firm-level).** $\\text{slope of } r \\text{ on } H < 0$ (deeper hole, less "
            "return) — the *sign* is the puzzle.\n"
            "- **H₃ (robustness).** The sign is stable across bucket definitions and weighting.\n"
            "- **H₄ (net).** H₁ survives costs and the underfunded-leg borrow.\n\n"
            "**The binding constraint is upstream of all four:** there is **no real tape** to test "
            "them on. So we test them on the *synthetic* world (where we can plant and null the "
            "effect) to certify the engine, and leave the real-tape verdict honestly open (`NONE`)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Franzoni-Marin's mechanism is *market under-reaction to an "
            "opaque accounting item* — the same family as [231 Sloan-Accruals](../../231-sloan-accruals/) "
            "and [522 %-Operating-Accruals](../../522-percent-operating-accruals/). The pension hole "
            "is off-balance-sheet leverage: economically senior, buried in a footnote, smoothed by "
            "pension accounting. If the market prices it slowly, the most-underfunded firms drift "
            "down. Testing that needs point-in-time funded status — the data we don't have for free, "
            "which is precisely why this study is synthetic-only."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $H = z\\big(-(\\text{assets}-\\text{PBO})/\\text{mktcap}\\big)$ — hole "
            "depth, z-scored (higher = more underfunded).\n"
            "- **Sort.** Quintile tails (60 firms each): well-funded (smallest $H$) vs underfunded "
            "(largest $H$).\n"
            "- **Inference.** Two-sample (Welch) *t* on the well − underfunded forward returns; a "
            "**label-shuffle placebo** null (2000 perms).\n"
            "- **Firm-level.** OLS of forward return on $H$ + the information coefficient — the *sign* "
            "is the puzzle.\n"
            "- **Robustness.** Quintile/decile/tercile tails, equal- and cap-weighted.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the underfunded short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted puzzle "
            "(`underfunding_alpha < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the hole is read from the last published report; entry is **after** it is public "
            "and the return is measured forward. In the synthetic panel this is baked in (`forward_ret` "
            "post-dates the funding observation) — no look-ahead. The report→entry lag is the single "
            "documented convention."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown (synthetic world)"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Well-funded vs underfunded forward returns, with the two-sample *t* and the placebo null."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2); ls = st.long_short_tstat(b)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=571)\n"
            "well_m, under_m, spr, t = b['well_mean']*100, b['under_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['WELL-FUNDED','UNDERFUNDED'], [well_m, under_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Well - underfunded = {spr:+.1f}%  (two-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "fig.suptitle('SYNTHETIC world (planted effect) — not a real tape', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Well +{well_m:.1f}% | Underfunded {under_m:+.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: on the planted world H₁ is **recovered** — spread **+{R['spread']}%** "
            f"(*t* {R['ls_t']}), placebo *p* = {R['placebo_p']} (deep in the tail). The engine detects "
            "the puzzle when it is present. (This certifies the tool; it is **not** a real-tape claim.)"
        ),
        md(
            "### 4b · The firm-level slope / IC — H₂\n\n"
            "Regress forward return on hole depth across all 300 firms. A *negative* slope is the "
            "puzzle."
        ),
        code(
            "reg = st.cross_section_regression(PANEL)\n"
            "d = st.hole_depth(PANEL); y = PANEL['forward_ret']*100\n"
            "slope, slope_t, ic = reg['slope']*100, reg['slope_t'], reg['ic']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(d, y, s=14, color=GREY, alpha=.6)\n"
            "xs = np.linspace(d.min(), d.max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs + (y.mean()/100 - reg['slope']*d.mean()))*100, c=RED, lw=2)\n"
            "ax.set_xlabel('pension-hole depth (higher = more underfunded)'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Slope {slope:+.1f}%/unit (t {slope_t:+.2f}), IC {ic:+.2f} — NEGATIVE = the puzzle')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:+.2f}%/unit, slope-t {slope_t:+.2f}, IC {ic:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **recovered** — slope **{R['slope']}%** per unit (*t* {R['slope_t']}), "
            f"IC **{R['ic']}**. Deeper hole → lower return, exactly Franzoni-Marin's sign, on the "
            "planted world."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign survive design choices?)\n\n"
            "The same long-short across bucket definitions and weighting."
        ),
        code(
            "rows = st.robustness_sweep(PANEL)\n"
            "tab = pd.DataFrame(rows)[['design','spread','t','slope_t']].copy()\n"
            "tab['spread'] = tab['spread']*100\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(range(len(tab)), tab['spread'], color=GREEN, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab['design'], rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('well - underfunded spread %')\n"
            "ax.set_title('Sign is stable across bucket definitions and weighting')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H₃ **holds** on the synthetic world — the spread stays positive "
            "across quintile/decile/tercile and equal/cap weighting (a touch smaller cap-weighted, as "
            "mega-caps carry smaller holes). Design-robust *when the effect exists*."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — **no real tape** (pension footnotes not free). The synthetic "
            f"control recovers H₁/H₂/H₃ (spread +{R['spread']}%, slope-*t* {R['slope_t']}, IC {R['ic']}), "
            "but the desk awards a real signal only for a robust *t* ≥ 2 on **real** data.\n"
            f"- **Tradability `MIRAGE`** — short leg is stressed/hard-to-borrow; gross +{R['spread']}% "
            f"→ net +{R['net']}%; and the signal is unsourceable for free."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Even where the effect exists (the synthetic world), the underfunded short leg carries a "
            "punitive borrow."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2)\n"
            "gross = b['spread']*100\n"
            "net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=150.0, holding_years=1.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('well - underfunded spread %')\n"
            "ax.set_title(f'Gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 150 bps borrow)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (synthetic world; real short leg is worse)')"
        ),
        md(
            "> 💡 In plain words: a 150 bps borrow on the stressed underfunded leg trims ~1.7pp even in "
            "the friendly synthetic world. On a real tape that leg is the expensive-to-borrow tail — "
            "and you cannot source the signal for free to begin with. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a puzzle? Plant an effect "
            "(`underfunding_alpha < 0`) of increasing strength and watch the firm-level slope-*t* go "
            "negative — averaged over 25 seeds so no single lucky seed can fake it — and confirm it "
            "sits at ~0 at the null."
        ),
        code(
            "alphas = [0.0, -0.01, -0.02, -0.035, -0.05, -0.065, -0.08]\n"
            "stats = [st.synthetic_mean_stats(data, underfunding_alpha=a, n_seeds=25) for a in alphas]\n"
            "ts = [s['mean_slope_t'] for s in stats]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (puzzle)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted underfunding_alpha (more negative = stronger puzzle)')\n"
            "ax.set_ylabel('mean firm slope-t (25 seeds)')\n"
            "ax.set_title('Faithful detector — flat at the null, past -2 as the planted puzzle grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for s in stats: print(f\"alpha {s['alpha']:+.3f} -> mean slope-t {s['mean_slope_t']:+.2f}, mean IC {s['mean_ic']:+.3f}\")"
        ),
        md(
            "At the null the slope-*t* is ≈ 0 (no false signal); it falls past −2 as the planted "
            "puzzle grows. So the engine is a faithful detector — which means the real-tape verdict "
            "here is a **data limitation, not a broken tool**: with point-in-time pension footnotes "
            "(Compustat) one could run the true Franzoni-Marin test. For the on-balance-sheet cousin "
            "see [154 Leverage-Anomaly](../../154-leverage-anomaly/); for the same under-reaction "
            "family see [231 Sloan-Accruals](../../231-sloan-accruals/)."
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
