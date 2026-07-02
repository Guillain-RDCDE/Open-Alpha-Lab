"""Generate the two narrative notebooks for Study 570 (Goodwill-Impairment).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../METHODOLOGY.md). This is a **synthetic-only**
study: there is no real tape (a point-in-time goodwill / impairment panel is out of reach for a
no-key stack), so every cell runs on the deterministic synthetic world in
``goodwill_impairment/data.py`` — fully offline and reproducible. The dict ``R`` mirrors the frozen
headline numbers in ``docs/results.md``.
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


# Frozen synthetic headline numbers — mirror of docs/results.md
# (ret_alpha=-0.14, imp_beta=2.2, seed=570, n=400; panel fp 6bd491ebb228; as-of 2026-06-30).
R = dict(
    fp_panel="6bd491ebb228", fp_truth="8710cfb12768",
    n=400, k=80, ret_alpha=-0.14, imp_beta=2.2,
    low_mean=9.42, high_mean=3.64, spread=5.78, ls_t=3.86, placebo_p=0.0005,
    slope=-17.3, slope_t=-4.22, corr=-0.21,
    low_imp=6.2, high_imp=12.5, imp_gap=6.2, imp_z=1.36,
    gross=5.78, net=4.08, cost_bps=5.0, borrow_bps=150.0,
    # robustness: (frac, spread%, welch_t)
    sweep=[(0.10, 7.71, 3.88), (0.20, 5.78, 3.86), (0.25, 5.30, 3.94), (0.30, 4.79, 3.72)],
    # control: (ret_alpha, mean slope-t over 25 seeds, mean ls-t over 25 seeds)
    ctrl=[(0.0, -0.05, 0.13), (-0.05, -1.99, 1.80), (-0.10, -3.77, 3.31),
          (-0.14, -5.02, 4.35), (-0.20, -6.57, 5.61)],
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

from goodwill_impairment import data, strategy as st

# Synthetic-only study: the headline world is deterministic and offline. There is NO real tape.
PANEL, TRUTH = data.synthetic_panel(ret_alpha=-0.14, imp_beta=2.2, seed=570)
HAVE_REAL = False   # a point-in-time goodwill/impairment panel is not reachable (see data.py)
print("synthetic goodwill panel:", len(PANEL), "firms | panel fp",
      data.fingerprint(PANEL), "| real tape present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Goodwill Impairment — can a bloated balance sheet warn you before the write-down? 🧾\n"
            "### When a company overpays for a takeover, the bill shows up later. Can you see it coming?\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "When one company buys another for **more than the target's assets are worth**, the extra "
            "money it paid gets parked on the balance sheet under a line called **goodwill**. It just "
            "sits there — until an accountant decides the acquisition isn't worth what was paid, and "
            "forces a **write-down** (an *impairment*). That write-down is an admission: *we overpaid.*\n\n"
            "The idea we're testing: a firm carrying a **big pile of goodwill** relative to its assets "
            "is a firm that probably overpaid — so it should (a) be **more likely to write it down** "
            "later, and (b) see its **stock underperform** as that bad news leaks out. In short: can a "
            "bloated goodwill balance predict the write-down *and* the drop, before they happen?\n\n"
            "> ⚠️ **Big caveat, stated up front.** To test this on *real* companies you'd need "
            "point-in-time goodwill figures and tagged write-down events for thousands of firms — data "
            "a free retail stack simply can't reach. So this study runs on a **synthetic (made-up but "
            "rule-governed) world** where we *know* the answer, to check whether the method works. It "
            "can't earn a `REAL` stamp — only a `WEAK` one.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does high goodwill predict the write-down? | **Yes, in the model** — the most "
            f"goodwill-heavy firms impaired at **{R['high_imp']}%** vs **{R['low_imp']}%** for the "
            "lean ones. |\n"
            "| Does high goodwill predict the *drop*? | **Yes, in the model** — the lean-goodwill "
            f"firms beat the bloated ones by **+{R['spread']}%** over the forward window. |\n"
            "| Is it just luck? | **No** — a shuffle test puts the odds at *p* = "
            f"{R['placebo_p']}. The effect the model planted is cleanly recovered. |\n"
            "| So can you trade it? | **Not from here.** There's no real data to test it on, so this "
            "is a *method demo*, not a live signal. |\n\n"
            "> The engine works — it catches the effect when it's really there and stays silent when "
            "it isn't. But *catching a planted effect in a simulation* is not the same as *finding it "
            "on the real market*. For that you'd need accounting data we can't get for free."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A high goodwill-to-assets ratio flags an overpaid acquisition; that overpayment "
            "later surfaces as a write-down, and high-goodwill firms underperform.\"*\n\n"
            "Why it's plausible: goodwill only exists because someone paid a premium. If the premium "
            "was justified, fine — but a *lot* of goodwill relative to the whole company is a bet that "
            "the acquisition will pay off. When it doesn't, accounting rules eventually force the firm "
            "to admit it (the impairment), and managers are known to **delay** that admission — which "
            "is exactly why the market can be slow to price it (Hayn & Hughes 2006; Li et al. 2011)."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it were reliably true on real data, it'd be a clean red-flag screen: avoid (or short) "
            "the serial over-payers before the write-down lands. It also connects to a broader theme "
            "the desk keeps finding — balance-sheet 'quality' flags that *sound* predictive. The "
            "accruals studies ([231](../../231-sloan-accruals/), [522](../../522-percent-operating-accruals/)) "
            "chase a cousin idea (earnings quality); here it's the **goodwill / overpaid-M&A** angle."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score each firm** by goodwill ÷ total assets — higher means more 'bloated'.\n"
            "2. **Sort** the firms into a *lean* fifth (lowest goodwill) and a *bloated* fifth "
            "(highest).\n"
            "3. **Look forward:** did the bloated fifth (a) write down more often, and (b) earn lower "
            "returns than the lean fifth?\n\n"
            "*Timing:* the goodwill number is this year's balance (known today); the write-downs and "
            "returns are all in the *future*. So the signal never peeks ahead.\n\n"
            "*The wall:* we can't do steps 1–3 on real firms — the point-in-time goodwill history and "
            "the tagged write-down events aren't in any free feed. So we build a synthetic world with "
            "a **knob** that plants the effect, and check the method catches it."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Lean fifth vs bloated fifth: who wins, and who writes down more?**"
        ),
        code(
            "b = st.bucket_returns(PANEL, frac=0.2)\n"
            "low_m, high_m = b['low_mean']*100, b['high_mean']*100\n"
            "low_i, high_i = b['low_imp_rate']*100, b['high_imp_rate']*100\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "ax1.bar(['LEAN\\ngoodwill','BLOATED\\ngoodwill'], [low_m, high_m], color=[GREEN, RED], width=.55)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_ylabel('forward return %')\n"
            "ax1.set_title(f'Lean beats bloated (+{low_m:.1f}% vs +{high_m:.1f}%)')\n"
            "ax2.bar(['LEAN\\ngoodwill','BLOATED\\ngoodwill'], [low_i, high_i], color=[GREEN, RED], width=.55)\n"
            "ax2.set_ylabel('impairment rate %')\n"
            "ax2.set_title(f'Bloated writes down more ({high_i:.1f}% vs {low_i:.1f}%)')\n"
            "fig.suptitle('synthetic world (effect planted) — the method demo', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'return: lean +{low_m:.1f}% vs bloated +{high_m:.1f}% (spread +{low_m-high_m:.1f}%)')\n"
            "print(f'impair: lean {low_i:.1f}% vs bloated {high_i:.1f}% (gap +{high_i-low_i:.1f}pp)')"
        ),
        md(
            f"Both legs of the story show up **in the model**: the bloated fifth wrote down about "
            f"twice as often (**{R['high_imp']}%** vs **{R['low_imp']}%**) and earned **{R['spread']}pp** "
            "less. That's exactly what the overpaid-acquisition idea predicts — *because we planted it*. "
            "The real question is whether the method is honest, which is beat 7."
        ),
        md(
            "**Does the pattern hold if we slice the buckets differently?** (deciles, quintiles, "
            "quartiles, thirds)"
        ),
        code(
            "sweep = " + repr([(s[0], s[1]) for s in R["sweep"]]) + "\n"
            "fracs = [s[0] for s in sweep]\n"
            "spreads = [st.bucket_returns(PANEL, f)['spread']*100 for f in fracs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([f'{int(f*100)}%' for f in fracs], spreads, color=GREEN, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('lean - bloated spread %')\n"
            "ax.set_xlabel('bucket tail fraction')\n"
            "ax.set_title('The lean-beats-bloated gap holds across bucket sizes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by bucket fraction:', [round(s,1) for s in spreads])"
        ),
        md(
            "The gap stays positive whether we use the top/bottom 10% or 30% — the sign is stable *on "
            "this world*. (Whether it'd be stable on the **real** market is exactly what we can't "
            "check.)"
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The idea is coherent and literature-backed, and the engine catches "
            "it cleanly *when it's planted*. But there's **no real tape** — so it can't earn `REAL`, "
            "only `WEAK`.\n"
            "- **Tradability — Mirage.** No real result means nothing to trade; and the bloated-goodwill "
            "names you'd short are exactly the illiquid, corporate-action-prone ones."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not from here. Even if the effect is real on the full market, the trade means **shorting** "
            "the most goodwill-bloated firms — often smaller, recently-merged names that are expensive "
            "to borrow and prone to corporate actions. And we have no real data to size, cost, or "
            "confirm any of it. A method that *works in simulation* is a starting point, not a signal."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The honest test needs real accounting data.** Point-in-time goodwill/assets at each "
            "year-end, plus tagged impairment events, for a survivorship-free universe — Compustat-"
            "grade data. Re-run the sort there and see if the lean-minus-bloated spread clears *t* = 2.\n"
            "- **Cousins on the bench.** The accruals anomalies "
            "([231](../../231-sloan-accruals/), [522](../../522-percent-operating-accruals/)) and the "
            "[distress-risk puzzle](../../540-distress-risk-anomaly/) test related balance-sheet flags.\n\n"
            "*Think you can get the data? Fork this, drop in a real point-in-time goodwill/impairment "
            "panel, and show the spread clearing t = 2 the right way.*"
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
            "# Goodwill Impairment — a quantitative teardown 🔬\n"
            "### goodwill/assets signal · quintile sort · two-sample *t* · label-shuffle placebo · impairment-rate *z* · firm-level slope · bucket-fraction sweep · costs & borrow · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a high "
            "**goodwill / total-assets** ratio predicts (a) higher impairment rates and (b) lower "
            "forward returns — the overpaid-acquisition puzzle.\n\n"
            "> ⚠️ **Synthetic-only, by necessity.** A point-in-time goodwill / impairment / "
            "forward-return panel is not reachable from a no-key retail stack, so there is **no real "
            "tape** and this study is capped at `WEAK` (a synthetic-only study can never earn `REAL`). "
            f"The deterministic world (`ret_alpha=-0.14`, seed 570, panel fp `{R['fp_panel']}`) is the "
            "positive control; every cell is offline and reproducible. Methods in "
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
            f"| **Signal** | `WEAK` | **No real tape** (synthetic-only ⇒ capped at `WEAK`). On the "
            f"planted world: lean − bloated spread **+{R['spread']}%** (Welch *t* **+{R['ls_t']}**, "
            f"placebo *p* {R['placebo_p']}); firm slope-*t* **{R['slope_t']}**; flat at the null "
            "(mean slope-*t* −0.05 / 25 seeds). Literature leans the right way. |\n"
            f"| **Tradability** | `MIRAGE` | Nothing real to trade; the bloated short leg is the "
            f"hard-to-borrow tail. Synthetic gross +{R['gross']}% → net +{R['net']}% (5 bps/leg + "
            f"{R['borrow_bps']:.0f} bps borrow) — net-of-nothing-real. |\n\n"
            "> 💡 In plain words: the engine is a faithful detector (proved by the control), but a "
            "faithful detector with no real tape to point at is a `WEAK` × `MIRAGE` result — an idea "
            "and a method, not a signal."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $G_i$ be firm $i$'s goodwill/assets ratio, $I_i \\in \\{0,1\\}$ a forward impairment "
            "event, and $r_i$ its forward return.\n\n"
            "- **H₁ (return long-short).** $\\mathbb{E}[r \\mid \\text{lean}] - "
            "\\mathbb{E}[r \\mid \\text{bloated}] > 0$ at *t* ≥ 2 (lean beats bloated).\n"
            "- **H₂ (impairment link).** $\\Pr(I \\mid \\text{bloated}) > \\Pr(I \\mid \\text{lean})$.\n"
            "- **H₃ (firm-level).** slope of $r$ on $G < 0$ (more goodwill, less return).\n"
            "- **H₄ (robustness).** the sign of H₁/H₃ holds across bucket definitions.\n"
            "- **H₅ (net).** H₁ survives costs and the bloated-leg borrow.\n\n"
            "On the **synthetic positive control** all five hold as planted. On a **real tape** we "
            "cannot test any of them — the data is unreachable — which is the whole reason the stamp "
            "is `WEAK`, not `REAL`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₅ held on real data, the goodwill/assets ratio would be a tradable red-flag "
            "screen. The literature supports the *direction* — high goodwill and overpaid stock-"
            "financed deals predict write-downs and negative drift (Li et al. 2011; Gu & Lev 2011; "
            "Hayn & Hughes 2006), and managers **delay** the write-down (Ramanna & Watts 2012), which "
            "is the friction that makes the balance predictive. But direction-in-the-literature is "
            "`WEAK` support, not a `REAL` certification, and we have no tape to close the gap."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $G = $ goodwill / total assets (higher = more overpaid).\n"
            "- **Sort.** Quintile tails (80 firms each): lean (lowest $G$) vs bloated (highest).\n"
            "- **Return inference.** Two-sample (Welch) *t* on lean − bloated forward returns; a "
            "**label-shuffle placebo** null (2000 perms).\n"
            "- **Impairment link.** Two-proportion *z* on the bloated − lean impairment-rate gap.\n"
            "- **Firm-level.** OLS of forward return on $G$ — the *sign* of the slope is the puzzle.\n"
            "- **Robustness.** Re-sort at bucket fractions 0.10 / 0.20 / 0.25 / 0.30.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the bloated short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted effect "
            "(`ret_alpha < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: $G$ is the year-*t* balance (public at scoring); $I$ and $r$ are strictly "
            "forward. No future data enters the signal (no look-ahead). One documented lag: score at "
            "year-*t* close, hold the lean-minus-bloated book over the forward window."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The return sort — H₁\n\n"
            "Lean vs bloated forward returns, with the two-sample *t* and the placebo null."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2); ls = st.long_short_tstat(b)\n"
            "p = st.placebo_pvalue(PANEL, 0.2, n_perm=2000, seed=570)\n"
            "low_m, high_m, spr, t = b['low_mean']*100, b['high_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['LEAN','BLOATED'], [low_m, high_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Lean - bloated = {spr:+.1f}%  (Welch t {t:+.2f}, placebo p {p:.4f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'lean +{low_m:.2f}% | bloated +{high_m:.2f}% | spread {spr:+.2f}% | t {t:+.2f} | placebo p {p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: on the planted world H₁ is recovered — spread **+{R['spread']}%** "
            f"(*t* +{R['ls_t']}), placebo *p* {R['placebo_p']}. The detector fires when the effect is "
            "really there."
        ),
        md(
            "### 4b · The impairment link — H₂\n\n"
            "Does bloated goodwill actually predict the write-down? Two-proportion *z* on the gap. "
            "(This link is planted by `imp_beta` and survives even at the return-null — the accounting "
            "fact is separate from the mis-pricing.)"
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2); gap = st.impairment_gap(b)\n"
            "# show the link persists at the return-null too\n"
            "pn, _ = data.synthetic_panel(ret_alpha=0.0, seed=570)\n"
            "gapn = st.impairment_gap(st.bucket_returns(pn, 0.2))\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['LEAN','BLOATED'], [gap['low_rate']*100, gap['high_rate']*100], color=[GREEN, RED], width=.5)\n"
            "ax.set_ylabel('impairment rate %')\n"
            "ax.set_title(f'Bloated impairs more: {gap[\"gap\"]*100:+.1f}pp gap (two-prop z {gap[\"z\"]:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'effect world: lean {gap[\"low_rate\"]*100:.1f}% vs bloated {gap[\"high_rate\"]*100:.1f}% | gap {gap[\"gap\"]*100:+.1f}pp | z {gap[\"z\"]:+.2f}')\n"
            "print(f'return-null:  gap {gapn[\"gap\"]*100:+.1f}pp | z {gapn[\"z\"]:+.2f}  (link persists — it is the accounting fact, not the mis-pricing)')"
        ),
        md(
            f"> 💡 In plain words: H₂ holds — the bloated fifth impairs **{R['imp_gap']}pp** more "
            f"(*z* +{R['imp_z']}), and the gap is identical at the return-null. Bloated goodwill "
            "writes down more *whether or not* the market mis-prices it — exactly the real story."
        ),
        md(
            "### 4c · The firm-level slope — H₃\n\n"
            "Regress forward return on goodwill/assets across all 400 firms. A *negative* slope is the "
            "puzzle."
        ),
        code(
            "reg = st.cross_section_regression(PANEL)\n"
            "g = st.goodwill_signal(PANEL); y = PANEL['forward_ret']*100\n"
            "slope, slope_t, corr = reg['slope']*100, reg['slope_t'], reg['corr']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(g, y, s=10, color=GREY, alpha=.5)\n"
            "xs = np.linspace(g.min(), g.max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs + (y.mean()/100 - reg['slope']*g.mean()))*100, c=RED, lw=2)\n"
            "ax.set_xlabel('goodwill / total assets'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Slope {slope:+.1f}%/unit (t {slope_t:+.2f}) — NEGATIVE = the puzzle')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:+.1f}%/unit, slope-t {slope_t:+.2f}, corr(goodwill, fwd) {corr:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₃ recovered — slope **{R['slope']}%** per unit (*t* {R['slope_t']}), "
            f"corr **{R['corr']}**. More goodwill, less return — the planted overpaid-M&A drag."
        ),
        md(
            "### 4d · Robustness — H₄ (does the sign hold across bucket cuts?)"
        ),
        code(
            "sweep = st.robustness_sweep(PANEL, fracs=(0.10, 0.20, 0.25, 0.30))\n"
            "tab = pd.DataFrame(sweep)\n"
            "tab['spread'] = tab['spread']*100\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([f'{int(f*100)}%' for f in tab['frac']], tab['spread'], color=GREEN, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('lean - bloated spread %'); ax.set_xlabel('bucket tail fraction')\n"
            "ax.set_title('Sign stable across bucket definitions (all t > 3.5)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab[['frac','spread','t','slope_t']].round(2).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H₄ holds on this world — the spread is positive and *t* > 3.5 at "
            "every cut. (Sign-stability on a **real** tape is precisely what the missing data prevents "
            "us from checking.)"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — no real tape (synthetic-only ⇒ capped at `WEAK`); on the control "
            f"H₁–H₄ recovered (spread +{R['spread']}%, *t* +{R['ls_t']}; slope-*t* {R['slope_t']}; "
            "flat at the null). Literature supports the direction.\n"
            f"- **Tradability `MIRAGE`** — nothing real to trade; bloated short = hard borrow; "
            f"synthetic gross +{R['gross']}% → net +{R['net']}%.\n"
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "On the synthetic world the spread survives costs; on the real market there's nothing to "
            "trade, and the bloated short leg is the hard-to-borrow tail."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2)\n"
            "gross = b['spread']*100\n"
            "net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=150.0, holding_years=1.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('lean - bloated spread %')\n"
            "ax.set_title(f'Synthetic gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 150 bps borrow)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}% -> net {net:+.2f}%  (synthetic world; no real tape to trade)')"
        ),
        md(
            "> 💡 In plain words: costs shave ~1.7pp on the synthetic world, so the *method* would "
            "leave something after frictions — but this is net-of-nothing-real. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a spread? Plant an "
            "overpaid-M&A effect (`ret_alpha < 0`) of increasing strength and watch the firm-level "
            "slope-*t* and the long-short *t* move — averaged over 25 seeds so no single lucky seed "
            "can fake it."
        ),
        code(
            "alphas = [0.0, -0.05, -0.10, -0.14, -0.20]\n"
            "res = [st.synthetic_mean_t(data, ret_alpha=a, n_seeds=25, base_seed=570) for a in alphas]\n"
            "slope_ts = [r['slope_t'] for r in res]; ls_ts = [r['ls_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.plot(alphas, slope_ts, 'o-', c=RED, lw=2, label='firm slope-t')\n"
            "ax.plot(alphas, ls_ts, 's-', c=GREEN, lw=2, label='long-short t')\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted ret_alpha (more negative = stronger overpaid-M&A effect)')\n"
            "ax.set_ylabel('mean t over 25 seeds')\n"
            "ax.set_title('Flat at the null, past +/-2 as the planted effect grows — a faithful detector')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, r in zip(alphas, res): print(f'ret_alpha {a:+.2f} -> mean slope-t {r[\"slope_t\"]:+.2f} | mean ls-t {r[\"ls_t\"]:+.2f}')"
        ),
        md(
            "Both stats sit ≈ 0 at the null and cross ±2 as the planted effect grows — the engine is a "
            "faithful detector, catching a real overpaid-M&A effect and staying silent on noise. That "
            "is why the honest verdict is **`WEAK`** (validated method + literature support) and not "
            "`NONE` — and why it **cannot** be `REAL`: there is no real goodwill/impairment tape to "
            "point it at. For the accruals cousins, see [231](../../231-sloan-accruals/) and "
            "[522](../../522-percent-operating-accruals/); for a related balance-sheet return puzzle, "
            "[540 distress-risk](../../540-distress-risk-anomaly/)."
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
