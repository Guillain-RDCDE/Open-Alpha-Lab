"""Generate the two narrative notebooks for Study 557 (Borrow-Fee-Signal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a **synthetic-only**
study — there is no free real borrow-fee tape — so every cell runs offline and deterministic on the
seed-557 synthetic world; ``HAVE_REAL`` is always False (kept for schema parity), and the frozen
headline numbers in ``R`` mirror docs/results.md. No cell ever sits under a real-tape banner.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 557,
# 120 names, planted fee_alpha=-0.035; panel fp f48c7a1477b7). Every number here was COMPUTED by
# running borrow_fee_signal/ — see docs/results.md.
R = dict(
    fp_panel="f48c7a1477b7", n=120, k=24, seed=557, fee_alpha=-0.035,
    fee_med=0.61, fee_p90=2.6, fee_max=8.5, corr_fee_si=0.60,
    cheap_mean=13.2, expensive_mean=0.5, spread=12.8, ls_t=2.53, placebo_p=0.009,
    short_leg_fee=3.1,
    slope=-4.2, slope_t=-2.58, corr=-0.23,
    inc_fee_slope=-4.3, inc_fee_t=-2.15, inc_si_slope=0.3, inc_si_t=0.15,
    net_q3m=11.8, net_q1y=9.4, net_d1y=7.0, top_decile_fee=4.1,
    # robustness: (frac, spread%, welch_t)
    sweep=[(0.10, 11.3, 1.32), (0.20, 12.8, 2.53), (0.30, 9.2, 2.11), (0.40, 9.1, 2.51)],
    # synthetic control: (fee_alpha, mean firm slope-t over 25 seeds, mean LS-t)
    ctrl=[(0.0, -0.00, -0.08), (-0.01, -0.61, 0.39), (-0.02, -1.22, 0.86),
          (-0.035, -2.13, 1.55), (-0.05, -3.05, 2.21)],
    # SI-only null: fee-sort incremental fee_t vs si_t when only SI carries signal
    si_null_fee_t=-0.42, si_null_si_t=-1.34,
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

from borrow_fee_signal import data, strategy as st

# Synthetic-only study: there is NO free real borrow-fee tape (private data). The reproducible core
# is the deterministic seed-557 world. HAVE_REAL is always False; nothing sits under a real banner.
PANEL, TRUTH = data.synthetic_panel()          # seed 557, planted fee_alpha=-0.035
HAVE_REAL = not data.fetch_panel(fetch=False).empty  # -> False by construction
print("synthetic panel:", len(PANEL), "names, fp", data.fingerprint(PANEL),
      "| planted fee_alpha", TRUTH.fee_alpha, "| real tape available:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Borrow-Fee Signal — does 'expensive to short' mean 'about to fall'? 💸\n"
            "### What the securities-lending fee is, and whether it predicts where a stock goes next\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Distinct_from_short_interest%3F: Yes](https://img.shields.io/badge/Distinct_from_short_interest%3F-Yes-2ea44f?style=flat-square)\n\n"
            "To bet *against* a stock you have to **borrow** it first — and someone charges you rent "
            "for the loan. That rent is the **borrow fee** (the securities-lending fee). For most "
            "stocks it's tiny (a fraction of a percent a year). But when lots of traders want to "
            "short the *same* name and there aren't many shares to lend, the fee spikes — sometimes "
            "to 20%, 50%, even 100%+ a year. Those names are called **hard-to-borrow** or "
            "**special**.\n\n"
            "Here's the claim researchers make: the stocks that are *most expensive to borrow* go on "
            "to earn the **lowest** returns. The idea is that a high fee is the fingerprint of a crowd "
            "of informed short sellers who know something. So we test it: sort stocks by borrow fee, "
            "and see whether the *cheap-to-borrow* ones beat the *expensive* ones.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** And a big honest caveat up front: **there is no free, "
            "real, historical borrow-fee dataset** — that data is private and expensive. So every "
            "chart below is drawn on a *synthetic* world we built to plant the effect and prove our "
            "engine can catch it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do the expensive-to-borrow stocks earn less? | **Yes, on our synthetic tape.** The "
            f"cheap-to-borrow bucket earned **+{R['cheap_mean']}%** vs the expensive bucket's "
            f"**+{R['expensive_mean']}%** — a **+{R['spread']}%** gap (*t* +{R['ls_t']}). |\n"
            f"| Is that just luck? | **Probably not** — a shuffle test puts the odds at *p* = "
            f"{R['placebo_p']}. The effect is real *in this planted world*. |\n"
            "| Is it just short interest in disguise? | **No.** The fee still predicts returns even "
            "after we account for how heavily a stock is shorted. The *price* of shorting says "
            "something the *quantity* doesn't. |\n"
            "| Could you trade it? | **Barely.** The catch is brutal: the stocks you'd want to short "
            "are *exactly* the expensive-to-borrow ones — so the fee that signals the trade is also "
            "the rent you pay to put it on. |\n\n"
            "> Two big honest limits: (1) **there's no real data** to confirm any of this on an actual "
            "tape — it's a private, paid dataset — so we can only stamp this **Weak**; and (2) even in "
            "our world the effect is *modest*, not a money pump."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The stocks that are most expensive to borrow earn the lowest future returns.\"* — the "
            "borrow-fee / shorting-premium literature (Cohen, Diether & Malloy 2007; D'Avolio 2002).\n\n"
            "Think of the lending market as supply and demand. **Supply** is how many shares are "
            "available to lend (depends on the float, who owns it, whether it's in big index funds). "
            "**Demand** is how many people want to short. When demand outruns supply, the price — the "
            "borrow fee — spikes. The claim is that a high fee means *informed, aggressive short "
            "sellers* are piling in, and they're usually right: the stock falls."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, the fee is a *cleaner* signal than the usual one people watch — short interest "
            "(what fraction of a stock is sold short). Two stocks can be shorted equally heavily, but "
            "one is cheap to borrow (lots of shares around) and one is on-special (thin supply). The "
            "claim says the *fee* — the price — is what carries the information, because it reflects "
            "the supply side that short interest can't see. The desk already looked at short interest "
            "in **[Study 262](../../262-short-interest/)** and found a coin flip; here we ask whether "
            "the *fee* does better."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Get the fee for each stock.** In the real world this is the hard part — the data is "
            "private. So we *build* a realistic fee world: most names near-free, a right tail of "
            "expensive 'special' names.\n"
            "2. **Sort.** Split into a *cheap-to-borrow* fifth and an *expensive-to-borrow* fifth.\n"
            "3. **Compare forward returns.** Did the cheap fifth beat the expensive fifth? The claim "
            "says yes.\n"
            "4. **Check it's not just short interest.** Re-do the test while holding short interest "
            "fixed — does the fee still matter?\n\n"
            "*Timing:* the fee is known on the day we form the sort; we enter at that close and hold "
            "forward. No peeking at the future.\n\n"
            "*The honest asterisk:* because the world is synthetic, this proves our **method** works "
            "and the effect is *plausible* — it can't prove the effect is real on the actual market. "
            "That's why the ceiling here is **Weak**."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what does the fee world look like?** Most stocks cheap, a few very expensive:"
        ),
        code(
            "fees = PANEL['borrow_fee']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.hist(fees, bins=40, color=GREY, edgecolor='white')\n"
            "ax.axvline(fees.median(), c=GREEN, lw=2, label=f'median {fees.median():.2f}%')\n"
            "ax.axvline(np.percentile(fees,90), c=RED, lw=2, ls='--', label=f'90th pct {np.percentile(fees,90):.1f}%')\n"
            "ax.set_xlabel('annualised borrow fee %'); ax.set_ylabel('number of stocks')\n"
            "ax.set_title('Most stocks near-free to borrow; a right tail of expensive \"special\" names')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'median {fees.median():.2f}%  90th pct {np.percentile(fees,90):.1f}%  max {fees.max():.1f}%')"
        ),
        md(
            "That's the real shape of the lending market: a big pile of general-collateral names you "
            "can borrow for almost nothing, and a thin, expensive tail of hard-to-borrow names.\n\n"
            "**Now the main event — did the cheap fifth beat the expensive fifth?**"
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2); ls = st.long_short_tstat(b)\n"
            "cheap_m, exp_m, spr, t = b['cheap_mean']*100, b['expensive_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['CHEAP to borrow','EXPENSIVE to borrow'], [cheap_m, exp_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Cheap-to-borrow beat expensive by {spr:+.0f}%  (t {t:+.2f})')\n"
            "fig.suptitle('synthetic world (seed 557) — no real borrow-fee tape exists', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cheap +{cheap_m:.1f}%  vs  Expensive +{exp_m:.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"There's the claim's sign: the **expensive-to-borrow** names (+{R['expensive_mean']}%) "
            f"badly lagged the **cheap** ones (+{R['cheap_mean']}%), a **+{R['spread']}%** spread. "
            "The hard-to-borrow tail underperformed — exactly what the borrow-fee premium predicts.\n\n"
            "**But is it just short interest wearing a fee costume?** Let's check whether the fee "
            "still matters once we hold short interest fixed:"
        ),
        code(
            "inc = st.incremental_regression(PANEL)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['borrow FEE','short INTEREST'], [inc['fee_t'], inc['si_t']],\n"
            "       color=[RED if abs(inc['fee_t'])>=2 else AMBER, GREY], width=.5)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('t-stat in the joint test')\n"
            "ax.set_title('Holding both at once: the FEE still predicts, short interest goes quiet')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fee t {inc[\"fee_t\"]:+.2f}  |  short-interest t {inc[\"si_t\"]:+.2f}')"
        ),
        md(
            f"When we let both compete, the **fee** keeps its punch (*t* {R['inc_fee_t']}) and short "
            f"interest goes to noise (*t* +{R['inc_si_t']}). So the fee — the *price* of shorting — "
            "carries something the *quantity* shorted doesn't. That's the whole point: it's a "
            "different, arguably better signal than the one everyone watches."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The effect is real *in our world* (spread +{R['spread']}%, *t* "
            f"+{R['ls_t']}) and survives short interest — but there's **no free real tape** to confirm "
            "it, and even here it's modest. Real data would be needed to earn anything stronger.\n"
            "- **Tradability — Fragile.** The alpha sits on the expensive-to-borrow short leg — the "
            "fee is both the signal *and* the rent. It survives at modest fees; the real hard-to-borrow "
            "tail would swamp it.\n"
            "- **Distinct from short interest? — Yes.** The fee predicts *beyond* how heavily a stock "
            "is shorted."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "This is the trap that makes the fee signal so hard to harvest. The trade is *long the "
            "cheap names, short the expensive ones* — but the 'expensive' names are literally the "
            "ones that cost the most to borrow. The very fee that tells you to short them is the bill "
            "you pay to do it. On our modest synthetic fees it still nets out positive; but a real "
            "hard-to-borrow name can cost 20%, 50%, 100%+ a year to short — and that eats a 12% edge "
            "for breakfast. The alpha and the cost are the *same variable*."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The neighbour.** [Study 262 — Short-Interest](../../262-short-interest/) sorts on the "
            "*quantity* shorted (short-% of float). This study sorts on the *price* (the fee) — and "
            "finds the price carries more.\n"
            "- **The missing piece is data.** With a real borrow-fee tape (Markit / S3 Partners) and a "
            "full cycle you could finally certify this on the actual market. Until then it's a "
            "well-supported *Weak*.\n\n"
            "*Have access to a real fee tape? Fork this, drop in the real panel, and show the "
            "cheap-minus-expensive spread clearing t = 2 on live data — that would flip Weak to Real.*"
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
            "# The Borrow-Fee Signal — a quantitative teardown 🔬\n"
            "### Fee-sorted quintiles · two-sample *t* · label-shuffle placebo · firm-level slope · incremental-over-short-interest OLS · bucket-width sweep · costs where the short leg pays its own borrow · seed-robust synthetic control + SI-only null\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Distinct_from_short_interest%3F: Yes](https://img.shields.io/badge/Distinct_from_short_interest%3F-Yes-2ea44f?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a synthetic "
            "borrow-fee cross-section (a general-collateral floor plus a special right tail), sort by "
            "fee, and test whether the expensive-to-borrow names earn the lowest returns — and whether "
            "the fee adds signal *beyond* raw short interest.\n\n"
            "> ⚠️ **Not investment advice, and synthetic-only by necessity.** No free historical "
            "borrow-fee tape exists (private data — Markit / IHS DataExplorers, S3 Partners), so the "
            f"whole study runs offline on the deterministic seed-{R['seed']} world (panel fp "
            f"`{R['fp_panel']}`). A synthetic-only study is **capped at WEAK/NONE** — REAL needs a "
            "robust *t* ≥ 2 on a real tape. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Cheap − expensive spread **+{R['spread']}%** (two-sample *t* "
            f"**+{R['ls_t']}**, placebo *p* {R['placebo_p']}); firm slope-*t* **{R['slope_t']}**; "
            "survives SI. **But synthetic-only** (no real tape → can't be REAL) and modest "
            "(seed-robust quintile *t* ≈ 1.5). |\n"
            f"| **Tradability** | `FRAGILE` | Alpha on the expensive short leg; gross +{R['spread']}% → "
            f"net +{R['net_q3m']}% (5 bps/leg + real {R['short_leg_fee']}% borrow, 3m). Real "
            "hard-to-borrow tail (20–100%+) would swamp it. |\n"
            f"| **Distinct from short interest?** | `YES` | Joint OLS: fee-*t* **{R['inc_fee_t']}** vs "
            f"SI-*t* **+{R['inc_si_t']}**. Fee carries the signal, SI doesn't. |\n\n"
            "> 💡 In plain words: the engine banks a real, modest, SI-orthogonal fee effect — but with "
            "no real data to certify it and the alpha stapled to the borrow cost, it's `WEAK` × "
            "`FRAGILE`."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $F_i$ be firm $i$'s standardized borrow fee, $S_i$ its standardized short interest, "
            "and $r_i$ its forward return.\n\n"
            "- **H₁ (the sort).** $\\mathbb{E}[r \\mid \\text{cheap}] - \\mathbb{E}[r \\mid "
            "\\text{expensive}] > 0$ at *t* ≥ 2 (cheap-to-borrow beats expensive).\n"
            "- **H₂ (firm-level).** slope of $r$ on $F < 0$ (higher fee, lower return).\n"
            "- **H₃ (incremental).** the fee slope in $r = a + bF + cS$ stays significant with $c$ "
            "(short interest) controlled — the fee is *not* laundered short interest.\n"
            "- **H₄ (robustness).** the sign of H₁ holds across bucket widths.\n"
            "- **H₅ (net).** H₁ survives costs — where the short leg pays *its own observed borrow*.\n\n"
            "On the synthetic tape we **fail to reject** H₁–H₄ (they hold) and find H₅ survives at "
            "modest fees — but the whole thing is capped `WEAK` for want of a real tape."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Two stocks with identical short interest can have wildly "
            "different borrow fees, because the fee also prices lendable **supply** (float, holder "
            "concentration, index membership) that the short-% quantity can't see. Cohen-Diether-"
            "Malloy (2007) argue the *price* (fee) is the informative side. If so, the fee should "
            "beat [Study 262](../../262-short-interest/)'s short-interest coin flip — and predict "
            "returns even after controlling for how heavily a name is shorted. That incrementality is "
            "H₃, the crux of the dedup."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $F = z(\\text{borrow fee})$ across the cross-section (higher = harder to "
            "borrow). A GC floor plus a demand/scarcity-driven special tail.\n"
            "- **Sort.** Quintile tails (24 names each): cheap (low $F$) vs expensive (high $F$).\n"
            "- **Inference.** Two-sample (Welch) *t* on cheap − expensive forward returns; a "
            "**label-shuffle placebo** null (2000 perms).\n"
            "- **Firm-level.** OLS of $r$ on $F$ — the *sign* of the slope is the claim.\n"
            "- **Incremental.** OLS of $r$ on $F$ **and** $S$ (short interest) jointly (H₃).\n"
            "- **Robustness.** Re-sort at four bucket widths; read the sign.\n"
            "- **Frictions.** Round-trip cost per leg + **the short leg's own observed borrow fee**, "
            "pro-rated over the hold (not a flat placeholder).\n"
            "- **Positive control.** A deterministic synthetic world with a planted effect "
            "(`fee_alpha < 0`) and a null — averaged over 25 seeds (house rule); plus a "
            "**short-interest-only null** proving the fee-sort doesn't fire on a pure-SI world.\n\n"
            "Timing: the fee and short interest are as-of the formation date (public that day); we "
            "enter at that close and hold over the forward window measured strictly after. No future "
            "data enters the signal (no look-ahead). The single execution lag — form on the snapshot, "
            "enter at that close — moves nothing material on a multi-month hold."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Cheap vs expensive forward returns, with the two-sample *t* and the placebo null."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2); ls = st.long_short_tstat(b)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=557)\n"
            "cheap_m, exp_m, spr, t = b['cheap_mean']*100, b['expensive_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['CHEAP','EXPENSIVE'], [cheap_m, exp_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Cheap - expensive = {spr:+.1f}%  (two-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cheap +{cheap_m:.1f}% | Expensive +{exp_m:.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: **H₁ holds.** The spread is **+{R['spread']}%** (*t* +{R['ls_t']}), "
            f"placebo *p* = {R['placebo_p']} — the expensive-to-borrow tail genuinely underperformed on "
            "this world. The claim's sign is present."
        ),
        md(
            "### 4b · The firm-level slope — H₂\n\n"
            "Regress forward return on the standardized fee across all 120 names. A *negative* slope "
            "is the claim."
        ),
        code(
            "reg = st.cross_section_regression(PANEL)\n"
            "d = st.fee_signal(PANEL); y = PANEL['forward_ret']*100\n"
            "slope, slope_t, corr = reg['slope']*100, reg['slope_t'], reg['corr']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(d, y, s=22, color=GREY)\n"
            "xs = np.linspace(d.min(), d.max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs + (y.mean()/100 - reg['slope']*d.mean()))*100, c=RED, lw=2)\n"
            "ax.set_xlabel('standardized borrow fee (higher = harder to borrow)'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Slope {slope:+.1f}%/sigma (t {slope_t:+.2f}) — NEGATIVE = the claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:+.1f}%/sigma, slope-t {slope_t:+.2f}, corr(fee, fwd) {corr:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **H₂ holds** — slope **{R['slope']}%** per fee-σ (*t* {R['slope_t']}), "
            f"corr {R['corr']}. More expensive to borrow, lower return."
        ),
        md(
            "### 4c · Incremental over short interest — H₃ (the dedup from Study 262)\n\n"
            "Regress forward return on **both** the fee and short interest. Does the fee survive?"
        ),
        code(
            "inc = st.incremental_regression(PANEL)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['borrow FEE','short INTEREST'], [inc['fee_t'], inc['si_t']],\n"
            "              color=[RED if abs(inc['fee_t'])>=2 else AMBER, GREY], width=.5)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='|t| = 2'); ax.axhline(2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('slope t-stat (joint OLS)')\n"
            "ax.set_title('The fee survives controlling for short interest; SI goes to noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'fee: slope {inc[\"fee_slope\"]*100:+.1f}%/sigma t {inc[\"fee_t\"]:+.2f}  |  '\n"
            "      f'SI: slope {inc[\"si_slope\"]*100:+.1f}%/sigma t {inc[\"si_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **H₃ holds** — in the joint test the fee keeps *t* {R['inc_fee_t']} "
            f"while short interest collapses to *t* +{R['inc_si_t']}. The fee carries information "
            "*beyond* the short quantity. This is the concrete distinction from "
            "[Study 262](../../262-short-interest/)."
        ),
        md(
            "### 4d · Robustness — H₄ (does the sign hold across cuts?)\n\n"
            "The same sort at four tail fractions."
        ),
        code(
            "rows = st.robustness_sweep(PANEL, fracs=(0.10, 0.20, 0.30, 0.40))\n"
            "tab = pd.DataFrame([(f'{f:.2f}', s*100, t) for f,s,t in rows],\n"
            "                   columns=['frac','spread%','welch_t']).set_index('frac')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(range(len(tab)), tab['spread%'], color=GREEN, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('tail fraction'); ax.set_ylabel('cheap - expensive spread %')\n"
            "ax.set_title('Positive and same-signed at every bucket width')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: **H₄ holds** — the spread is **positive at every cut** (+11% to +13%), "
            "strongest at the quintile/decile tails as the literature predicts. Not a single-cut "
            "artefact."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁–H₄ hold (spread +{R['spread']}%, *t* +{R['ls_t']}; firm slope-*t* "
            f"{R['slope_t']}; survives SI, *t* {R['inc_fee_t']}). But **synthetic-only** (no real tape "
            "→ REAL is off the table) and modest (seed-robust quintile *t* ≈ 1.5).\n"
            f"- **Tradability `FRAGILE`** — alpha on the expensive short leg; gross +{R['spread']}% → "
            f"net +{R['net_q3m']}% at modest fees, but the real hard-to-borrow tail would swamp it.\n"
            "- **Distinct from short interest? `YES`** — the fee predicts beyond the short quantity."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the borrow fee is the alpha AND the bill\n\n"
            "The unique honesty here: the short leg is the expensive-to-borrow tail, so we charge the "
            "*observed* fee, not a placeholder. Watch the net at different holds and at the deeper "
            "(costlier) decile tail."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.2)\n"
            "b10 = st.bucket_returns(PANEL, 0.1)\n"
            "gross = b['spread']*100\n"
            "net_3m = st.net_spread(b, cost_bps=5.0, holding_years=0.25)*100\n"
            "net_1y = st.net_spread(b, cost_bps=5.0, holding_years=1.0)*100\n"
            "net_d1y = st.net_spread(b10, cost_bps=5.0, holding_years=1.0)*100\n"
            "labels = ['gross', 'net 3m\\n(quintile)', 'net 1y\\n(quintile)', 'net 1y\\n(decile tail)']\n"
            "vals = [gross, net_3m, net_1y, net_d1y]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(labels, vals, color=[GREY, AMBER, AMBER, RED], width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('cheap - expensive spread %')\n"
            "ax.set_title('Borrow eats the edge as you hold longer / short deeper into the special tail')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'short-leg fee: quintile {b[\"short_leg_fee\"]:.1f}%, decile {b10[\"short_leg_fee\"]:.1f}%')\n"
            "print(f'gross {gross:+.1f}% | net 3m {net_3m:+.1f}% | net 1y {net_1y:+.1f}% | net 1y decile {net_d1y:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: at the *modest* synthetic fees the edge survives (net +{R['net_q3m']}% "
            f"quintile 3m; +{R['net_q1y']}% at 1y; +{R['net_d1y']}% at the decile tail). But the mean "
            f"short-leg fee is already {R['short_leg_fee']}% and rises into the tail — and real "
            "hard-to-borrow names run 20–100%+ a year, which *inverts* the trade. The signal and the "
            "cost are the same variable. `FRAGILE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control (+ the SI-only null)\n\n"
            "Is the engine a faithful detector, or does it always print a signal? Plant a real fee "
            "effect (`fee_alpha < 0`) of increasing strength and watch the firm-level slope-*t* go "
            "negative — averaged over 25 seeds so no lucky seed can fake it. Then plant the effect in "
            "**short interest instead** and confirm the fee-sort's *incremental* signal stays dead."
        ),
        code(
            "alphas = [0.0, -0.01, -0.02, -0.035, -0.05]\n"
            "st_t = [st.synthetic_mean_t(data, fee_alpha=a, n_seeds=25) for a in alphas]\n"
            "ls_t = [st.synthetic_mean_ls_t(data, fee_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, st_t, 'o-', c=GREEN, lw=2, label='firm slope-t')\n"
            "ax.plot(alphas, ls_t, 's--', c=AMBER, lw=2, label='long-short t')\n"
            "ax.axhline(-2, ls=':', c=GREY, lw=1); ax.axhline(2, ls=':', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted fee_alpha (more negative = stronger effect)')\n"
            "ax.set_ylabel('mean t-stat (25 seeds)')\n"
            "ax.set_title('The harness banks a planted fee effect — flat at the null, past |2| as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, s, l in zip(alphas, st_t, ls_t): print(f'fee_alpha {a:+.3f} -> slope-t {s:+.2f}, LS-t {l:+.2f}')"
        ),
        code(
            "# SI-only null: plant the effect in SHORT INTEREST (fee_alpha=0, si_alpha=-0.03).\n"
            "# The fee-sort's INCREMENTAL signal (joint regression) should die; SI's should carry it.\n"
            "psi, _ = data.synthetic_panel(fee_alpha=0.0, si_alpha=-0.03, seed=557)\n"
            "incsi = st.incremental_regression(psi)\n"
            "print(f'SI-only world -> incremental fee-t {incsi[\"fee_t\"]:+.2f} (dead), '\n"
            "      f'si-t {incsi[\"si_t\"]:+.2f} (carries it)')\n"
            "print('=> the fee-sort headline is NOT laundered short interest.')"
        ),
        md(
            f"The firm slope-*t* is ≈ 0 at the null and falls past −2 as the planted effect grows — a "
            f"faithful detector. At the headline `fee_alpha = {R['fee_alpha']}` the 25-seed mean slope-*t* "
            "is ≈ −2.1 (the single headline seed is a touch luckier), and the quintile LS-*t* averages "
            "only ≈ +1.5 — the effect is genuinely **modest**, which is exactly why the signal is "
            "`WEAK`. The SI-only null confirms the fee-sort reads the *fee*, not short interest. For the "
            "short-*quantity* cousin, see [Study 262 — Short-Interest](../../262-short-interest/)."
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
