"""Generate the two narrative notebooks for Study 574 (Penny-Beat).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a **synthetic-only**
study — there is no real consensus-vs-actual EPS tape a no-key stack can reach — so every code cell
runs the deterministic, offline synthetic world in ``penny_beat/data.py`` and re-derives its own
numbers; the frozen dict ``R`` below mirrors ``docs/results.md`` and is quoted in the prose. No cell
ever claims a real-tape number (there is none): ``HAVE_REAL`` is always ``False`` here.
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


# Frozen headline numbers — mirror of docs/results.md (synthetic, seed 574, n=6000,
# spike=0.55, penny_penalty=-0.05; panel fp 1ad98df82652; as-of 2026-06-30).
R = dict(
    fp_panel="1ad98df82652", fp_hist="3b43e570d8d5", n=6000,
    # discontinuity
    c_penny=988, neigh=381.5, excess=606, disc_z=17.7, disc_p=0.0005, deficit=0.48,
    # return test
    penny_mean=-1.07, decisive_mean=3.84, spread=-4.92, ret_t=-9.22, ret_p=0.0005,
    n_penny=988, n_decisive=2067,
    # decomposition
    comp_spread=-1.83, comp_t=-3.48, clean_penalty=-5.76, clean_t=-6.69,
    # costs
    net=4.47, cost_bps=5.0, borrow_bps=100.0,
    # robustness: (spike, disc_z, spread%, t)
    sweep=[(0.20, 8.0, -3.3, -5.23), (0.40, 13.9, -4.3, -7.52),
           (0.55, 17.7, -4.9, -9.22), (0.70, 21.3, -5.2, -10.30)],
    # synthetic control: (penny_penalty, mean_disc_z, mean_spread_t) over 25 seeds
    ctrl=[(0.00, 17.9, -3.7), (-0.02, 17.9, -6.0), (-0.05, 17.9, -9.3), (-0.10, 17.9, -14.5)],
    null_disc_z=0.3, null_spread_t=-2.3,
    # histogram slice -3..+4 for the bar
    hist={-3: 151, -2: 164, -1: 180, 0: 400, 1: 988, 2: 363, 3: 362, 4: 301},
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

from penny_beat import data, strategy as st

# Synthetic-only study: the reproducible core IS the deterministic synthetic world.
PANEL, TRUTH = data.synthetic_panel()   # seed 574, spike=0.55, penny_penalty=-0.05
HAVE_REAL = False                       # no free consensus-vs-actual EPS tape exists
print("synthetic penny-beat panel:", len(PANEL), "firm-quarters | HAVE_REAL:", HAVE_REAL)
print("panel fingerprint:", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Penny Beat — do firms that beat by a *cent* pay for it later? 🪙\n"
            "### Why clearing the analyst forecast by exactly one penny looks *managed*\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Discontinuity_real%3F: Confirmed](https://img.shields.io/badge/Discontinuity_real%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Every quarter, Wall Street analysts publish a *consensus* forecast for a company's "
            "earnings per share. Beat it and the stock usually pops; miss it and it drops. So there's "
            "a huge temptation to *just barely* clear the bar — to turn a would-be miss of a penny "
            "into a **beat by a penny**, with a little accounting nudge.\n\n"
            "If lots of firms do that, it leaves a fingerprint. Line up every company's *surprise* "
            "(actual EPS minus the forecast) and you'd expect a smooth bell curve. Instead there's a "
            "suspicious **spike right at +1¢** and a **dip just below zero** — mass shoved across the "
            "line. That's the **penny-beat discontinuity**. The follow-on claim: these penny-beaters, "
            "having borrowed from the future, *underperform* later.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *z*-scores, the placebo tests and the "
            "honest decomposition? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice, and this is a *synthetic* study.** The real data (analyst "
            "consensus vs actual EPS) is a licensed dataset a free retail stack can't reach — so we "
            "build a deterministic toy world with the effect dialled in, and test whether our engine "
            "*catches* it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the +1¢ spike real? | **Yes.** There's a huge excess of firms landing on exactly "
            f"+1¢ — **{R['c_penny']}** where a smooth curve expects ~{R['neigh']:.0f}, and the "
            "just-below-zero bins are half-empty. It's a smoking gun. |\n"
            f"| Do penny-beaters earn less later? | **Directionally yes** — they trail decisive "
            f"beaters by **{R['spread']}%**. |\n"
            f"| Is that a management penalty? | **Only about half of it.** Even with *zero* "
            f"manipulation dialled in, penny-beaters still trail by **{R['comp_spread']}%** — because "
            "a 1¢ beat is a *smaller* beat, and small beats just drift less. The rest is real "
            "post-earnings drift, not punishment. |\n"
            "| Could you trade it? | **No.** The data is licensed and unreachable, the penny bucket "
            "is tiny each quarter, and half the 'edge' is ordinary drift you could harvest directly. |"
            "\n\n> The spike is famous and robust. The *return penalty* is real in the textbooks too "
            "(Bhojraj et al. 2009) — but half of the naive version is just arithmetic, and we can't "
            "test it on a real tape for free. So: interesting, `Weak`, not tradable."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms that beat consensus EPS by exactly a penny managed the number — and pay for "
            "it with weaker future returns.\"* (Burgstahler & Dichev 1997; Degeorge-Patel-Zeckhauser "
            "1999; Bhojraj et al. 2009.)\n\n"
            "Two ideas stacked. **First**, the *discontinuity*: if you histogram earnings surprises "
            "across thousands of firm-quarters, a smooth distribution would glide through zero. The "
            "real one doesn't — it has a cliff just below zero and a spike just above, because firms "
            "manage small misses into small beats. **Second**, the *penalty*: a number that was "
            "propped up this quarter has to give it back, so the just-beaters underperform the firms "
            "that beat *decisively* and honestly."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "The discontinuity is one of accounting's cleanest pieces of evidence that earnings are "
            "*managed* — it shows up without you having to prove any single firm cheated. And if the "
            "return penalty were real *and* separable, it would be a short: fade the penny-beaters, "
            "own the decisive ones. The desk has already looked at two other manipulation tells — the "
            "[Beneish M-score (229)](../../229-beneish-m-score/) (an accounting-ratio screen) and "
            "[Benford's Law (328)](../../328-benford-law/) (a leading-digit screen). This one is the "
            "*shape of the surprise histogram*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the surprise histogram.** For every firm-quarter, surprise = actual − "
            "forecast, rounded to the nearest cent. Count how many land in each bin.\n"
            "2. **Look for the step.** Is the +1¢ bin far taller than its neighbours? Are the "
            "small-miss bins suspiciously empty? That's the management fingerprint.\n"
            "3. **Compare returns.** Do the penny-beaters (+1¢) earn less over the next quarter than "
            "the decisive beaters (+3¢ or more)?\n"
            "4. **Be honest about *why*.** A 1¢ beat is a *smaller* surprise than a 5¢ beat — and "
            "small surprises drift less anyway. So we check: how much of the gap survives if we turn "
            "*off* the manipulation entirely?\n\n"
            "Catch: we can't get the real forecast-vs-actual data for free (it's licensed), so we "
            "build a *synthetic* world where we control the truth, and test whether the engine reads "
            "it correctly."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The surprise histogram — is there a spike at +1¢?**"
        ),
        code(
            "h = data.surprise_histogram(PANEL)\n"
            "sl = h.loc[-6:6]\n"
            "cols = [RED if k == 1 else (AMBER if k in (-1,-2,-3) else GREY) for k in sl.index]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.bar(sl.index, sl.values, color=cols, width=.8)\n"
            "ax.axvline(0.5, c='k', ls='--', lw=1)\n"
            "ax.set_xlabel('earnings surprise (cents)'); ax.set_ylabel('number of firm-quarters')\n"
            "ax.set_title('The +1c spike (red) and the just-below-zero deficit (amber)')\n"
            "plt.tight_layout(); plt.show()\n"
            "dz = st.discontinuity_z(PANEL)\n"
            "print(f\"+1c bin: {dz['c_penny']:.0f}  vs smooth expectation ~{dz['neighbour_avg']:.0f}  \"\n"
            "      f\"-> excess {dz['excess']:+.0f} (z {dz['z']:+.1f})\")"
        ),
        md(
            f"There it is. The +1¢ bin holds **{R['c_penny']}** firm-quarters where a smooth curve "
            f"would put ~{R['neigh']:.0f} — an excess of **{R['excess']}** (a z-score of "
            f"**+{R['disc_z']}**). And the small-miss bins to the left are depleted: mass was pushed "
            "*up* across zero. This is the earnings-management fingerprint, and it's unmistakable."
        ),
        md(
            "**Do the penny-beaters earn less later — and is it really a *penalty*?**\n\n"
            "Compare penny-beaters (+1¢) to decisive beaters (≥+3¢), then re-run with the "
            "manipulation *turned off* to see how much of the gap is just ordinary drift."
        ),
        code(
            "b = st.penny_vs_decisive(PANEL)\n"
            "# composition-only world: same spike, but NO management penalty\n"
            "pc, _ = data.synthetic_panel(penny_penalty=0.0)\n"
            "bc = st.penny_vs_decisive(pc)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "labs = ['penny-beaters\\n(+1c)', 'decisive\\n(>=+3c)']\n"
            "ax.bar(labs, [b['penny_mean']*100, b['decisive_mean']*100], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f\"Penny-beaters {b['penny_mean']*100:+.1f}% vs decisive {b['decisive_mean']*100:+.1f}%  \"\n"
            "             f\"(spread {b['spread']*100:+.1f}%)\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"full spread   : {b['spread']*100:+.1f}%  (management + drift)\")\n"
            "print(f\"NO-manipulation spread: {bc['spread']*100:+.1f}%  (pure drift -- a 1c beat is a smaller beat)\")"
        ),
        md(
            f"Directionally the claim holds: penny-beaters trail decisive beaters by "
            f"**{R['spread']}%**. **But** flip the manipulation *off* and the gap is *still* "
            f"**{R['comp_spread']}%** — because a 1¢ beat is a smaller surprise than a 5¢ beat, and "
            "small beats drift less regardless. So roughly **half** the 'penalty' is ordinary "
            "post-earnings drift, not punishment for cheating. That distinction is why this reads "
            "`Weak`, not a slam-dunk."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The discontinuity is real and the penalty is directionally there, "
            "but half of it is mechanical drift, and there's no free real tape to confirm it on.\n"
            "- **Tradability — Mirage.** Unreachable data, a tiny penny bucket each quarter, and "
            "half the edge is drift you could harvest more cheaply and directly.\n"
            "- **Discontinuity real? — Confirmed.** The +1¢ spike / just-below-zero deficit is famous, "
            "robust, and the engine nails it (z +17.7)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not really. The trade would be *short the penny-beaters, long the decisive ones* — but "
            "you'd need licensed forecast-vs-actual data to even find them, the penny bucket is small "
            "and crowded, and once you strip out the ordinary drift there's not much *management* "
            "signal left to short. The **discontinuity** is a beautiful diagnostic of aggregate "
            "behaviour; it is not, by itself, a clean per-stock return edge."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other manipulation tells.** [Beneish M-score (229)](../../229-beneish-m-score/) "
            "and [Benford's Law (328)](../../328-benford-law/) attack the same question from the "
            "accounting ratios and the digits.\n"
            "- **The real test needs the real tape.** With licensed I/B/E/S consensus-vs-actual data, "
            "you could histogram real surprises and run the within-threshold (managed vs honest) "
            "return test that separates management from drift.\n\n"
            "*Think the penny-beat penalty is a clean short? Fork this, get point-in-time consensus "
            "data, and show a penny-minus-decisive spread that survives *after* you net out "
            "post-earnings drift.*"
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
            "# The Penny Beat — a quantitative teardown 🔬\n"
            "### surprise-histogram discontinuity *z* · smooth-relabel placebo · penny-vs-decisive two-sample *t* · PEAD-composition decomposition · within-bin clean penalty · robustness sweep · costs & borrow · seed-robust control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Discontinuity_real%3F: Confirmed](https://img.shields.io/badge/Discontinuity_real%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We histogram earnings "
            "surprises, measure the +$0.01 discontinuity, test the penny-vs-decisive return spread, "
            "and — the honest bit — decompose that spread into a **management penalty** and a "
            "**mechanical PEAD composition** gap.\n\n"
            "> ⚠️ **Not investment advice, and synthetic-only.** No free consensus-vs-actual EPS tape "
            f"exists for a no-key stack, so the reproducible core is the deterministic synthetic world "
            f"(seed 574, n={R['n']}, panel fp `{R['fp_panel']}`; as-of 2026-06-30). `HAVE_REAL` is "
            "always `False`; the Signal ceiling is `WEAK`. Methods in "
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
            f"| **Signal** | `WEAK` | Discontinuity z **+{R['disc_z']}** (placebo *p* {R['disc_p']}); "
            f"penny−decisive spread **{R['spread']}%** (*t* **{R['ret_t']}**) — but "
            f"composition-only spread **{R['comp_spread']}%** (*t* {R['comp_t']}) ⇒ ~half is PEAD; and "
            "no real tape. |\n"
            f"| **Tradability** | `MIRAGE` | Unreachable licensed data; tiny per-quarter penny bucket; "
            f"half the edge is harvestable drift. (Gross {R['spread']}% → net harvest {R['net']}% at "
            f"5 bps/leg + {R['borrow_bps']:.0f} bps borrow — but the trade isn't the problem, the data "
            "is.) |\n"
            f"| **Discontinuity real?** | `CONFIRMED` | +1c excess **+{R['excess']}** (z +{R['disc_z']}), "
            f"deficit ratio **{R['deficit']}**; flat at the null (z +{R['null_disc_z']} spike-off). |\n\n"
            "> 💡 In plain words: the +1c spike is unmistakable and the engine is faithful (the "
            "control proves it), but the *return* penalty is half mechanical drift and can't be "
            "checked on a real tape — so `WEAK`, not `REAL`."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_i$ be firm-quarter $i$'s rounded EPS surprise (cents) and $r_i$ its forward "
            "return.\n\n"
            "- **H₁ (discontinuity).** The count at $s=+1$ exceeds a local smooth of its neighbours "
            "at $z \\ge 2$ (an earnings-management spike).\n"
            "- **H₂ (return penalty).** $\\mathbb{E}[r \\mid s=+1] - \\mathbb{E}[r \\mid s\\ge+3] < 0$ "
            "at $t \\le -2$ (penny-beaters underperform decisive beaters).\n"
            "- **H₃ (separability).** The H₂ gap is a *management* penalty, **not** merely PEAD "
            "composition (i.e. it survives turning the manipulation knob to zero).\n"
            "- **H₄ (net / tradable).** H₂ survives costs and a real, reachable data source.\n\n"
            "On the synthetic tape we find **H₁ strongly supported** (z +17.7), **H₂ supported** "
            "(spread −4.9%, *t* −9.2), **H₃ only *partly*** (composition-only spread −1.8%, *t* −3.5 "
            "⇒ ~half the gap is PEAD, not management), and **H₄ fails** (no reachable data). That "
            "partial H₃ + failed H₄ is exactly what caps the Signal at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The **discontinuity** is a celebrated, robust piece of evidence that earnings are managed "
            "in aggregate (Burgstahler-Dichev 1997) — the desk *confirms* it. The **return penalty** "
            "(Bhojraj et al. 2009) is the tradable claim, and it lives or dies on **separability**: a "
            "+1c surprise is mechanically a *smaller* surprise than a ≥+3c decisive beat, so under any "
            "post-earnings-announcement drift (PEAD; Bernard-Thomas 1989) the penny bucket drifts less "
            "*even absent management*. Only a within-threshold test (managed vs honest firms that both "
            "landed on +1c) isolates the management penalty — and that needs a label a real analyst "
            "never sees. Cousins: [229 Beneish](../../229-beneish-m-score/), "
            "[328 Benford](../../328-benford-law/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Discontinuity.** $z = \\dfrac{c(+1) - \\tfrac12[c(0)+c(+2)]}{\\sqrt{c(+1) + \\tfrac14[c(0)+c(+2)]}}$"
            " — excess mass at +1c over a local smooth, Poisson-standardised. Plus a **smooth-relabel "
            "placebo** (redraw the local histogram uniformly, read the +1c z's tail).\n"
            "- **Return test.** Penny-beaters ($s=+1$) vs decisive beaters ($s\\ge+3$): a two-sample "
            "(Welch) *t* on forward returns + a **label-shuffle placebo**.\n"
            "- **Decomposition.** Re-run with `penny_penalty = 0` (spike still on) for the "
            "composition-only spread; and a **within-+1c-bin** managed-vs-honest test for the clean "
            "penalty (synthetic-truth diagnostic only).\n"
            "- **Frictions.** Round-trip cost per leg + a borrow on the (short) penny leg.\n"
            "- **Positive control.** Plant the effect and average the stats over 25 seeds (house "
            "rule); confirm the discontinuity z is flat at the null.\n\n"
            "Timing: the surprise is public at the earnings release; the position is entered at that "
            "release close and held over the forward window — one documented execution lag, no "
            "look-ahead. The sweep and the control use the identical convention."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The discontinuity — H₁\n\n"
            "The surprise histogram with the +1c excess and the smooth-relabel placebo."
        ),
        code(
            "h = data.surprise_histogram(PANEL); sl = h.loc[-6:6]\n"
            "dz = st.discontinuity_z(PANEL); dp = st.discontinuity_placebo_p(PANEL, n_perm=2000)\n"
            "cols = [RED if k == 1 else (AMBER if k in (-1,-2,-3) else GREY) for k in sl.index]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.bar(sl.index, sl.values, color=cols, width=.8)\n"
            "ax.axvline(0.5, c='k', ls='--', lw=1)\n"
            "ax.set_xlabel('surprise (cents)'); ax.set_ylabel('firm-quarters')\n"
            "ax.set_title(f\"Discontinuity z {dz['z']:+.1f} (placebo p {dp:.4f}) | +1c excess {dz['excess']:+.0f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"c(+1)={dz['c_penny']:.0f}  neigh~{dz['neighbour_avg']:.0f}  excess {dz['excess']:+.0f}  \"\n"
            "      f\"z {dz['z']:+.2f}  placebo p {dp:.4f}  deficit ratio {dz['deficit_ratio']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: **H₁ supported, decisively.** z = **+{R['disc_z']}**, placebo *p* = "
            f"{R['disc_p']} — the +1c spike is not something a smooth distribution produces. The "
            f"small-miss bins sit at {R['deficit']}× their mirror: mass moved *up* across zero."
        ),
        md(
            "### 4b · The return spread — H₂\n\n"
            "Penny-beaters vs decisive beaters, two-sample *t* + placebo."
        ),
        code(
            "b = st.penny_vs_decisive(PANEL); tt = st.spread_tstat(b)\n"
            "rp = st.return_placebo_p(PANEL, n_perm=2000)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['penny (+1c)','decisive (>=+3c)'], [b['penny_mean']*100, b['decisive_mean']*100],\n"
            "       color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f\"spread {b['spread']*100:+.1f}%  (Welch t {tt['t']:+.2f}, placebo p {rp:.4f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"penny {b['penny_mean']*100:+.2f}% (n={b['n_penny']}) | decisive {b['decisive_mean']*100:+.2f}% \"\n"
            "      f\"(n={b['n_decisive']}) | spread {b['spread']*100:+.2f}% | t {tt['t']:+.2f} | placebo p {rp:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: **H₂ supported** on its face — penny-beaters trail by "
            f"**{R['spread']}%** (*t* {R['ret_t']}, placebo *p* {R['ret_p']}). But *why*? That's 4c."
        ),
        md(
            "### 4c · Decomposition — H₃ (management vs mechanical drift)\n\n"
            "Turn the manipulation knob to **zero** (spike still on): how much spread is left? Then "
            "isolate the clean penalty *within* the +1c bin (managed vs honest — a synthetic-truth "
            "diagnostic)."
        ),
        code(
            "pc, _ = data.synthetic_panel(penny_penalty=0.0)     # composition-only world\n"
            "bc = st.penny_vs_decisive(pc); tc = st.spread_tstat(bc)\n"
            "wb = st.within_bin_penalty(PANEL)                   # clean penalty, uses latent truth\n"
            "rows = [('full spread\\n(mgmt + PEAD)', b['spread']*100, tt['t']),\n"
            "        ('composition-only\\n(penalty=0)', bc['spread']*100, tc['t']),\n"
            "        ('within-+1c clean\\n(managed-honest)', wb['penalty']*100, wb['t'])]\n"
            "labs = [r[0] for r in rows]; vals = [r[1] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(labs, vals, color=[RED, GREY, AMBER], width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('spread / penalty %')\n"
            "ax.set_title('Half the naive spread is mechanical PEAD, not management')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab, v, t in rows: print(f\"{lab.replace(chr(10),' '):32s} {v:+.2f}%  (t {t:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: **H₃ only partly holds.** With manipulation *off* the spread is "
            f"*still* **{R['comp_spread']}%** (*t* {R['comp_t']}) — pure PEAD, because a 1c beat is a "
            f"smaller surprise. The *clean* management penalty (within +1c, managed vs honest) is "
            f"**{R['clean_penalty']}%** (*t* {R['clean_t']}) — but that needs a label no real analyst "
            "sees. The naive penny-minus-decisive spread over-states management by roughly 2×."
        ),
        md(
            "### 4d · Robustness — does the sign hold across management intensities?\n\n"
            "Sweep the spike strength (holding the planted penalty fixed)."
        ),
        code(
            "sweep = st.robustness_sweep()\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(sweep.index, sweep['disc_z'], 'o-', c=GREEN, lw=2); a1.axhline(2, ls='--', c=GREY)\n"
            "a1.set_xlabel('spike (fraction managed up)'); a1.set_ylabel('discontinuity z'); a1.set_title('discontinuity grows with the spike')\n"
            "a2.plot(sweep.index, sweep['spread']*100, 'o-', c=RED, lw=2); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_xlabel('spike (fraction managed up)'); a2.set_ylabel('penny - decisive spread %'); a2.set_title('spread stays negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: both the discontinuity z and the (negative) spread grow monotonically "
            "with the management intensity — the sign never flips. Stable *given the planted world* "
            "(this is an engine statement, not a real tape)."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ strong (z +{R['disc_z']}), H₂ supported (spread {R['spread']}%, "
            f"*t* {R['ret_t']}), but H₃ only partial (composition-only spread {R['comp_spread']}%) and "
            "H₄ fails (no real tape). Synthetic-only ⇒ ceiling `WEAK`.\n"
            "- **Tradability `MIRAGE`** — unreachable data, tiny penny bucket, half the edge is "
            "harvestable drift.\n"
            f"- **Discontinuity real? `CONFIRMED`** — +1c excess +{R['excess']} (z +{R['disc_z']}), "
            f"flat at the null (z +{R['null_disc_z']})."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Short penny-beaters, long decisive beaters. Costs are a footnote against the raw "
            "magnitude — but the real blocker is the data and the PEAD confound, not the friction."
        ),
        code(
            "b = st.penny_vs_decisive(PANEL)\n"
            "gross = b['spread']*100\n"
            "net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=100.0, holding_years=0.25)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['|gross spread|','net harvest\\n(short penny, costs+borrow)'], [abs(gross), net],\n"
            "       color=[GREY, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('harvestable %')\n"
            "ax.set_title(f'raw magnitude {abs(gross):.1f}% -> net harvest {net:.1f}% (but the data is unreachable)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross spread {gross:+.2f}% | net harvest {net:+.2f}% (5 bps/leg + 100 bps borrow, 0.25y)')"
        ),
        md(
            "> 💡 In plain words: costs don't kill the *magnitude* — the data unavailability and the "
            "PEAD confound do. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a spike? Plant a real penalty "
            "of increasing strength (spike fixed) and watch the penny−decisive *t* deepen — averaged "
            "over 25 seeds — while checking the discontinuity z is **flat at the joint null**."
        ),
        code(
            "pens = [0.0, -0.02, -0.05, -0.10]\n"
            "ctrl = [st.synthetic_control(penny_penalty=p, spike=0.55, n_seeds=25) for p in pens]\n"
            "zs = [c['mean_disc_z'] for c in ctrl]; ts = [c['mean_spread_t'] for c in ctrl]\n"
            "nullc = st.synthetic_control(penny_penalty=0.0, spike=0.0, n_seeds=25)  # joint null\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(pens, ts, 'o-', c=RED, lw=2); a1.axhline(-2, ls='--', c=GREY, label='t = -2')\n"
            "a1.set_xlabel('planted penny_penalty'); a1.set_ylabel('mean penny-decisive t (25 seeds)')\n"
            "a1.set_title('penalty deepens the spread t'); a1.legend()\n"
            "a2.bar(['spike ON\\n(mgmt world)','spike OFF\\n(joint null)'], [zs[0], nullc['mean_disc_z']],\n"
            "       color=[GREEN, GREY], width=.5); a2.axhline(2, ls='--', c=GREY)\n"
            "a2.set_ylabel('mean discontinuity z (25 seeds)'); a2.set_title('discontinuity flat at the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "for p, c in zip(pens, ctrl): print(f'penalty {p:+.2f} -> disc z {c[\"mean_disc_z\"]:+.1f}, spread t {c[\"mean_spread_t\"]:+.2f}')\n"
            "print(f'JOINT NULL (spike 0, penalty 0) -> disc z {nullc[\"mean_disc_z\"]:+.1f}, spread t {nullc[\"mean_spread_t\"]:+.2f}')"
        ),
        md(
            f"The discontinuity z is **≈ 0 at the joint null** (z +{R['null_disc_z']}, spike off) and "
            f"jumps to ~{R['ctrl'][0][1]:.0f} once a spike is planted — a faithful, false-signal-free "
            f"detector. The penny−decisive *t* deepens monotonically with the planted penalty; note it "
            f"sits at ≈ {R['ctrl'][0][2]:.1f} even at the *return* null (penalty 0) because of the PEAD "
            "composition floor. So the engine works — the ceiling here is set by the missing real tape "
            "and the composition confound, not the code. Cousins: "
            "[229 Beneish](../../229-beneish-m-score/), [328 Benford](../../328-benford-law/)."
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
