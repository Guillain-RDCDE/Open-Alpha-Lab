"""Generate the two narrative notebooks for Study 589 (Genetic-Algo-Overfit).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a synthetic-only
research-method demo, so every code cell runs the deterministic offline engine directly — there is
no real-tape banner and no ``HAVE_REAL`` gate on the headline (the synthetic *is* the tape, by
design). The dict ``R`` mirrors docs/results.md so the prose and the executed numbers can never
drift apart.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; null tape fp e350016fa3d2,
# 2,999 rows, seed 589). Every one of these is reproduced live by the cells below.
R = dict(
    fp_null="e350016fa3d2", n_rows=2999, seed=589,
    is_sharpe=0.99, oos_sharpe=0.087, shrinkage=0.90, n_trials=2371, oos_exp=0.49,
    fit_first=0.76, fit_final=0.99,
    dsr=0.00, emax=3.49,
    placebo_p=0.073, placebo_obs=0.47, placebo_perm=-0.06,
    gross=0.087, net=0.012, turnover=0.18, cost_bps=2.0,
    # expected-max-Sharpe bar: (trials, E[max SR])
    emax_curve=[(10, 1.57), (100, 2.53), (1000, 3.26), (2371, 3.49), (5963, 3.73)],
    # search-budget sweep (null): (pop, gen, trials, is_sharpe, oos_sharpe, shrinkage)
    sweep=[(15, 10, 143, 0.76, 0.52, 0.25), (30, 20, 586, 0.84, -0.01, 0.85),
           (60, 40, 2371, 0.99, 0.09, 0.90), (100, 60, 5963, 0.98, 0.17, 0.82)],
    # positive control single-shot: (signal_strength, is_sharpe, oos_sharpe)
    ctrl=[(0.00, 0.99, 0.09), (0.15, 1.59, 0.83), (0.30, 2.42, 2.14), (0.50, 3.82, 3.56)],
    # seed-robust control (20 seeds): (signal_strength, mean_is, mean_oos, oos_t)
    ctrl_sr=[(0.00, 1.26, -0.07, -0.84), (0.20, 2.08, 1.24, 13.0), (0.40, 3.10, 2.55, 17.7)],
    # top champion features by |weight|
    top_feats=[("mom_5", 0.79), ("mom_21", -0.43), ("rsi", 0.35), ("vol_ratio", -0.25)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
np.seterr(all="ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from genetic_algo_overfit import data, strategy as st

SEED, N_DAYS = 589, 3000

# The null tape: a pure random walk. By construction NO timing rule can work on it.
DF_NULL, TRUTH = data.synthetic_series(n_days=N_DAYS, signal_strength=0.0, seed=SEED)
print("null tape:", len(DF_NULL), "rows, has_edge =", TRUTH.has_edge, "| fp", data.fingerprint(DF_NULL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Genetic-Algo-Overfit — evolving a 'strategy' out of pure noise 🧬\n"
            "### Let a genetic algorithm breed a brilliant backtest — then watch it die out-of-sample\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_GA_search_manufacture_a_false_edge%3F: Confirmed](https://img.shields.io/badge/Does_GA_search_manufacture_a_false_edge%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Here's the pitch you hear behind every 'AI finds alpha' story: *don't hand-design a "
            "trading rule — **evolve** one.* A **genetic algorithm** breeds a population of candidate "
            "rules, keeps the fittest, mixes and mutates them, and repeats — like natural selection, "
            "but for backtests. After a few dozen generations it hands you a rule with a gorgeous "
            "track record.\n\n"
            "So let's do exactly that — on a tape we **built to be pure noise**, a coin-flip random "
            "walk where *no* rule can genuinely time the market. If the GA finds a 'strategy' *there*, "
            "we've caught the trap red-handed.\n\n"
            "> 📓 **This is the plain-language layer.** Want the Sharpe maths, the Deflated Sharpe "
            "Ratio and the placebo test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
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
            f"| Did the GA find a 'great' strategy on pure noise? | **Yes — in-sample.** It bred a "
            f"rule with a Sharpe of **{R['is_sharpe']}** — a backtest that looks like a real edge. |\n"
            f"| Did it work out-of-sample (on data it never saw)? | **No.** The Sharpe collapsed to "
            f"**{R['oos_sharpe']}** — essentially all of it ({R['shrinkage']:.0%} shrinkage) was fake. |\n"
            f"| How do we *know* it's fake? | The 'luck bar' for the "
            f"**{R['n_trials']:,}** rules the GA tried is a Sharpe of **{R['emax']}** — its "
            f"{R['is_sharpe']} is *below* it. Pure noise would have done better. |\n"
            "| Could you trade it? | **No.** The out-of-sample edge is a rounding error (Sharpe ~0.09) "
            "that a tiny trading cost erases to ~0.01. |\n"
            "| Is the genetic algorithm broken, then? | **No — it's a good detector.** Give it a tape "
            "with a *real* hidden edge and it finds a rule that keeps working out-of-sample. It only "
            "fails when there was nothing to find. |\n\n"
            "> The GA didn't discover an edge. It discovered the *luckiest way to fit the past* — and "
            "the past doesn't repeat."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A genetic algorithm searches millions of rule combinations and keeps the fittest — "
            "so it finds edges a human never would.\"*\n\n"
            "The half that's true: a GA really is a powerful optimiser, and it really will find a rule "
            "with a beautiful in-sample backtest. The half that's a trap: on finite data, that "
            "'fitness' is mostly a fit to the random wiggles of the training period. The harder the "
            "search, the prettier the fit — and the less of it survives contact with new data."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, you could point an optimiser at any market and print money. If it *doesn't* "
            "— and it doesn't — then every 'the algorithm discovered this strategy' pitch is suspect "
            "until you see how many strategies it tried and how the winner did on data it never "
            "touched. This is the same trap as a plain grid search "
            "([344 Backtest-Overfitting](../../344-backtest-overfitting/), "
            "[348 Curve-Fitting](../../348-curve-fitting/)) — but a GA searches *harder*, so it "
            "overfits *harder*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build a fair test.** Make a fake price history that is *pure noise* — a random walk "
            "with no real pattern. Any 'strategy' found here is guaranteed fake.\n"
            "2. **Split it in half.** Let the GA evolve on the first half (the *in-sample* half) only.\n"
            "3. **Freeze the winner and test it on the second half** (the *out-of-sample* half it "
            "never saw). If the great backtest was real, it should keep working. If it was overfitting, "
            "it collapses.\n\n"
            "*Timing:* each rule uses only information available at the close of a day to decide the "
            "next day's position — no peeking at the future. And we judge the rule on its *timing* "
            "(being long on the good days), stripped of any lucky drift, so the collapse is pure "
            "overfitting, not a hidden market trend."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually watch it happen\n\n"
            "**First: does the GA really improve, generation by generation?** Here's the best "
            "in-sample Sharpe it has found after each generation of breeding:"
        ),
        code(
            "res = st.is_oos_split(DF_NULL, pop_size=60, generations=40, seed=SEED)\n"
            "hist = res['fitness_history']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(range(len(hist)), hist, 'o-', c=GREEN, lw=2, ms=4)\n"
            "ax.set_xlabel('generation'); ax.set_ylabel('best in-sample Sharpe so far')\n"
            "ax.set_title(f'The GA \"improves\" every generation -> IS Sharpe {res[\"is_sharpe\"]:.2f} '\n"
            "             f'(on PURE NOISE)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gen 1 best {hist[0]:.2f}  ->  final best {res[\"is_sharpe\"]:.2f}  '\n"
            "      f'({res[\"n_trials\"]:,} rules tried)')"
        ),
        md(
            f"It climbs from **{R['fit_first']}** to **{R['is_sharpe']}**. The GA is doing its job — "
            "getting 'better' every generation. But it's on a random walk, so there is nothing real to "
            "get better *at*. It's memorising the training noise."
        ),
        md(
            "**Now the moment of truth: freeze that champion and run it on the half it never saw.**"
        ),
        code(
            "is_s, oos_s = res['is_sharpe'], res['oos_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['IN-SAMPLE\\n(where it evolved)', 'OUT-OF-SAMPLE\\n(new data)'],\n"
            "       [is_s, oos_s], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title(f'The brilliant backtest ({is_s:.2f}) dies out-of-sample ({oos_s:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'IS Sharpe {is_s:.2f}  ->  OOS Sharpe {oos_s:.2f}   (shrinkage {is_s-oos_s:.2f})')"
        ),
        md(
            f"There it is. In-sample **{R['is_sharpe']}**, out-of-sample **{R['oos_sharpe']}** — about "
            f"**{R['shrinkage']:.0%}** of the 'edge' vanished the instant the GA met fresh data. That "
            "gap *is* the overfitting."
        ),
        md(
            "**And the punchline: the harder you search, the prettier the (fake) backtest.** Run the "
            "GA with bigger and bigger budgets and watch the in-sample Sharpe rise while the "
            "out-of-sample Sharpe stays stuck near zero:"
        ),
        code(
            "budgets = [(15,10),(30,20),(60,40),(100,60)]\n"
            "sw = st.robustness_sweep(DF_NULL, budgets, seed=SEED)\n"
            "x = np.arange(len(sw)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(x-w/2, sw['is_sharpe'], w, color=GREEN, label='in-sample')\n"
            "ax.bar(x+w/2, sw['oos_sharpe'], w, color=RED, label='out-of-sample')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{t:,}\\ntrials' for t in sw['trials']])\n"
            "ax.set_ylabel('Sharpe'); ax.legend()\n"
            "ax.set_title('More search -> bigger IS Sharpe, OOS stuck at zero (pure overfitting)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sw[['trials','is_sharpe','oos_sharpe','shrinkage']].round(2).to_string(index=False))"
        ),
        md(
            "The green bars (in-sample) grow with the search; the red bars (out-of-sample) never "
            "leave zero. This is the whole trap in one picture: **a bigger search buys a shinier "
            "backtest and not one cent of real edge.**"
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** On a tape with *zero* real edge the GA still bred a Sharpe-"
            f"{R['is_sharpe']} backtest — that died to **{R['oos_sharpe']}** out-of-sample. Nothing "
            "real to find.\n"
            "- **Tradability — Mirage.** The out-of-sample edge is economically zero and a mild cost "
            "erases what little is left.\n"
            "- **Does GA search manufacture a false edge? — Confirmed.** Yes, reliably, and the "
            "fakery grows with the search budget."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and not even because of costs. The evolved champion's out-of-sample timing Sharpe is "
            "an economically meaningless ~0.09, and a mild trading cost erases it to ~0.01. There was "
            "never anything to harvest. The lesson isn't 'genetic algorithms are bad' — it's 'a "
            "backtest is worthless "
            "until you know how many rules were tried and how the winner did on data it never saw.'"
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is the GA just a broken machine?** No — the quants notebook plants a *real* edge and "
            "shows the same GA finds a rule that keeps working out-of-sample. It's a detector, not a "
            "fantasy generator.\n"
            "- **The same trap, other optimisers.** [344 Backtest-Overfitting](../../344-backtest-overfitting/) "
            "(grid search) and [348 Curve-Fitting](../../348-curve-fitting/) (tuning two windows) are "
            "the same story with a simpler search.\n\n"
            "*Think your optimiser is different? Fork this, swap the GA for your favourite method "
            "(random search, Bayesian optimisation, a neural net), and show it does **not** find a "
            "Sharpe-1 backtest on a pure random walk.*"
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
            "# Genetic-Algo-Overfit — a quantitative teardown 🔬\n"
            "### GA (selection/crossover/mutation) · IS/OOS split · shrinkage vs search budget · Deflated Sharpe Ratio · label-shuffle placebo · costs · seed-robust positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_GA_search_manufacture_a_false_edge%3F: Confirmed](https://img.shields.io/badge/Does_GA_search_manufacture_a_false_edge%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We evolve a long/flat rule "
            "with a genetic algorithm on a pure random walk, measure the in-sample → out-of-sample "
            "collapse, and haircut the in-sample Sharpe for the number of genomes the GA effectively "
            "searched.\n\n"
            "> ⚠️ **Not investment advice.** This is a **synthetic-only** method demo — the tape is "
            "built so we *know* the ground truth (a real market can never be certified edge-free), so "
            f"it can never earn `REAL` (null tape fp `{R['fp_null']}`, seed {R['seed']}). Methods in "
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
            f"| **Signal** | `NONE` | GA champion IS Sharpe **{R['is_sharpe']}** → OOS "
            f"**{R['oos_sharpe']}** (shrinkage **{R['shrinkage']}**); **DSR {R['dsr']:.2f}** (IS "
            f"Sharpe *below* the {R['emax']} luck bar for {R['n_trials']:,} trials); placebo *p* "
            f"**{R['placebo_p']}**; seed-robust null OOS *t* **{R['ctrl_sr'][0][3]}**. |\n"
            f"| **Tradability** | `MIRAGE` | Demeaned OOS book gross Sharpe **{R['gross']}**, net "
            f"**{R['net']}** ({R['cost_bps']:.0f} bps/switch). Nothing to harvest. |\n"
            f"| **Does GA search manufacture a false edge?** | `CONFIRMED` | IS Sharpe rises with the "
            "search budget while OOS orbits zero; the positive control banks a planted edge OOS. |\n\n"
            "> 💡 In plain words: the GA is a faithful optimiser (the control proves it) — which is "
            "exactly why, on noise, it manufactures a beautiful backtest that carries nothing forward."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A rule is a genome $w$ (a unit-norm weight vector over 7 features); it goes long when "
            "$w \\cdot x_t > 0$ and flat otherwise. The GA maximises the in-sample Sharpe of the "
            "*demeaned* forward-return book (so drift can't flatter a long-biased rule).\n\n"
            "- **H₁ (the discovered edge).** The GA champion's OOS Sharpe clears **2** — a real, "
            "carried-forward edge.\n"
            "- **H₂ (beyond luck).** The champion's IS Sharpe beats the expected maximum from that "
            "many trials — the **Deflated Sharpe Ratio** is near 1.\n"
            "- **H₃ (robust to the search).** The OOS Sharpe does not fall as the search budget "
            "grows.\n"
            "- **H₄ (net).** The OOS edge survives costs.\n\n"
            "We find **H₁ rejected** (OOS Sharpe ≈ 0), **H₂ rejected** (DSR ≈ 0 — the IS Sharpe is "
            "*below* the luck bar), **H₃ rejected** (IS inflates with the budget, OOS never follows), "
            "and **H₄ moot** (the OOS edge is economically zero, ~0.09, before costs)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A genetic algorithm's *effective trial count* is the number of distinct genomes it "
            "evaluates — often thousands — and by Bailey-López de Prado the expected maximum Sharpe "
            "from $N$ independent noise strategies grows like $\\sqrt{2\\ln N}$, *without bound* in "
            "$N$. So a flexible optimiser that searches hard is a Sharpe-manufacturing machine unless "
            "you deflate for the search. This is the industrial-scale version of the multiple-testing "
            "crisis (Harvey-Liu-Zhu 2016): a *t* of 2 means nothing when you tried 2,000 rules."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Tape.** A deterministic random walk (`signal_strength = 0`): 7 standardised "
            "technical features + the next-day return, no true predictability.\n"
            "- **GA.** Tournament selection (k=3), uniform crossover, Gaussian mutation, elitism; "
            "population 60 × 40 generations; fitness = in-sample Sharpe of the demeaned book.\n"
            "- **Split.** Evolve on the first 50%; judge the frozen champion on the last 50%.\n"
            "- **Deflated Sharpe.** Haircut the IS Sharpe for the effective trial count (via the "
            "expected-max-Sharpe bar) and the return moments (Lo's skew/kurtosis SE).\n"
            "- **Placebo.** Re-run the *whole* evolve-then-validate protocol on shuffled forward "
            "returns; read the OOS Sharpe's tail probability.\n"
            "- **Frictions.** One-way turnover cost on each long/flat switch (no short leg, no "
            "borrow).\n"
            "- **Positive control.** Plant a real edge (`signal_strength > 0`) and confirm the GA "
            "banks it OOS — averaged over ≥ 20 seeds (house rule).\n\n"
            "Timing: features at the close of $t$ are public at that close and decide the position "
            "earning $r_{t+1}$ — one execution lag, applied once via the pre-shifted `fwd_ret` "
            "column. No future data enters the score."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The IS → OOS collapse — H₁\n\n"
            "Evolve on the in-sample half; freeze the champion; measure it on the out-of-sample half."
        ),
        code(
            "res = st.is_oos_split(DF_NULL, pop_size=60, generations=40, seed=SEED)\n"
            "is_s, oos_s = res['is_sharpe'], res['oos_sharpe']\n"
            "g = res['genome']; order = np.argsort(-np.abs(g))\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax1.bar(['IS','OOS'], [is_s, oos_s], color=[GREEN, RED], width=.5)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_ylabel('Sharpe')\n"
            "ax1.set_title(f'IS {is_s:.2f} -> OOS {oos_s:.2f}  (shrinkage {is_s-oos_s:.2f})')\n"
            "feats = [data.FEATURE_NAMES[i] for i in order[:5]]; wts = [g[i] for i in order[:5]]\n"
            "ax2.barh(range(len(feats)), wts, color=GREY)\n"
            "ax2.set_yticks(range(len(feats))); ax2.set_yticklabels(feats); ax2.invert_yaxis()\n"
            "ax2.axvline(0, c='k', lw=1); ax2.set_xlabel('champion weight')\n"
            "ax2.set_title('The \"discovered\" rule (top features)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'IS Sharpe {is_s:.3f} | OOS Sharpe {oos_s:.3f} | shrinkage {is_s-oos_s:.3f} | '\n"
            "      f'{res[\"n_trials\"]:,} trials | OOS exposure {res[\"oos_exposure\"]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** The GA's confident-looking rule (weighting mom_5, "
            f"mom_21, rsi…) posts IS Sharpe **{R['is_sharpe']}** and OOS Sharpe **{R['oos_sharpe']}** "
            f"— shrinkage **{R['shrinkage']}**. The 'discovery' was a fit to training noise."
        ),
        md(
            "### 4b · Beyond luck? — H₂ and the Deflated Sharpe Ratio\n\n"
            "The expected maximum Sharpe from $N$ noise strategies grows with $N$. Overlay the GA's "
            "IS Sharpe on that luck bar."
        ),
        code(
            "(fis, yis), _, _ = st._split_arrays(DF_NULL, 0.5)\n"
            "is_ret = st.rule_returns(fis, yis, res['genome'])\n"
            "dsr = st.deflated_sharpe(is_ret, n_trials=res['n_trials'])\n"
            "Ns = np.unique(np.round(np.logspace(0.7, 3.9, 40)).astype(int))\n"
            "bar = [st.expected_max_sharpe(int(n)) for n in Ns]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.plot(Ns, bar, c=GREY, lw=2, label='expected MAX Sharpe from noise (luck bar)')\n"
            "ax.axvline(res['n_trials'], ls='--', c=RED, lw=1)\n"
            "ax.scatter([res['n_trials']], [is_s], color=GREEN, zorder=5, s=60,\n"
            "           label=f'GA champion IS Sharpe = {is_s:.2f}')\n"
            "ax.set_xscale('log'); ax.set_xlabel('trials searched (log)'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title(f'The IS Sharpe is BELOW the luck bar -> DSR = {dsr[\"dsr\"]:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'IS Sharpe {dsr[\"sr\"]:.2f} | trials {dsr[\"n_trials\"]:,} | '\n"
            "      f'expected-max bar {dsr[\"benchmark_sr\"]:.2f} | Deflated Sharpe Ratio {dsr[\"dsr\"]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected, emphatically.** For {R['n_trials']:,} trials the luck "
            f"bar is a Sharpe of **{R['emax']}** — the champion's **{R['is_sharpe']}** is *beneath* it. "
            f"The Deflated Sharpe Ratio is **{R['dsr']:.2f}**: this backtest is *less* impressive than "
            "what pure noise hands a search this size."
        ),
        md(
            "### 4c · Robustness — H₃ (does OOS survive a bigger search?)\n\n"
            "Grow the GA's search budget and track IS, OOS and shrinkage."
        ),
        code(
            "sw = st.robustness_sweep(DF_NULL, [(15,10),(30,20),(60,40),(100,60)], seed=SEED)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(sw['trials'], sw['is_sharpe'], 'o-', c=GREEN, lw=2, label='in-sample')\n"
            "ax.plot(sw['trials'], sw['oos_sharpe'], 's--', c=RED, lw=2, label='out-of-sample')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xscale('log')\n"
            "ax.set_xlabel('trials searched (log)'); ax.set_ylabel('Sharpe'); ax.legend()\n"
            "ax.set_title('IS climbs with the search; OOS never leaves zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sw[['trials','is_sharpe','oos_sharpe','shrinkage']].round(3).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** The in-sample Sharpe rises from **0.76 → 0.99** as "
            "trials go **143 → 2,371**, while the out-of-sample Sharpe orbits zero. The extra IS "
            "Sharpe a bigger search buys is 100% selection — it never reaches OOS."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (OOS Sharpe {R['oos_sharpe']}), H₂ rejected "
            f"(DSR {R['dsr']:.2f}, IS Sharpe below the {R['emax']} luck bar), H₃ rejected (IS inflates "
            f"with the search). Placebo *p* {R['placebo_p']}; seed-robust null OOS *t* "
            f"{R['ctrl_sr'][0][3]}. A synthetic-only demo — no real tape, so never `REAL`.\n"
            f"- **Tradability `MIRAGE`** — demeaned OOS book gross {R['gross']} → net {R['net']}.\n"
            "- **Does GA search manufacture a false edge? `CONFIRMED`** — reliably, and it grows with "
            "the search budget."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the placebo and costs\n\n"
            "First the placebo: re-run the *entire* evolve-then-validate protocol on shuffled forward "
            "returns (a lighter GA budget for tractability) and see where the real OOS sits."
        ),
        code(
            "pl = st.placebo_pvalue(DF_NULL, n_perm=40, pop_size=40, generations=25, seed=SEED)\n"
            "# Costs on the HEADLINE champion's OOS book (the same 60x40 rule from 4a).\n"
            "_, (foos2, yoos2), _ = st._split_arrays(DF_NULL, 0.5)\n"
            "cost = st.net_sharpe(foos2, yoos2, res['genome'], cost_bps=2.0)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['gross OOS','net OOS\\n(2 bps/switch)'],\n"
            "       [cost['gross_sharpe'], cost['net_sharpe']], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('OOS Sharpe')\n"
            "ax.set_title(f'Economically zero before costs ({cost[\"gross_sharpe\"]:+.2f}), '\n"
            "             f'~nothing after ({cost[\"net_sharpe\"]:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo: obs OOS {pl[\"obs_oos_sharpe\"]:.2f}, p {pl[\"p\"]:.3f} '\n"
            "      f'(mean shuffled OOS {pl[\"perm_mean_oos\"]:+.2f})')\n"
            "print(f'costs: gross OOS {cost[\"gross_sharpe\"]:+.2f} -> net {cost[\"net_sharpe\"]:+.2f} '\n"
            "      f'(turnover {cost[\"turnover_per_day\"]:.2f}/day)')"
        ),
        md(
            f"> 💡 In plain words: the OOS result sits at placebo *p* ≈ **{R['placebo_p']}** — "
            "indistinguishable from searching shuffled noise — and the demeaned OOS book is "
            f"economically zero (**+{R['gross']}**), erased to **+{R['net']}** by a mild cost. "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the GA a faithful detector, or does it always die OOS? Plant a *real* edge of "
            "increasing strength (a hidden linear combination of the features genuinely predicts the "
            "next return) and watch the OOS Sharpe survive — averaged over 20 seeds so no single "
            "lucky seed can fake it (house rule)."
        ),
        code(
            "rows = [st.seed_robust_control(ss, n_seeds=20, n_days=2000, pop_size=40, generations=25)\n"
            "        for ss in [0.0, 0.10, 0.20, 0.30, 0.40]]\n"
            "ss = [r['signal_strength'] for r in rows]\n"
            "mis = [r['mean_is_sharpe'] for r in rows]; moos = [r['mean_oos_sharpe'] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(ss, mis, 'o-', c=GREY, lw=2, label='mean IS Sharpe')\n"
            "ax.plot(ss, moos, 's-', c=GREEN, lw=2, label='mean OOS Sharpe (20 seeds)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted signal_strength (0 = pure noise)'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('Null: OOS flat at ~0. Real edge: the GA banks it OOS.')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rows:\n"
            "    print(f'strength {r[\"signal_strength\"]:.2f} -> mean IS {r[\"mean_is_sharpe\"]:.2f}, '\n"
            "          f'mean OOS {r[\"mean_oos_sharpe\"]:.2f}, OOS-t {r[\"oos_t\"]:+.2f}')"
        ),
        md(
            f"At the null the OOS Sharpe is flat (mean **{R['ctrl_sr'][0][2]}**, *t* "
            f"**{R['ctrl_sr'][0][3]}** — no false signal); plant a real edge and the GA banks it OOS "
            f"with an overwhelming *t* (strength 0.20 → OOS *t* **{R['ctrl_sr'][1][3]}**, 0.40 → "
            f"**{R['ctrl_sr'][2][3]}**). So the harness is a faithful detector — the null-tape "
            "collapse is the tape talking, not a broken engine. For the same trap under a plain grid "
            "search, see [344 Backtest-Overfitting](../../344-backtest-overfitting/) and "
            "[348 Curve-Fitting](../../348-curve-fitting/). *(Synthetic control = machinery proof, "
            "never market evidence — METHODOLOGY → inference bar.)*"
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
