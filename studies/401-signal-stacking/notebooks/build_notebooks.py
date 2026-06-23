"""Generate the two narrative notebooks for Study 401 (Signal-Stacking).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The synthetic null and positive control run
anywhere with no network; the real-tape cells read the cached SPY stack under ../_cache/ and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
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


# Frozen headline numbers — mirror of docs/results.md.
R = dict(
    # real SPY tape
    real_start="1995-10-16", real_end="2026-06-18", real_days=7719, real_years=30.7,
    n_signals_real=12, fingerprint="a4bfd3d8a8cd",
    # (A) null: ic=0, corr=0, 10 signals
    null=dict(comp_sh=0.11, comp_t=0.36, best=0.63, mean=0.14,
              comp_ic=0.0122, mean_ic=0.0147, lift=0.83, sqrtk=3.16, perm_p=0.375,
              is_sh=0.48, oos_sh=0.36, n_chosen=6),
    # (B) positive control: ic=0.05, corr=0, 10 signals
    pos=dict(comp_sh=2.00, comp_t=6.40, best=1.21, mean=0.71,
             comp_ic=0.1644, mean_ic=0.0535, lift=3.07, sqrtk=3.16, perm_p=0.000,
             is_sh=2.12, oos_sh=1.75, n_chosen=10),
    # (C) redundant: ic=0.05, corr=0.9, 10 signals
    red=dict(comp_sh=0.52, comp_t=1.67, best=0.72, mean=0.46,
             comp_ic=0.0524, mean_ic=0.0500, lift=1.05, sqrtk=3.16, perm_p=0.051),
    # (D) sqrt(K) curve, decorrelated: (K, lift)
    curve_decorr=[(1, 1.00), (2, 1.39), (5, 2.21), (10, 3.07), (20, 4.35), (40, 5.97)],
    # (E) sqrt(K) curve, redundant: (K, lift)
    curve_redund=[(1, 1.00), (2, 1.03), (5, 1.04), (10, 1.05), (20, 1.05), (40, 1.05)],
    sqrtk_ref=[(1, 1.00), (2, 1.41), (5, 2.24), (10, 3.16), (20, 4.47), (40, 6.32)],
    # (F) real SPY stack
    real=dict(comp_sh=-0.26, comp_t=-1.63, best=0.20, mean=-0.14,
              comp_ic=-0.0666, mean_ic=0.0360, perm_p=0.982, bh_sh=0.62,
              snoop_is=-0.05, snoop_oos=-0.46, n_chosen=4,
              chosen=["noise_a", "ma_gap_200", "noise_c", "mom_60"]),
    # (G) snoop tax
    snoop=[("positive control (ic=0.05)", 2.12, 1.75, 10),
           ("null (ic=0)", 0.48, 0.36, 6),
           ("real SPY tape", -0.05, -0.46, 4)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![A_composite_edge%3F: Busted](https://img.shields.io/badge/A_composite_edge%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from signal_stacking import data, strategy as st

def have_real():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = have_real()
SIG = FWD = None
if HAVE_REAL:
    SIG, FWD = data.load_real()
print("real SPY stack cache present:", HAVE_REAL,
      "| signals:", (0 if SIG is None else SIG.shape[1]),
      "| days:", (0 if SIG is None else len(SIG)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Stack fifty weak signals into a secret edge? — Signal-Stacking 🧮\n"
            "### The viral 'combine coin-flips into a composite field retail has never seen,' in plain English\n\n"
            + BADGES +
            "A thread goes viral every few months: *\"Take 50 signals — none of them works alone, "
            "each barely better than a coin flip. Normalise them into one composite score, weight "
            "each by how well it did historically, and you get an edge the retail crowd has never "
            "seen.\"* It comes with a gorgeous equity curve labelled with Greek letters (the "
            "\"ζ-field\") soaring past buy-and-hold.\n\n"
            "Here's the honest answer, and it has a real half and a fatal half. Combining weak "
            "signals **does** help — but only when the signals are **real** *and* **different** from "
            "each other, and even then only by about **√K** (the square root of how many you stack). "
            "On pure noise it does nothing. On fifty copies of the same idea it barely beats one. And "
            "that beautiful curve? It's drawn by *cherry-picking* which signals to keep — and it "
            "falls apart the moment you leave the data you mined it from.\n\n"
            "> 📓 **Plain-language layer.** Want the √K law, the permutation test and the snooping "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A note up front.** This is a **methods demo** (a cousin of "
            "[Studies 343–350](../../343-data-mining-roulette/) and "
            "[399](../../399-kalshi-efficiency/)). We prove the lesson on a controlled lab panel "
            "where we *know* the truth, then show the same machine on a real SPY tape. No secret "
            "edge is being sold. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I stack 10 pure-noise signals, do I get an edge? | **No.** The composite's Sharpe "
            f"is **{R['null']['comp_sh']:.2f}** — pure luck. A coin-flip reshuffle beats it "
            f"**{R['null']['perm_p']*100:.0f}%** of the time. |\n"
            "| If the signals are real *and* different? | **Yes — that's the real half.** With a "
            "genuine little edge in each, the composite Sharpe jumps to "
            f"**{R['pos']['comp_sh']:.2f}**, beating even the best single signal. |\n"
            "| If the signals are real but *redundant* (copies of one idea)? | **It collapses.** "
            f"Same edge, but correlated — composite Sharpe falls to **{R['red']['comp_sh']:.2f}**. "
            "Fifty copies of one signal are worth barely more than one. |\n"
            "| And on a real market (SPY)? | **It loses.** Stacking 12 textbook timing signals gives "
            f"a **negative** composite (Sharpe **{R['real']['comp_sh']:.2f}**) — just holding SPY "
            f"(**+{R['real']['bh_sh']:.2f}**) beats it. |\n"
            "| Then where does the magic curve come from? | **Cherry-picking.** Choose the "
            "best-looking signals *after* seeing the data and the curve looks great — then it dies "
            "out of sample. |\n\n"
            "> Stacking is real maths, not magic. It pays **only** when your signals are both real "
            "and decorrelated. The viral version supplies neither."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Each signal is weak — 52% hit-rate, nothing you'd trade alone. But standardise all "
            "K of them into z-scores, weight each by its historical Sharpe, and add them up. The "
            "composite **ζ-field** stays flat when the signals disagree and fires when they align. "
            "The backtest against buy-and-hold is something retail has never seen.\"*\n\n"
            "The picture is seductive because part of it is **true**. Diversifying across many small, "
            "*independent* bets is exactly how real quant books are built — it's the oldest idea in "
            "finance wearing a new costume. The sleight of hand is in two words the thread never "
            "stresses: the bets have to be **real**, and they have to be **different**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things hide inside \"stack the signals,\" and only one is honest. (1) *Do "
            "decorrelated real edges add up?* Yes — like **√K**: stack 4 independent signals and you "
            "roughly double the information; stack 100 and you get 10×. That's genuine. (2) *Does any "
            "of that survive if the signals are noise, or redundant, or hand-picked after the fact?* "
            "**No.** A composite of coin-flips is still a coin-flip. Fifty correlated signals are one "
            "signal. And weighting by *historical* Sharpe means choosing winners with hindsight — the "
            "purest form of fooling yourself. The whole game is whether the √K boost is **earned** or "
            "**manufactured**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We build the *exact* machine — z-score each signal, weight, sum into a composite, trade "
            "its sign — and then we run it on cases where we **know the truth**:\n\n"
            "1. **The null.** 10 signals that are *pure noise* (zero real edge). Stacking can only "
            "bottle luck — so the composite had better look like luck.\n"
            "2. **The positive control.** 10 signals each with a tiny *real* edge, **decorrelated**. "
            "Now the stacker has something to compound — the √K boost should appear.\n"
            "3. **The redundancy test.** Same real edge, but the signals are near-**copies**. The "
            "boost should collapse.\n"
            "4. **The real tape.** The same 12-signal stack on SPY — what happens when there's no "
            "decorrelated edge lying around?\n"
            "5. **The honest curve test.** Cherry-pick signals in the first half of history, measure "
            "in the second half, and watch the \"edge\" evaporate."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: stacking noise makes nothing.** Here are three worlds side by side — 10 "
            "pure-noise signals, 10 *real-and-different* signals, and 10 *real-but-redundant* "
            "signals — and what the composite's Sharpe ratio comes out to in each."
        ),
        code(
            "cases = [('10 noise\\nsignals', 0.0, 0.0, GREY),\n"
            "         ('10 real +\\ndifferent', 0.05, 0.0, GREEN),\n"
            "         ('10 real but\\nredundant', 0.05, 0.9, AMBER)]\n"
            "shs = []\n"
            "for _, ic, corr, _ in cases:\n"
            "    s, r, _ = data.synthetic_panel(n_signals=10, signal_ic=ic, signal_corr=corr, seed=401)\n"
            "    shs.append(st.sharpe(st.backtest_composite(s, r)))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar([c[0] for c in cases], shs, color=[c[3] for c in cases], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(shs): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('composite Sharpe ratio')\n"
            "ax.set_title('Stacking only pays when signals are REAL and DIFFERENT')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe — noise / real+different / real+redundant:', [round(v,2) for v in shs])"
        ),
        md(
            f"There it is. Noise stacks to a Sharpe of about **{R['null']['comp_sh']:.2f}** (nothing). "
            f"Real-and-different stacks to **{R['pos']['comp_sh']:.2f}** (a genuine blend). But "
            f"real-but-*redundant* — the same edge, just correlated — collapses to "
            f"**{R['red']['comp_sh']:.2f}**. The thread's silent assumption is that your 50 signals "
            "are all different. They almost never are."
        ),
        md(
            "**The √K law, drawn.** As you stack more signals, how much does the composite's "
            "*information* grow? The honest answer is **√K** — *if* the signals are decorrelated. "
            "Watch the redundant case flatline."
        ),
        code(
            "ks = [1, 2, 5, 10, 20, 40]\n"
            "if HAVE_REAL or True:  # synthetic, runs anywhere\n"
            "    dec, red = [], []\n"
            "    for k in ks:\n"
            "        sd, rd, _ = data.synthetic_panel(n_signals=k, signal_ic=0.05, signal_corr=0.0, seed=401)\n"
            "        dec.append(st.composite_ic(sd, rd)['lift'])\n"
            "        sr, rr, _ = data.synthetic_panel(n_signals=k, signal_ic=0.05, signal_corr=0.9, seed=401)\n"
            "        red.append(st.composite_ic(sr, rr)['lift'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot(ks, [np.sqrt(k) for k in ks], '--', c=GREY, lw=2, label='the √K ideal')\n"
            "ax.plot(ks, dec, 'o-', c=GREEN, lw=2, label='real + decorrelated (rides √K)')\n"
            "ax.plot(ks, red, 's-', c=RED, lw=2, label='real but redundant (plateaus)')\n"
            "ax.set_xlabel('number of signals stacked (K)'); ax.set_ylabel('information lift vs one signal')\n"
            "ax.set_title('Decorrelated edges add up like √K; redundant ones hit a ceiling')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('decorrelated lift @K=40:', round(dec[-1],2), '(√40 =', round(np.sqrt(40),2), ') | redundant:', round(red[-1],2))"
        ),
        md(
            f"The green dots climb almost exactly along the dashed **√K** line — at 40 signals the "
            f"lift is **{R['curve_decorr'][-1][1]:.2f}** (√40 ≈ 6.32). The red squares barely leave "
            f"**{R['curve_redund'][-1][1]:.2f}**, no matter how many you pile on. *Correlation is the "
            "thing the viral thread never mentions, and it decides everything.*"
        ),
        md(
            "**Where the magic curve really comes from.** Now the honest test of that soaring "
            "backtest: pick the best-looking signals in the **first half** of history, then measure "
            "the very same blend in the **second half** it never saw. On a real edge it should "
            "survive; on noise (and on real markets) it should die."
        ),
        code(
            "labels, is_v, oos_v = [], [], []\n"
            "for tag, ic, corr in [('real edge\\n(ic=0.05)', 0.05, 0.0), ('pure noise\\n(ic=0)', 0.0, 0.0)]:\n"
            "    s, r, _ = data.synthetic_panel(n_signals=10, signal_ic=ic, signal_corr=corr, seed=401)\n"
            "    sn = st.snoop_split(s, r); labels.append(tag); is_v.append(sn['is_sharpe']); oos_v.append(sn['oos_sharpe'])\n"
            "if HAVE_REAL:\n"
            "    sn = st.snoop_split(SIG, FWD); labels.append('real SPY\\nstack'); is_v.append(sn['is_sharpe']); oos_v.append(sn['oos_sharpe'])\n"
            "else:\n"
            "    labels.append('real SPY\\nstack'); is_v.append(R['real']['snoop_is']); oos_v.append(R['real']['snoop_oos'])\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, is_v, .4, color=AMBER, label='in-sample (the pretty curve)')\n"
            "ax.bar(x+.2, oos_v, .4, color=GREY, label='out-of-sample (what you get)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('Sharpe ratio of the cherry-picked blend')\n"
            "ax.set_title('Cherry-picking survives only when the edge is REAL'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('IS->OOS Sharpe:', [(l.replace(chr(10),' '), round(a,2), round(b,2)) for l,a,b in zip(labels,is_v,oos_v)])"
        ),
        md(
            f"On a **real** edge the cherry-picked blend mostly survives ({R['pos']['is_sh']:.2f} → "
            f"{R['pos']['oos_sh']:.2f}). On **noise** the in-sample shine is just selection. And on "
            f"the **real SPY tape** the snooper even hand-picks **two of the three pure decoys** "
            f"(`noise_a`, `noise_c`) as 'best signals,' prints a flat in-sample number, then loses "
            f"out of sample ({R['real']['snoop_is']:.2f} → {R['real']['snoop_oos']:.2f}). *That* is "
            "how the viral curve is manufactured."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** This is a **methods demo**: the lesson holds in the lab, and on the "
            f"real SPY tape the composite is *negative* (Sharpe **{R['real']['comp_sh']:.2f}**, beaten "
            f"by buy-and-hold **+{R['real']['bh_sh']:.2f}**). No secret edge is being sold.\n"
            "- **Tradability — Mirage.** On noise the composite is luck (permutation "
            f"**p = {R['null']['perm_p']:.2f}**); redundant signals plateau; the only positive "
            "backtest is the snooped one, and it dies out of sample.\n"
            "- **A composite edge? — Busted.** Stacking is **√(effective breadth)** in disguise. It "
            "compounds only when signals are real *and* different. Same shape as "
            "[343–350](../../343-data-mining-roulette/) and [399](../../399-kalshi-efficiency/)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually run it? — the redundancy trap\n\n"
            "Forget significance for a second. Suppose you really did have 10 genuinely-edged signals. "
            "The operational killer is that signals built by humans are **correlated** (everyone uses "
            "momentum, everyone uses moving averages). Here's the composite Sharpe as your signals go "
            "from independent to identical — the cliff is the whole story."
        ),
        code(
            "corrs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]\n"
            "shs = []\n"
            "for c in corrs:\n"
            "    s, r, _ = data.synthetic_panel(n_signals=10, signal_ic=0.05, signal_corr=c, seed=401)\n"
            "    shs.append(st.sharpe(st.backtest_composite(s, r)))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([c*100 for c in corrs], shs, 'o-', c=RED, lw=2)\n"
            "ax.set_xlabel('how correlated the 10 signals are (%)'); ax.set_ylabel('composite Sharpe')\n"
            "ax.set_title('The same real edge, dissolved by redundancy')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe falls from', round(shs[0],2), '(independent) to', round(shs[-1],2), '(near-identical)')"
        ),
        md(
            "The same 10 real edges are worth a Sharpe of ~2 when independent and almost nothing when "
            "they overlap. In the real world your fifty 'signals' are mostly the *same three ideas* "
            "re-skinned — which is why the gorgeous composite is, operationally, a mirage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same trap, other costumes.** "
            "[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/) randomises the "
            "*rule*; [Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/) and "
            "[Study 348 — Curve-Fitting](../../348-curve-fitting/) fit the *parameters*; here we "
            "fit the *combination*. All three are the same act of selecting on noise.\n"
            "- **A demo cousin.** [Study 399 — Kalshi-Efficiency](../../399-kalshi-efficiency/): a "
            "machine validated in the lab, with the 'real' tape labelled an illustration — no `REAL` "
            "stamp claimed.\n"
            "- **Build your own.** Swap in *your* favourite signals, measure their pairwise "
            "correlations, and ask the honest question: how many *independent* bets do you really "
            "have? Usually far fewer than the number of columns.\n\n"
            "*Think your stack of fifty is the real thing? Plant a real decorrelated edge, show the "
            "composite landing **outside** its permutation null **and** surviving out of sample — "
            "then we'll talk.*"
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
            "# Signal-Stacking — a quantitative teardown 🔬\n"
            "### The composite z-score · the √K information-coefficient law · a permutation null on "
            "the composite Sharpe · the in/out-of-sample snoop tax · the redundancy ceiling\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We take "
            "the viral *\"stack K weak signals into a Sharpe-weighted composite\"* recipe at its word "
            "and separate the two things it fuses: a **real but bounded** √K diversification benefit "
            "from the **manufactured** benefit of selecting signals and weights in-sample. The "
            "decisive objects are not the point estimates but **Grinold & Kahn's fundamental law** "
            "(IC × √breadth), a **permutation null** on the composite Sharpe, and an **in/out-of-"
            "sample split** that prices the snooping tax.\n\n"
            "> ⚠️ **Methods-demo note.** The lesson is established on a deterministic analytic panel "
            "where every IC and cross-correlation is **known in closed form**; the real SPY stack is "
            "an explicit **illustration** of the machinery (it can't back a `REAL` stamp). Real data: "
            "yfinance SPY total-return, 1995→2026, cache-first. Offline core is deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Methods demo. Null composite Sharpe **{R['null']['comp_sh']:.2f}** "
            f"(perm **p = {R['null']['perm_p']:.2f}**, IC lift **{R['null']['lift']:.2f}** — no √K); "
            f"real SPY composite **{R['real']['comp_sh']:.2f}** (perm **p = {R['real']['perm_p']:.2f}**), "
            f"beaten by buy-and-hold **+{R['real']['bh_sh']:.2f}**. No `REAL` edge claimed. |\n"
            f"| **Tradability** | `MIRAGE` | Positive control composite **{R['pos']['comp_sh']:.2f}** "
            f"(lift **{R['pos']['lift']:.2f} ≈ √10**, perm **p = {R['pos']['perm_p']:.3f}**) is the *only* "
            f"real blend; redundancy (corr 0.9) crushes lift to **{R['red']['lift']:.2f}**; the snooped "
            f"blend's IS Sharpe evaporates OOS. |\n"
            f"| **A composite edge?** | `BUSTED` | The √K boost is √(*effective breadth*): "
            f"decorrelated lift rides √K (**{R['curve_decorr'][-1][1]:.2f}** at K=40), redundant lift "
            f"plateaus at **{R['curve_redund'][-1][1]:.2f}**. Same shape as 343–350 / 399. |\n\n"
            "> 💡 In plain words: stacking is **diversification with extra steps**. It compounds an "
            "edge only when the bets are real *and* independent; on noise, on redundancy, or on a "
            "real tape with no decorrelated structure, the 'composite field' is luck plus selection."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_{k,t}$ be K standardised signals known at the close of $t$ and $r_{t}$ the "
            "$t\\to t+1$ return. The composite is $Z_t = \\sum_k w_k\\,z(x_{k,t})$, traded by sign "
            "past a band. The recipe's three hypotheses:\n\n"
            "- **H₁ (stacking lifts the IC).** $\\mathrm{IC}(Z) \\approx \\sqrt{K}\\cdot\\overline{\\mathrm{IC}}(x_k)$ "
            "— Grinold & Kahn's fundamental law, $\\mathrm{IR} \\approx \\mathrm{IC}\\sqrt{\\text{breadth}}$.\n"
            "- **H₂ (the lift is free of redundancy).** Breadth is the *effective* number of "
            "independent bets; for equicorrelated signals it is governed by $1/(1+(K-1)\\rho)$.\n"
            "- **H₃ (the composite carries a deployable edge).** The Sharpe-weighted composite beats "
            "its best single signal *out of sample*, net of the selection used to build it.\n\n"
            "We find **H₁ true only under decorrelation** (lift rides √K in the positive control, "
            f"**{R['pos']['lift']:.2f} ≈ √10**), **H₂ rejected the moment $\\rho$ rises** (redundant "
            f"lift **{R['red']['lift']:.2f}**), and **H₃ rejected off the controlled panel** (null perm "
            f"**p = {R['null']['perm_p']:.2f}**; real SPY composite negative; snooped IS Sharpe decays "
            "OOS). The law is real; the viral application supplies neither realness nor independence."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one decomposition: the composite's information coefficient against "
            "the average single signal's, judged by **effective breadth** and by a **null that breaks "
            "the signal→return link**.\n\n"
            "$$\\mathrm{IC}(Z)\\;=\\;\\frac{\\sum_k w_k\\,\\mathrm{IC}(x_k)}"
            "{\\sqrt{\\,w^\\top \\Sigma\\, w\\,}},\\qquad "
            "\\text{lift}\\;=\\;\\frac{\\mathrm{IC}(Z)}{\\overline{\\mathrm{IC}}(x_k)}"
            "\\;\\xrightarrow[\\;\\Sigma=I\\;]{}\\;\\sqrt{K}.$$\n\n"
            "When $\\Sigma=I$ (decorrelated) the lift is √K; as the off-diagonals of the signal "
            "covariance $\\Sigma$ grow, the denominator inflates and the lift collapses toward 1. A "
            "raw **Sharpe** of the composite is the wrong lens alone — a flexible object fit to the "
            "data will look good by construction — so the Signal axis is decided by a **permutation "
            "test** (shuffle $r$, recompute the composite Sharpe) and an **in/out-of-sample split** "
            "(select on the first half, pay on the second)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Synthetic panel.** `data.synthetic_panel(signal_ic, signal_corr, …)` — analytic, so "
            "$\\mathrm{IC}(x_k)=\\texttt{signal\\_ic}$ and "
            "$\\mathrm{corr}(x_j,x_k)=\\texttt{ic}^2+(1-\\texttt{ic}^2)\\,\\texttt{corr}$ are known "
            "exactly. **Null** = `ic=0`; **positive control** = `ic=0.05` decorrelated; **redundant** "
            "= `ic=0.05, corr=0.9`. One execution lag baked into the panel.\n"
            f"- **Real tape.** `data.load_real()` — 12 textbook weak signals (momentum, MA-gaps, an "
            f"oscillator, a low-vol screen, **3 pure decoys**) on SPY total-return "
            f"({R['real_start']}→{R['real_end']}, {R['real_days']:,} days), standardised, aligned to "
            "the next-day return. Fingerprint `" + R['fingerprint'] + "`.\n"
            "- **The √K test.** `composite_ic` — composite IC vs the mean single IC, with the √K "
            "reference.\n"
            "- **Null #1 (permutation).** `permutation_pvalue` — fix the composite/position, shuffle "
            "$r$ 2,000×, report $\\Pr[\\text{Sharpe}_{\\text{shuffled}} \\ge \\text{Sharpe}_{\\text{obs}}]$.\n"
            "- **Null #2 (snoop split).** `snoop_split` — rank single-signal Sharpes on the first "
            "half, keep the winners, weight by that in-sample Sharpe, measure on the held-out half.\n"
            "- **HAC inference.** Newey-West *t* on the composite's mean daily return."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The composite vs its own signals — null, real, redundant\n\n"
            "Composite Sharpe against the **best** and **mean** single-signal Sharpe in three worlds. "
            "Only the real-and-decorrelated composite clears the best single signal."
        ),
        code(
            "cases = [('null\\nic=0', 0.0, 0.0), ('real+decorr\\nic=0.05', 0.05, 0.0), ('redundant\\nic=0.05,ρ=0.9', 0.05, 0.9)]\n"
            "comp, best, mean = [], [], []\n"
            "for _, ic, corr in cases:\n"
            "    s, r, _ = data.synthetic_panel(n_signals=10, signal_ic=ic, signal_corr=corr, seed=401)\n"
            "    ex = st.run_experiment(s, r, n_perm=800, seed=401)\n"
            "    comp.append(ex['composite']['sharpe']); best.append(ex['best_single']); mean.append(ex['mean_single'])\n"
            "x = np.arange(len(cases))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "ax.bar(x-.27, comp, .27, color=GREEN, label='composite (equal-weight)')\n"
            "ax.bar(x,      best, .27, color=AMBER, label='best single signal')\n"
            "ax.bar(x+.27, mean, .27, color=GREY,  label='mean single signal')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases])\n"
            "ax.set_ylabel('Sharpe ratio'); ax.set_title('Composite beats the BEST single only when signals are real + decorrelated')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('composite Sharpe:', {c[0].split(chr(10))[0]: round(v,2) for c,v in zip(cases,comp)})"
        ),
        md(
            f"> 💡 In plain words: in the null the composite ({R['null']['comp_sh']:.2f}) is below the "
            f"luckiest single signal ({R['null']['best']:.2f}) — pure selection. In the positive "
            f"control the composite ({R['pos']['comp_sh']:.2f}) **beats the best single** "
            f"({R['pos']['best']:.2f}): genuine compounding. Redundancy drags the composite "
            f"({R['red']['comp_sh']:.2f}) back below its best single. H₁ holds **only** under "
            "decorrelation."
        ),
        md(
            "### 4b · The decisive law — composite IC lift vs √K\n\n"
            "Grinold & Kahn made visual: the information lift over a single signal as K grows, for "
            "decorrelated vs redundant signals, against the √K ideal."
        ),
        code(
            "ks = [1, 2, 5, 10, 20, 40]\n"
            "dec, red = [], []\n"
            "for k in ks:\n"
            "    sd, rd, _ = data.synthetic_panel(n_signals=k, signal_ic=0.05, signal_corr=0.0, seed=401)\n"
            "    dec.append(st.composite_ic(sd, rd)['lift'])\n"
            "    sr, rr, _ = data.synthetic_panel(n_signals=k, signal_ic=0.05, signal_corr=0.9, seed=401)\n"
            "    red.append(st.composite_ic(sr, rr)['lift'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.plot(ks, [np.sqrt(k) for k in ks], '--', c=GREY, lw=2, label=r'$\\sqrt{K}$ ideal (full breadth)')\n"
            "ax.plot(ks, dec, 'o-', c=GREEN, lw=2, label=r'decorrelated ($\\rho=0$): rides $\\sqrt{K}$')\n"
            "ax.plot(ks, red, 's-', c=RED, lw=2, label=r'redundant ($\\rho=0.9$): plateaus')\n"
            "ax.set_xlabel('K (signals stacked)'); ax.set_ylabel(r'IC lift = IC(Z) / mean IC$(x_k)$')\n"
            "ax.set_title('The fundamental law: lift = √(effective breadth)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('decorr lift:', [round(v,2) for v in dec]); print('redund lift:', [round(v,2) for v in red])"
        ),
        md(
            f"> 💡 In plain words: decorrelated lift tracks √K almost perfectly "
            f"(**{R['curve_decorr'][-1][1]:.2f}** at K=40 vs √40 = 6.32); redundant lift is pinned at "
            f"**~{R['curve_redund'][-1][1]:.2f}** for *all* K. Effective breadth, not the column "
            "count, is what stacking buys you — and correlation is what you pay it with."
        ),
        md(
            "### 4c · The permutation null — is the composite better than its reshuffled self?\n\n"
            "Hold the composite/position fixed, shuffle the return vector to break any real link, and "
            "recompute the Sharpe many times. The observed composite Sharpe (line) against that null "
            "(histogram) is the honest test. Null stack vs real-and-decorrelated."
        ),
        code(
            "def perm_dist(s, r, n=2000, seed=401):\n"
            "    score = st.composite_score(s); pos = st.position_from_score(score).to_numpy(float)\n"
            "    rv = r.to_numpy(float); obs = st._sharpe_arr(pos*rv)\n"
            "    rng = np.random.default_rng(seed)\n"
            "    draws = np.array([st._sharpe_arr(pos*rv[rng.permutation(rv.size)]) for _ in range(n)])\n"
            "    return obs, draws\n"
            "sN,rN,_ = data.synthetic_panel(n_signals=10, signal_ic=0.0, signal_corr=0.0, seed=401)\n"
            "sP,rP,_ = data.synthetic_panel(n_signals=10, signal_ic=0.05, signal_corr=0.0, seed=401)\n"
            "obN,dN = perm_dist(sN,rN); obP,dP = perm_dist(sP,rP)\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2),sharey=True)\n"
            "a1.hist(dN,bins=45,color=GREY,alpha=.85); a1.axvline(obN,c=RED,lw=2.5,label=f'observed {obN:.2f}')\n"
            "a1.set_title(f'NULL stack — composite is luck (p≈{(np.mean(dN>=obN)):.2f})'); a1.set_xlabel('shuffled Sharpe'); a1.legend()\n"
            "a2.hist(dP,bins=45,color=GREY,alpha=.85); a2.axvline(obP,c=GREEN,lw=2.5,label=f'observed {obP:.2f}')\n"
            "a2.set_title(f'REAL+decorr — composite is real (p≈{(np.mean(dP>=obP)):.3f})'); a2.set_xlabel('shuffled Sharpe'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null perm p ≈ {np.mean(dN>=obN):.3f} | positive-control perm p ≈ {np.mean(dP>=obP):.3f}')"
        ),
        md(
            f"> 💡 In plain words: the null composite (left) sits **inside** the luck cloud — a coin "
            f"reshuffle matches it **p ≈ {R['null']['perm_p']:.2f}** of the time. The real "
            f"decorrelated composite (right) is far in the right tail (**p ≈ {R['pos']['perm_p']:.3f}**). "
            "The permutation test cleanly separates a manufactured composite from a real one."
        ),
        md(
            "### 4d · The snooping tax — the headline curve, priced\n\n"
            "Select the best signals + Sharpe-weights on the **first half**, then run the *same* blend "
            "on the held-out half. The in-sample bar is the pretty backtest; the out-of-sample bar is "
            "the truth. Real edge survives; noise and the real tape do not."
        ),
        code(
            "rows = []\n"
            "for tag, ic, corr in [('positive control\\n(ic=0.05)', 0.05, 0.0), ('null\\n(ic=0)', 0.0, 0.0)]:\n"
            "    s, r, _ = data.synthetic_panel(n_signals=10, signal_ic=ic, signal_corr=corr, seed=401)\n"
            "    sn = st.snoop_split(s, r); rows.append((tag, sn['is_sharpe'], sn['oos_sharpe'], sn['n_chosen']))\n"
            "if HAVE_REAL:\n"
            "    sn = st.snoop_split(SIG, FWD); rows.append(('real SPY\\nstack', sn['is_sharpe'], sn['oos_sharpe'], sn['n_chosen']))\n"
            "    chosen = sn['chosen']\n"
            "else:\n"
            "    rows.append(('real SPY\\nstack', R['real']['snoop_is'], R['real']['snoop_oos'], R['real']['n_chosen']))\n"
            "    chosen = R['real']['chosen']\n"
            "x = np.arange(len(rows))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x-.2, [r[1] for r in rows], .4, color=AMBER, label='in-sample (selected)')\n"
            "ax.bar(x+.2, [r[2] for r in rows], .4, color=GREY, label='out-of-sample (paid)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])\n"
            "ax.set_ylabel('Sharpe of the snooped blend'); ax.set_title('In-sample selection is the source of the magic curve')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('IS->OOS:', [(r[0].replace(chr(10),' '), round(r[1],2), round(r[2],2)) for r in rows])\n"
            "print('real-tape snooper picked:', chosen, '<- note the pure decoys')"
        ),
        md(
            f"> 💡 In plain words: the snooping tax is the IS→OOS drop. A real edge mostly survives "
            f"(**{R['pos']['is_sh']:.2f} → {R['pos']['oos_sh']:.2f}**); noise and the real tape do not "
            f"(**SPY {R['real']['snoop_is']:.2f} → {R['real']['snoop_oos']:.2f}**, and the snooper "
            "literally selects **two pure noise decoys** as 'best signals'). The viral equity curve "
            "*is* the in-sample bar — and it was never going to survive the held-out half."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal `NONE`** — a **methods demo**. The √K law is established on a controlled "
            f"panel; the real SPY composite is **negative** (Sharpe **{R['real']['comp_sh']:.2f}**, "
            f"HAC *t* = {R['real']['comp_t']:.2f}, perm **p = {R['real']['perm_p']:.2f}**), beaten by "
            f"buy-and-hold (**+{R['real']['bh_sh']:.2f}**). Illustrative tape ⇒ no `REAL` stamp.\n"
            f"- **Tradability `MIRAGE`** — null composite is luck (perm **p = {R['null']['perm_p']:.2f}**, "
            f"lift **{R['null']['lift']:.2f}**); redundancy crushes lift to **{R['red']['lift']:.2f}**; "
            "the only positive backtest is the **snooped** one, whose IS Sharpe evaporates OOS. EV "
            "comes from real decorrelated information the viral stack does not supply.\n"
            "- **A composite edge? `BUSTED`** — stacking is **√(effective breadth)** in Greek "
            f"letters: decorrelated lift rides √K (**{R['curve_decorr'][-1][1]:.2f}** at K=40), "
            f"redundant lift plateaus at **{R['curve_redund'][-1][1]:.2f}**. Same family as the "
            "research-method demos [343–350](../../343-data-mining-roulette/) and "
            "[399](../../399-kalshi-efficiency/)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the effective-breadth curve\n\n"
            "The operational truth in one picture. Even granting K genuinely-edged signals, the "
            "*deliverable* boost is √(effective breadth), and effective breadth dies in correlation: "
            "for equicorrelated signals it is $K/(1+(K-1)\\rho)$. Here is the realised composite Sharpe "
            "across the correlation axis, with the theoretical effective-breadth fraction overlaid."
        ),
        code(
            "corrs = np.linspace(0, 0.95, 10)\n"
            "sh = []\n"
            "for c in corrs:\n"
            "    s, r, _ = data.synthetic_panel(n_signals=10, signal_ic=0.05, signal_corr=float(c), seed=401)\n"
            "    sh.append(st.sharpe(st.backtest_composite(s, r)))\n"
            "K = 10; eff = K/(1+(K-1)*corrs)            # effective number of independent bets\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(corrs, sh, 'o-', c=RED, lw=2, label='realised composite Sharpe (K=10, ic=0.05)')\n"
            "ax.set_xlabel('pairwise signal correlation ρ'); ax.set_ylabel('composite Sharpe', color=RED)\n"
            "ax2 = ax.twinx(); ax2.plot(corrs, eff, '--', c=GREY, lw=2, label='effective # of independent bets')\n"
            "ax2.set_ylabel('effective breadth  K/(1+(K-1)ρ)', color=GREY); ax2.grid(False)\n"
            "ax.set_title('The deliverable edge is √(effective breadth) — and ρ destroys it')\n"
            "l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2, la1+la2, loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe at ρ=0:', round(sh[0],2), '| at ρ≈0.95:', round(sh[-1],2), '| eff breadth there:', round(eff[-1],2))"
        ),
        md(
            "> 💡 In plain words: the grey dashed line is how many *independent* bets 10 correlated "
            "signals are really worth — it falls toward **1** as ρ → 1, and the red Sharpe falls with "
            "it. Real-world signals built by humans (momentum, MA-gaps, oscillators) are heavily "
            "correlated, so your nominal 'fifty signals' are a handful of effective bets. There is no "
            "weighting scheme that manufactures breadth you don't have — **correlation is the verdict.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The combination is just one knob to overfit.** "
            "[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/) (the *rule*), "
            "[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/) and "
            "[Study 348 — Curve-Fitting](../../348-curve-fitting/) (the *parameters*), "
            "[Study 346 — Multiple-Testing](../../346-multiple-testing/) (the *count*). Reading them "
            "together is the cleanest demonstration that selection-on-noise has many costumes.\n"
            "- **A demo cousin.** [Study 399 — Kalshi-Efficiency](../../399-kalshi-efficiency/): a "
            "validated machine with an explicitly illustrative tape and no `REAL` stamp.\n"
            "- **Estimate your own effective breadth.** Take your signal matrix, compute its "
            "correlation eigenvalues, and read off the participation ratio — the honest count of "
            "independent bets. It is almost always a small fraction of the column count.\n\n"
            "*The reproducible core is offline and deterministic; the real SPY tape is an explicit "
            "illustration. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
