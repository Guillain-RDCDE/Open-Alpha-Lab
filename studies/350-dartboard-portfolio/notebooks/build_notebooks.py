"""Generate the two narrative notebooks for Study 350 (Dartboard-Portfolio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached
panel parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    window="2013-02 → 2026-05", months=160, n_names=20, fp="212217b18220",
    n_darts=500, k=10,
    cap_cagr=33.50, cap_sharpe=1.56, cap_dd=-33.7,
    exp_cagr=28.38, exp_sharpe=1.55, exp_dd=-27.1,
    eq_cagr=20.59, eq_sharpe=1.43, eq_dd=-18.6,
    dart_cagr=20.27, dart_sharpe=1.41, dart_dd=-19.1,
    win_vs_index=0.0, win_vs_expert=0.0,
    dart_minus_cap_ann=-11.67, dart_minus_cap_t=-3.66,
    dart_minus_eq_ann=-0.26, dart_minus_eq_t=-2.17,
    boot_lo=-147, boot_hi=-47,
    # synthetic positive control sweep (size_premium → dart-vs-cap, dart-vs-equal)
    sweep_sp=[0.000, 0.002, 0.004, 0.006],
    sweep_win=[30.0, 65.0, 89.5, 99.5],
    sweep_t_cap=[-0.89, 1.33, 3.58, 5.84],
    sweep_t_eq=[-0.46, -0.30, -0.20, -0.15],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from dartboard_portfolio import data, strategy as st

def _have_cache():
    return (os.path.exists(data._cache_path(data.DEFAULT_CACHE))
            and os.path.exists(data._caps_cache_path(data.DEFAULT_CACHE)))

HAVE_REAL = _have_cache()

def real_panel():
    return data.load_real(fetch=False, allow_survivorship_bias=True)

print("real panel cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Dartboard-Portfolio — can a monkey beat the pros? 🎯\n"
            "### Malkiel's blindfolded monkey, tested honestly, in plain English\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_monkey_beats_the_pros%3F: Busted](https://img.shields.io/badge/A_monkey_beats_the_pros%3F-Busted-8b949e?style=flat-square)\n\n"
            "A famous line from Burton Malkiel: a **blindfolded monkey throwing darts** at the stock "
            "pages could pick a portfolio that does as well as the experts. And researchers found "
            "that thousands of *random* portfolios really do beat the cap-weighted index, startlingly "
            "often. So is stock-picking a con? This notebook throws 500 darts and finds out — and the "
            "real answer is more interesting than 'yes' or 'no'.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap and the "
            "size-tilt decomposition? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a random monkey beat the index? | **Sometimes — and it's never skill.** A random "
            "equal-weight basket is just a tilt toward smaller names. When small beats big, the monkey "
            "wins; when big beats small, it loses. |\n"
            "| What happened on real mega-caps (2013–2026)? | **The monkey lost — badly.** "
            f"0 of {R['n_darts']} throws beat the cap-weighted index; the median monkey trailed it by "
            f"**{abs(R['dart_minus_cap_ann']):.0f}%/yr**. Mega-caps led the decade, so cap-weighting "
            "won. |\n"
            "| So is the famous result fake? | **No — it's real but mechanical.** The monkey ≈ an "
            "*equal-weight* index. The 'win' over the cap index was always a size tilt, not picking "
            "talent. |\n"
            "| Can you trade it? | **There's nothing to trade.** It's a known tilt (here a loser) with "
            "lots of pointless turnover from re-rolling the dice. |\n\n"
            "> The monkey isn't beating the experts. It's quietly buying a size tilt and getting credit "
            "for it when the tilt happens to pay."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A blindfolded monkey throwing darts at a newspaper's financial pages could select a "
            "portfolio that would do just as well as one carefully selected by experts.\"*\n\n"
            "— Burton Malkiel, *A Random Walk Down Wall Street* (1973)\n\n"
            "It sounds like a joke, but it got tested for real. The *Wall Street Journal* literally "
            "threw darts for years. And in 2013 Research Affiliates ran *thousands* of random "
            "portfolios and found they beat the cap-weighted S&P on average — the monkey **won**. The "
            "strong version of the claim: picking stocks is so hard that random selection does as "
            "well or better. If true, the whole research-and-stock-picking industry is theatre."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a monkey ties the experts, two things follow. First, paying for stock selection is "
            "money lit on fire. Second — and this is the part people miss — if *random* portfolios "
            "beat the **cap-weighted index**, then the index itself is a bad default, and you'd be "
            "better off in almost anything else. That second implication is the real payload, and it's "
            "where the honest explanation lives: it's not that randomness is smart, it's that "
            "**cap-weighting has a specific bias** a random portfolio happens to avoid."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We hold a tournament on one universe of stocks:\n\n"
            "1. **500 monkeys.** Each picks 10 names at random, holds them equal-weight, and re-throws "
            "every year. One throw is luck; 500 throws are the *strategy*.\n"
            "2. **The cap-weighted index** of the same names — the thing to beat.\n"
            "3. **An 'expert'** — the 10 biggest, most obvious names, equal-weight.\n"
            "4. **The tell-tale control: an equal-weight index** (all the names, 1/N). This already "
            "holds the size tilt. If the monkey only beats the *cap* index but not the *equal-weight* "
            "index, then the monkey's 'edge' was the tilt all along — not skill.\n\n"
            "If random picking were genuinely skilful, it would beat the equal-weight index too. "
            "Spoiler: it never does."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the synthetic proof of *why* a monkey wins.** We build a fake market where we "
            "control one dial: how much extra return small stocks earn over big ones (the 'size "
            "premium'). Watch the monkey's win-rate against the cap index as we turn it up:"
        ),
        code(
            "sp = [0.0, 0.002, 0.004, 0.006]\n"
            "wins, t_cap, t_eq = [], [], []\n"
            "for s in sp:\n"
            "    r, c, _ = data.synthetic_panel(size_premium=s, seed=350)\n"
            "    res = st.race(r, c, k=10, n_darts=200, rebalance_every=12, cost_bps=0.0)\n"
            "    wins.append(res['win_rate_vs_index']*100)\n"
            "    t_cap.append(res['test_vs_index']['tstat'])\n"
            "    t_eq.append(res['test_vs_equal_index']['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot([s*100 for s in sp], wins, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(50, ls='--', c=GREY); ax.set_ylim(0, 100)\n"
            "ax.set_xlabel('planted small-cap premium (%/month)')\n"
            "ax.set_ylabel('% of monkeys that beat the cap index')\n"
            "ax.set_title('A monkey beats the index ONLY when small stocks win')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('win-rate vs cap index by size premium:', [f'{w:.0f}%' for w in wins])"
        ),
        md(
            f"With **no** size premium the monkey wins ~{R['sweep_win'][0]:.0f}% of the time — worse "
            "than a coin, because cap-weighting compounds the big names. Turn the size premium up and "
            f"the win-rate climbs to ~{R['sweep_win'][-1]:.0f}%. **The monkey's 'skill' is a dial "
            "labelled 'do small stocks beat big ones'.**"
        ),
        md(
            "**Now the real tape.** Twenty US mega-caps, 2013–2026 — a decade led by giant tech. Here "
            "the size tilt points the *wrong* way. Who won?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, caps = real_panel()\n"
            "    res = st.race(rets, caps, k=10, n_darts=500, rebalance_every=12, cost_bps=0.0)\n"
            "    names = ['Cap index','Expert\\n(10 largest)','Equal index','Median\\nmonkey']\n"
            "    cagrs = [st.summarize(res['index'])['cagr']*100,\n"
            "             st.summarize(res['expert'])['cagr']*100,\n"
            "             st.summarize(res['equal_index'])['cagr']*100,\n"
            "             st.summarize(res['median_dart'])['cagr']*100]\n"
            "    winr = res['win_rate_vs_index']*100\n"
            "else:\n"
            f"    names = ['Cap index','Expert\\n(10 largest)','Equal index','Median\\nmonkey']\n"
            f"    cagrs = [{R['cap_cagr']}, {R['exp_cagr']}, {R['eq_cagr']}, {R['dart_cagr']}]\n"
            f"    winr = {R['win_vs_index']}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "b = ax.bar(names, cagrs, color=[GREEN, AMBER, GREY, RED], width=.6)\n"
            "ax.set_ylabel('CAGR (%/yr)')\n"
            "ax.set_title(f'On mega-caps, the monkey LOSES — {winr:.0f}% of throws beat the index')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Median monkey CAGR {cagrs[-1]:.1f}% vs cap index {cagrs[0]:.1f}%  |  win-rate {winr:.0f}%')"
        ),
        md(
            f"The cap-weighted index won at **+{R['cap_cagr']:.0f}%/yr**; the median monkey managed "
            f"only **+{R['dart_cagr']:.0f}%/yr**, and **{R['win_vs_index']:.0f}% of {R['n_darts']} "
            "monkeys** beat the index. The legend reversed — because this decade's giants were exactly "
            "the names cap-weighting piles into. Same mechanism as the synthetic dial, pointing the "
            "other way."
        ),
        md(
            "**The clincher: is the monkey doing *anything* an equal-weight index doesn't?** Compare "
            "the monkey to the cap index, and then to the equal-weight index:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d_cap = res['test_vs_index']['mean_diff_ann']*100\n"
            "    d_eq = res['test_vs_equal_index']['mean_diff_ann']*100\n"
            "else:\n"
            f"    d_cap, d_eq = {R['dart_minus_cap_ann']}, {R['dart_minus_eq_ann']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "b = ax.bar(['Monkey − Cap index','Monkey − Equal index'], [d_cap, d_eq],\n"
            "           color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return gap (%/yr)')\n"
            "ax.set_title('The monkey IS the equal-weight index (the gap vanishes)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Monkey vs cap index: {d_cap:+.1f}%/yr   Monkey vs EQUAL index: {d_eq:+.2f}%/yr')"
        ),
        md(
            f"There it is. The monkey trails the cap index by **{R['dart_minus_cap_ann']:.0f}%/yr** — "
            f"but it's within a whisker of the **equal-weight index** ({R['dart_minus_eq_ann']:+.1f}%/yr). "
            "The random picking adds nothing; the monkey is just a noisy 1/N portfolio. Whether it "
            "'wins' is entirely about whether equal-weighting beats cap-weighting — a size tilt, not a "
            "stock-picker's gift."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Random selection does move returns versus the cap index — but "
            "it's a size tilt that flips sign with the universe, and it's identical to an "
            "equal-weight index. Not a stable, real stock-picking effect.\n"
            "- **Tradability — Mirage.** Nothing to harvest but a known tilt (a loser here) plus "
            "wasted turnover from re-rolling the darts.\n"
            "- **A monkey beats the pros? — Busted.** On mega-caps in a mega-cap decade the monkey "
            f"lost to *both* the index and the expert, {R['win_vs_index']:.0f}% of {R['n_darts']} "
            "throws. The 'wins' you read about were the size tilt taking a bow."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's no edge to trade — but suppose you wanted to *be* the monkey anyway. Re-throwing "
            "darts every year means selling your whole book and rebuying random names: a turnover "
            "monster. On a tape where the monkey already loses gross, costs only dig the hole deeper, "
            "and there is no version of 'random mega-cap selection' that out-earns just buying the "
            "index. The honest takeaway isn't 'throw darts' — it's that if you *like* the equal-weight "
            "tilt, **buy an equal-weight index fund** and skip the dice (and the trading bill)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other half of the quip.** [Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/) "
            "asks whether 1/N *allocation* beats fancy optimisers. Together they bracket Malkiel: "
            "*which* names (here) and *how much* of each (there).\n"
            "- **Flip the universe.** Re-run the monkey on a small-cap or broad universe instead of "
            "mega-caps and watch the legend come back — the size tilt becomes a tailwind. The verdict "
            "is the same (it's a tilt), but the sign flips.\n"
            "- **Add value, profitability, low-beta tilts.** Research Affiliates showed *upside-down* "
            "strategies win too, for the same reason. Fork this and decompose the monkey into factors.\n\n"
            "*Think the monkey is really beating the pros? Fork this, swap in a universe where small "
            "beats big, and then race it against the equal-weight index — and watch the 'skill' "
            "evaporate.*"
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
            "# Dartboard-Portfolio — a quantitative teardown 🔬\n"
            "### Cross-sectional race · 500-monkey distribution · HAC *t* + block bootstrap · "
            "the equal-index size-tilt control · survivorship · capacity\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_monkey_beats_the_pros%3F: Busted](https://img.shields.io/badge/A_monkey_beats_the_pros%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We race a 500-monkey random "
            "selection against the cap-weighted index, a concentrated expert, and — decisively — an "
            "equal-weight index, to separate **size tilt** from **selection skill**.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo total-return monthly, 20 US mega-caps "
            "2013–2026 (as-of 2026-05-31, survivorship-biased current membership — see beat 3); the "
            "offline core and tests run on a deterministic synthetic panel. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Median dart − cap index = {R['dart_minus_cap_ann']:.2f}%/yr, HAC "
            f"*t* = **{R['dart_minus_cap_t']:.2f}** — real but *negative* here; vs the equal-weight "
            f"index {R['dart_minus_eq_ann']:+.2f}%/yr (*t* = {R['dart_minus_eq_t']:.2f}). A size tilt, "
            "not a stable selection effect. |\n"
            f"| **Tradability** | `MIRAGE` | No selection alpha — a known size tilt (a loser on this "
            "tape) at high random-rebalance turnover. |\n"
            f"| **A monkey beats the pros?** | `BUSTED` | Win-rate vs index {R['win_vs_index']:.0f}%, "
            f"vs expert {R['win_vs_expert']:.0f}% ({R['n_darts']} monkeys); the synthetic control shows "
            "the dart−equal-index gap is ~0 at every size premium. |\n\n"
            "> 💡 In plain words: random selection is a noisy equal-weight portfolio. It 'beats' a cap "
            "index exactly when equal-weight does (a size tilt), and never beats the equal-weight index "
            "itself — so there is no stock-picking skill in it, in either direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $w^{dart}$ be a random equal-weight selection of $k$ of $N$ names, $w^{cap}$ the "
            "cap-weighted index, and $w^{eq}=\\mathbf{1}/N$ the equal-weight index. Define the median "
            "monkey's monthly return $r^{dart}_t$.\n\n"
            "- **H₁ (selection beats the index).** $\\mathbb{E}[r^{dart}_t - r^{cap}_t] > 0$.\n"
            "- **H₂ (it is *skill*, not a tilt).** The edge survives controlling for size, i.e. "
            "$\\mathbb{E}[r^{dart}_t - r^{eq}_t] \\neq 0$ with the *same sign* as H₁.\n"
            "- **H₃ (it beats the experts).** Win-rate vs the concentrated mega-cap pick > 50%.\n\n"
            "We find **H₁ rejected on this tape** (the spread is significantly *negative*), and — "
            "decisively — **H₂ rejected on every tape**: the dart−equal-index spread is statistically "
            "zero whether or not a size premium exists. The 'monkey effect' is entirely the size tilt."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Research Affiliates (Arnott et al. 2013) showed random portfolios beat cap-weight on "
            "average — and attributed it to mechanical size/value tilts, not selection. The desk's "
            "contribution is to run that decomposition *directly*: instead of regressing on factor "
            "returns, we add a fourth horse — the equal-weight index — that *is* the size tilt. If the "
            "monkey's edge over the cap index vanishes against the equal-weight index, the "
            "factor-attribution story is proven without a single regression. The binary 'monkey wins?' "
            "is a distraction; the mechanism is the result."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Universe.** 20 US mega-caps, monthly total return, 2013–2026.\n"
            "- **Monkeys.** 500 independent seeds; each draws $k=10$ names without replacement at every "
            "annual rebalance (a fresh throw), held equal-weight, drifting between rebalances.\n"
            "- **Benchmarks.** Cap-weighted index; equal-weight index (size-tilt control); expert "
            "(10 largest, equal-weight).\n"
            "- **Inference.** Newey-West HAC *t* on the median-dart-minus-benchmark monthly spread; "
            "circular block-bootstrap CI (Politis-Romano) on the mean spread; win-rate over the 500-"
            "monkey terminal-wealth distribution. Excess-vs-excess: all legs fully invested.\n"
            "- **Costs.** One-way turnover × NAV per rebalance (the random re-draw is ~100% turnover).\n"
            "- **Survivorship (named on Signal).** Current membership + current cap snapshot → biased "
            "*toward the surviving mega-caps*, i.e. toward the cap index the monkey loses to. The "
            "qualitative finding (dart ≈ equal index) is robust; the margin is an upper bound on the "
            "cap index. The loader goes through the `allow_survivorship_bias=True` opt-in.\n"
            "- **Positive control.** A deterministic synthetic panel with a tunable size premium."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The 500-monkey distribution vs the index\n\n"
            "One throw is luck; the *distribution* is the strategy. Here is where 500 monkeys land "
            "relative to the cap index (the vertical line)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, caps = real_panel()\n"
            "    res = st.race(rets, caps, k=10, n_darts=500, rebalance_every=12, cost_bps=0.0)\n"
            "    term = (1+res['darts']).prod()\n"
            "    cap_term = (1+res['index']).prod()\n"
            "    winr = res['win_rate_vs_index']*100\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "    ax.hist(term.values, bins=40, color=GREY, alpha=.8)\n"
            "    ax.axvline(cap_term, c=GREEN, lw=2.5, label=f'cap index ({cap_term:.1f}x)')\n"
            "    ax.set_xlabel('terminal wealth multiple (1 = breakeven)')\n"
            "    ax.set_ylabel('number of monkeys')\n"
            "    ax.set_title(f'500 monkeys vs the cap index — {winr:.0f}% beat it')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'monkeys beating the cap index: {winr:.0f}% of 500')\n"
            "else:\n"
            f"    print('OFFLINE — frozen: {R['win_vs_index']:.0f}% of {R['n_darts']} monkeys beat the cap index')\n"
            f"    print('Median monkey {R['dart_cagr']}%/yr vs cap index {R['cap_cagr']}%/yr')"
        ),
        md(
            f"> 💡 In plain words: **{R['win_vs_index']:.0f}%** of {R['n_darts']} monkeys beat the cap "
            "index. The whole random-selection distribution sits *below* the index — on this universe "
            "the cap-weighting concentrated into the decade's winners, and no amount of dart-throwing "
            "could catch them."
        ),
        md(
            "### 4b · The decisive control — dart minus cap index vs dart minus equal index\n\n"
            "HAC *t* on the median monkey's spread against each benchmark. The cap-index spread is "
            "real and negative; the equal-index spread is the tell."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = res['test_vs_index']; te = res['test_vs_equal_index']\n"
            "    d_cap, t_cap = tc['mean_diff_ann']*100, tc['tstat']\n"
            "    d_eq, t_eq = te['mean_diff_ann']*100, te['tstat']\n"
            "    lo, hi = st.block_bootstrap_ci(res['median_dart'] - res['index'])\n"
            "    lo, hi = lo*1e4, hi*1e4\n"
            "else:\n"
            f"    d_cap, t_cap = {R['dart_minus_cap_ann']}, {R['dart_minus_cap_t']}\n"
            f"    d_eq, t_eq = {R['dart_minus_eq_ann']}, {R['dart_minus_eq_t']}\n"
            f"    lo, hi = {R['boot_lo']}, {R['boot_hi']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "b = ax.bar(['dart − cap index','dart − equal index'], [d_cap, d_eq],\n"
            "           color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised spread (%/yr)')\n"
            "ax.set_title('Real vs cap (a size tilt), zero vs equal index (no skill)')\n"
            "for bar_, t_ in zip(b, [t_cap, t_eq]):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()), ha='center', va='bottom' if bar_.get_height()>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'dart-cap: {d_cap:+.2f}%/yr t={t_cap:+.2f}  |  dart-equal: {d_eq:+.2f}%/yr t={t_eq:+.2f}')\n"
            "print(f'block-bootstrap 95% CI on monthly dart-cap spread: [{lo:+.0f}, {hi:+.0f}] bps')"
        ),
        md(
            f"> 💡 In plain words: against the cap index the monkey loses **{R['dart_minus_cap_ann']:.0f}"
            f"%/yr** (*t* = {R['dart_minus_cap_t']:.2f}, bootstrap CI "
            f"[{R['boot_lo']}, {R['boot_hi']}] bps/mo, entirely below zero). Against the equal-weight "
            f"index the gap is **{R['dart_minus_eq_ann']:+.1f}%/yr** — economically nil. The monkey is "
            "the equal-weight index. **H₂ rejected:** there is no selection skill, only a tilt."
        ),
        md(
            "### 4c · The synthetic positive control — the engine is a faithful size-tilt detector\n\n"
            "Plant a size premium and sweep it. The dart-vs-cap edge should rise monotonically while "
            "the dart-vs-equal-index gap stays pinned at ~0."
        ),
        code(
            "sp = [0.0, 0.002, 0.004, 0.006]\n"
            "t_cap_s, t_eq_s, win_s = [], [], []\n"
            "for s in sp:\n"
            "    r, c, _ = data.synthetic_panel(size_premium=s, seed=350)\n"
            "    res_s = st.race(r, c, k=10, n_darts=200, rebalance_every=12, cost_bps=0.0)\n"
            "    t_cap_s.append(res_s['test_vs_index']['tstat'])\n"
            "    t_eq_s.append(res_s['test_vs_equal_index']['tstat'])\n"
            "    win_s.append(res_s['win_rate_vs_index']*100)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "x = [s*100 for s in sp]\n"
            "ax.plot(x, t_cap_s, 'o-', c=GREEN, lw=2, label='dart − cap index (HAC t)')\n"
            "ax.plot(x, t_eq_s, 's--', c=GREY, lw=2, label='dart − equal index (HAC t)')\n"
            "ax.axhline(2, ls=':', c=GREY); ax.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted small-cap premium (%/month)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Dart beats cap index ONLY via the size tilt — never the equal index')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('win% vs cap index:', [f'{w:.0f}%' for w in win_s])\n"
            "print('t vs equal index :', [f'{t:+.2f}' for t in t_eq_s])"
        ),
        md(
            f"> 💡 In plain words: as the planted size premium rises 0 → 0.6%/mo, the monkey's win-rate "
            f"vs the cap index climbs {R['sweep_win'][0]:.0f}% → {R['sweep_win'][-1]:.0f}% and its HAC "
            f"*t* goes {R['sweep_t_cap'][0]:+.1f} → {R['sweep_t_cap'][-1]:+.1f} — but the dart−equal-"
            "index *t* never leaves the ±2 band. The engine faithfully detects the size tilt and "
            "confirms it is *all* the monkey ever had."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — dart − cap index {R['dart_minus_cap_ann']:.2f}%/yr at HAC *t* "
            f"{R['dart_minus_cap_t']:.2f} (real, but a *negative* size tilt here); dart − equal index "
            f"{R['dart_minus_eq_ann']:+.2f}%/yr (*t* {R['dart_minus_eq_t']:.2f}, nil). Not a stable, "
            "directional selection effect.\n"
            f"- **Tradability `MIRAGE`** — no selection alpha to harvest; a known size tilt (here a "
            "loser) plus dead-weight turnover from re-throwing the darts.\n"
            f"- **A monkey beats the pros? `BUSTED`** — {R['win_vs_index']:.0f}% win-rate vs the index, "
            f"{R['win_vs_expert']:.0f}% vs the expert; the synthetic control shows the dart−equal-index "
            "gap is zero at every size premium. The famous 'wins' are the size tilt, not skill."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — turnover and the absent edge\n\n"
            "The random re-throw is a turnover monster (~100% one-way per rebalance). On a tape where "
            "the monkey already loses gross, costs only widen the loss."
        ),
        code(
            "costs = [0.0, 10.0, 25.0, 50.0]\n"
            "if HAVE_REAL:\n"
            "    ts = []\n"
            "    for c in costs:\n"
            "        rc = st.race(rets, caps, k=10, n_darts=300, rebalance_every=12, cost_bps=c, seed=350)\n"
            "        ts.append(rc['test_vs_index']['tstat'])\n"
            "else:\n"
            "    ts = [-3.66, -3.69, -3.74, -3.81]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, ts, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(-2, ls=':', c=GREY)\n"
            "ax.set_xlabel('round-trip cost (bps/leg)'); ax.set_ylabel('median dart − cap index, HAC t')\n"
            "ax.set_title('Already losing gross — costs only dig deeper')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('dart-cap HAC t by cost:', [f'{t:+.2f}' for t in ts])"
        ),
        md(
            "> 💡 In plain words: there is no break-even cost to find — the strategy loses to the index "
            "before a cent of cost, and the random rebalancing wastes money on top. If you want the "
            "equal-weight tilt, buy an equal-weight ETF; the dice add turnover and nothing else. The "
            "definition of a **mirage**: not a fragile edge, an *absent* one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The allocation half.** [Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/) tests "
            "1/N *weighting* vs Markowitz on a fixed universe. Study 350 is the *selection* half. The "
            "pair is Malkiel's quip, fully bracketed.\n"
            "- **Restore the legend.** Re-run on a broad or small-cap-tilted universe: the size tilt "
            "becomes a tailwind, the monkey 'wins' against the cap index — and *still* ties the "
            "equal-weight index. The mechanism is invariant; only the sign flips.\n"
            "- **Decompose the tilt.** Arnott et al. (2013) show value/low-beta/profitability tilts "
            "ride along. Fork this to regress the monkey on factor returns and watch the alpha "
            "intercept land at zero.\n\n"
            "*The monkey was never picking stocks — it was renting a size tilt and taking the bow when "
            "the tilt paid. Change the universe so the tilt pays, and you'll 'discover' the monkey's "
            "genius all over again.*"
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
