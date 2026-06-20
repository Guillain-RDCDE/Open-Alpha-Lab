"""Generate the two narrative notebooks for Study 344 (Backtest-Overfitting).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-SPY cell reads the cached parquet
under ../_cache/ if present and otherwise quotes the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs end-to-end for any reader.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    nd=1260, n_trials=1000, n_splits=10,
    # synthetic null hero
    null_fp="2abc7518fe78",
    null_champ="f2_s47", null_is=0.88, null_oos=-0.49, null_oos_t=-0.77,
    null_sr0=3.26, null_dsr=0.00, null_pbo=0.68,
    # N-sweep (IS-split champion + expected-max bar)
    sweep_n=[10, 50, 200, 1000],
    sweep_is=[0.65, 0.88, 0.88, 0.88],
    sweep_oos=[-0.37, -0.49, -0.49, -0.49],
    sweep_sr0=[1.57, 2.28, 2.77, 3.26],
    # positive control (real but un-timeable drift)
    ctrl_bh_sharpe=0.92, ctrl_bh_dsr=0.98,
    ctrl_champ_is=1.19, ctrl_champ_oos=0.03, ctrl_dsr=0.00, ctrl_pbo=0.80,
    # real SPY worked example
    spy_window="2000-01 → 2026-05", spy_fp="022e57e68917", spy_days=6641,
    spy_champ="f10_s188", spy_is=0.68, spy_oos=0.81, spy_oos_t=3.10,
    spy_dsr=0.00, spy_pbo=0.60, spy_bh_sharpe=0.51, spy_bh_dsr=0.996,
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

from backtest_overfitting import data, strategy as st

ND, N_TRIALS, N_SPLITS = 1260, 1000, 10

def _have_spy():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE, "SPY"))

HAVE_REAL = _have_spy()
print("SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Backtest-Overfitting — why every backtest is a 10 and every live run is a 3 📉\n"
            "### How searching hard enough turns pure noise into a 'great strategy'\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_grid--search_fake_an_edge%3F: Confirmed](https://img.shields.io/badge/Does_grid--search_fake_an_edge%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Every backtest you're ever shown looks gorgeous. Then real money goes in and the magic "
            "evaporates. This isn't bad luck — it's *arithmetic*. If you try enough strategy settings "
            "on the same data, one of them is **guaranteed** to look brilliant, even when there is "
            "nothing there. We'll prove it on a market we built to have **zero edge**, then show the "
            "two tools that catch the trap red-handed.\n\n"
            "> 📓 **This is the plain-language layer.** Want the Deflated Sharpe Ratio formula and the "
            "PBO cross-validation? That's the companion, "
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
            "| Why does every backtest look amazing? | **Because you went looking.** Try 1,000 "
            "settings on one dataset and the luckiest will post a beautiful Sharpe — *by chance*. |\n"
            "| We tested it on a market with **no edge**. What happened? | The best of 1,000 rules "
            f"showed an in-sample Sharpe of **{R['null_is']:.2f}** — then **{R['null_oos']:+.2f}** "
            "out-of-sample. The 'strategy' was noise wearing a costume. |\n"
            "| Can a tool catch it? | **Yes — two of them.** The *Deflated Sharpe Ratio* haircuts the "
            f"Sharpe for how many things you tried (it dropped to ~{R['null_dsr']:.0f}). The "
            f"*Probability of Overfitting* (PBO) came out **{R['null_pbo']:.0%}** — your winner is a "
            "loser more often than a coin flip. |\n"
            "| So is there anything to trade here? | **No.** This is a study about *method*, not a "
            "signal. The lesson is the product. |\n\n"
            "> A backtest is a *hypothesis you already peeked at*. The more you peeked, the less the "
            "pretty number means — and there's a formula for exactly how much less."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Quants Marcos López de Prado and David Bailey put it bluntly: most published backtests "
            "are **statistically meaningless**, because the researcher tried many configurations and "
            "reported only the best one — without saying how many they tried. Their 2014 paper has a "
            "title that says it all: *Pseudo-Mathematics and Financial Charlatanism*. The strong "
            "version of the claim: **with enough trials you can hit any target Sharpe on random "
            "data**, so a backtest's Sharpe — on its own — tells you almost nothing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is the single most expensive mistake in quantitative investing. It's why the 'factor "
            "zoo' has hundreds of published anomalies and so few survive replication; it's why a fund's "
            "backtested Sharpe of 2 turns into a live Sharpe of 0.3; it's why your own weekend strategy "
            "looked unstoppable and then bled. If we can *measure* how much a backtest was inflated by "
            "searching, we can tell a real edge from a mirage **before** we fund it. That's the payload."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We rig the experiment so the answer is *known*:\n\n"
            "1. **Build a market with no edge.** A pure coin-flip price path (a 'random walk'). There "
            "is, provably, no timing rule that works on it.\n"
            "2. **Unleash a parameter zoo.** Grid-search 1,000 moving-average crossover rules — the "
            "bread-and-butter of retail backtesting — and keep the best one *in-sample*.\n"
            "3. **Show it live.** Run that champion on data it never saw (out-of-sample).\n"
            "4. **Score the trap.** The **Deflated Sharpe Ratio** asks: is the best Sharpe better than "
            "what pure luck would hand you after 1,000 tries? The **PBO** asks: how often is the "
            "in-sample winner an out-of-sample loser?\n\n"
            "If grid-searching is honest, the champion should keep working live. Spoiler: on a tape "
            "with no edge, it can't — and the tools say so."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the gorgeous backtest that is built from nothing.** Here's the in-sample equity "
            "curve of the *best* of 1,000 crossover rules on our no-edge market — and then what the "
            "very same rule does the moment it goes live."
        ),
        code(
            "prices, _ = data.synthetic_prices(n_days=ND, edge=0.0, seed=344)\n"
            "mat = st.sweep(prices, n_trials=N_TRIALS, cost_bps=0.0)\n"
            "champ = st.in_sample_champion(mat, frac=0.5)\n"
            "col = champ['champion']\n"
            "cut = champ['is_n']\n"
            "eq = (1 + mat[col]).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(eq.values[:cut], color=GREEN, lw=2, label=f'in-sample (Sharpe {champ[\"is_sharpe\"]:.2f})')\n"
            "ax.plot(range(cut, len(eq)), eq.values[cut:], color=RED, lw=2,\n"
            "        label=f'out-of-sample (Sharpe {champ[\"oos_sharpe\"]:+.2f})')\n"
            "ax.axvline(cut, ls='--', c=GREY); ax.set_xlabel('trading day'); ax.set_ylabel('growth of $1')\n"
            "ax.set_title('The luckiest of 1,000 rules on a market with NO edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Best-of-{mat.shape[1]} champion {col}: IS Sharpe {champ[\"is_sharpe\"]:.2f} -> OOS Sharpe {champ[\"oos_sharpe\"]:+.2f}')"
        ),
        md(
            f"The green half is a textbook winner — Sharpe **{R['null_is']:.2f}**, a curve any pitch "
            f"deck would love. The red half is the same rule, live: Sharpe **{R['null_oos']:+.2f}**. "
            "Nothing changed about the rule; we just stopped peeking. **There was never anything there "
            "— we manufactured the beauty by searching.**"
        ),
        md(
            "**Second: the more you try, the prettier the lie.** Watch the bar that *pure luck* clears "
            "(the expected best Sharpe under no skill) climb as we try more rules. A Sharpe that would "
            "impress at 10 trials is *unremarkable* at 1,000."
        ),
        code(
            "ns = [10, 50, 200, 1000]\n"
            "is_s, sr0s = [], []\n"
            "for n in ns:\n"
            "    m = st.sweep(prices, n_trials=n, cost_bps=0.0)\n"
            "    c = st.in_sample_champion(m, frac=0.5)\n"
            "    d = st.deflated_sharpe_ratio(m[c['champion']], n_trials=m.shape[1])\n"
            "    is_s.append(c['is_sharpe']); sr0s.append(d['sr0'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "x = range(len(ns))\n"
            "ax.bar([i-0.2 for i in x], is_s, width=0.4, color=GREEN, label='best backtest Sharpe (per-day annualised view)')\n"
            "ax.bar([i+0.2 for i in x], [s/np.sqrt(252)*np.sqrt(252) for s in sr0s], width=0.4, color=GREY,\n"
            "       label='Sharpe luck alone clears (expected max)')\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels([str(n) for n in ns])\n"
            "ax.set_xlabel('number of rules tried'); ax.set_ylabel('Sharpe (per-period units)')\n"
            "ax.set_title('The luck bar rises with every extra rule you try'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('expected-max Sharpe (luck) by N:', [f'{s:.2f}' for s in sr0s])"
        ),
        md(
            f"The grey bar — the Sharpe you'd get **from luck alone** — climbs from "
            f"~{R['sweep_sr0'][0]:.1f} at 10 trials to ~{R['sweep_sr0'][-1]:.1f} at 1,000. That is "
            "why a Sharpe number is meaningless without its trial count. This is the entire engine of "
            "the disease, in one chart."
        ),
        md(
            "**Third: the catch — does the trap also fool a single honest idea?** No. The tools punish "
            "*searching*, not *having an edge*. On a market with a small **real** trend (that you'd "
            "earn just by buying and holding), the honest buy-and-hold keeps a clean bill of health — "
            "while the grid-searched timing rule is still flagged as overfit."
        ),
        code(
            "pe, _ = data.synthetic_prices(n_days=ND, edge=0.0003, seed=344)\n"
            "bh = st.buy_and_hold(pe)\n"
            "dbh = st.deflated_sharpe_ratio(bh, n_trials=1)\n"
            "mat_e = st.sweep(pe, n_trials=N_TRIALS, cost_bps=0.0)\n"
            "ch_e = st.in_sample_champion(mat_e, frac=0.5)\n"
            "d_e = st.deflated_sharpe_ratio(mat_e[ch_e['champion']], n_trials=mat_e.shape[1])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar(['Buy & hold\\n(1 honest idea)', 'Grid-searched\\ntiming champion'],\n"
            "       [dbh['dsr'], d_e['dsr']], color=[GREEN, RED], width=0.5)\n"
            "ax.set_ylim(0, 1.05); ax.set_ylabel('Deflated Sharpe (1 = trustworthy, 0 = luck)')\n"
            "ax.set_title('The tools spare an honest idea and bust the over-searched one')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy-hold DSR {dbh[\"dsr\"]:.2f}  |  grid-searched champion DSR {d_e[\"dsr\"]:.2f}')"
        ),
        md(
            f"Buy-and-hold's Deflated Sharpe is **~{R['ctrl_bh_dsr']:.2f}** (trustworthy) because it "
            "was one honest hypothesis, not the survivor of a search. The timing champion — riding the "
            f"*same* real trend — deflates to **~{R['ctrl_dsr']:.0f}**, because its in-sample shine was "
            "bought with 1,000 tries. **The trap is in the *selection*, not in whether an edge exists.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** By design there is no timing edge; the in-sample Sharpe is pure "
            "selection luck and deflates to zero. Nothing real to detect.\n"
            "- **Tradability — Mirage.** A backtest you can't trade — the pretty number is gone the "
            "moment you stop peeking.\n"
            "- **Does grid-searching fake an edge? — Confirmed.** On a market we *know* is empty, a "
            f"1,000-rule sweep built a Sharpe-{R['null_is']:.2f} 'strategy' out of nothing; the "
            f"Deflated Sharpe (~0) and PBO ({R['null_pbo']:.0%}) catch it cold."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There is, deliberately, nothing to trade — that's the whole point. But the *practical* "
            "takeaway is huge: before you risk a cent on any backtest (yours or someone else's), ask "
            "**two questions**. (1) *How many configurations did you try?* — then deflate the Sharpe "
            "for it. (2) *What's the PBO?* — if your in-sample winner is a coin-flip-or-worse out of "
            "sample, you don't have a strategy, you have a fitted ghost. Most blow-ups would have been "
            "caught by either question. We run the real SPY version of exactly this in the "
            "[quants notebook](02_for_the_quants.ipynb) — and even there, where the champion *looked* "
            "fine live, the tools still flash red, because it was riding the market, not timing it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Add costs and slippage** to the sweep and watch even more 'edges' die — overfitting "
            "and cost-blindness are the two horsemen of the dead backtest.\n"
            "- **Walk-forward instead of one split.** Re-optimise on a rolling window and you'll see "
            "the champion *change every period* — a tell that it was never stable.\n"
            "- **Pre-register your hypothesis.** The cleanest defence: decide the rule *before* you "
            "touch the data, so there's nothing to deflate.\n\n"
            "*Think your favourite backtest is the real thing? Fork this, paste in your strategy's "
            "trial returns, and read off its Deflated Sharpe and PBO. If the number survives, you've "
            "got something. If it doesn't, you just saved yourself a live experiment.*"
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
            "# Backtest-Overfitting — a quantitative teardown 🔬\n"
            "### Grid-search on a known-null tape · Deflated Sharpe Ratio · PBO via CSCV · "
            "expected-max-Sharpe under selection · a real SPY worked example\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_grid--search_fake_an_edge%3F: Confirmed](https://img.shields.io/badge/Does_grid--search_fake_an_edge%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We grid-search 1,000 long/flat "
            "crossover rules on a martingale tape (where the timing edge is provably zero), then put "
            "Bailey & López de Prado's two numbers on the result: the **Deflated Sharpe Ratio** and "
            "the **Probability of Backtest Overfitting** (CSCV).\n\n"
            "> ⚠️ **Not investment advice.** Synthetic core is deterministic & offline; one real "
            "worked example uses SPY total-return daily (as-of 2026-05-31). Methods in "
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
            f"| **Signal** | `NONE` | On a known-null tape the best of {R['n_trials']} rules posts "
            f"IS Sharpe **{R['null_is']:.2f}** → OOS **{R['null_oos']:+.2f}** (OOS HAC *t* "
            f"{R['null_oos_t']:+.2f}); Deflated Sharpe **≈{R['null_dsr']:.2f}**. No timing edge exists "
            "and none is detected. |\n"
            f"| **Tradability** | `MIRAGE` | The in-sample Sharpe is a selection artefact; it does not "
            "survive the out-of-sample split, by construction. |\n"
            f"| **Does grid-search fake an edge?** | `CONFIRMED` | PBO (CSCV) = **{R['null_pbo']:.2f}** "
            f"on the null tape; the expected-max Sharpe bar rises {R['sweep_sr0'][0]:.1f}→"
            f"{R['sweep_sr0'][-1]:.1f} as N goes {R['sweep_n'][0]}→{R['sweep_n'][-1]}; and the "
            f"positive control spares honest buy-and-hold (DSR {R['ctrl_bh_dsr']:.2f}). |\n\n"
            "> 💡 In plain words: the disease is *selection under multiple testing*. The DSR corrects "
            "the best Sharpe for the number of trials and the return moments; PBO measures how often "
            "the in-sample winner is an out-of-sample loser. Both are designed to be un-fooled by a "
            "single lucky outcome — which is exactly why they still flag the real SPY champion."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a researcher evaluate $N$ strategy configurations on one sample of length $T$, with "
            "per-period Sharpe estimates $\\widehat{SR}_1,\\dots,\\widehat{SR}_N$, and report "
            "$\\widehat{SR}^* = \\max_i \\widehat{SR}_i$.\n\n"
            "- **H₀ (no skill).** Under the null, the $\\widehat{SR}_i$ are mean-zero; yet "
            "$\\mathbb{E}[\\widehat{SR}^*]$ grows like $\\sqrt{\\operatorname{Var}}\\,$ times an "
            "extreme-value term in $N$ — so a large $\\widehat{SR}^*$ is *expected* purely from "
            "searching.\n"
            "- **H₁ (overfitting → OOS collapse).** The in-sample champion's out-of-sample rank is "
            "no better than median: $\\Pr[\\text{IS champion below OOS median}] \\ge 0.5$ (this is "
            "PBO).\n"
            "- **H₂ (the diagnostics discriminate).** A *single, un-selected* hypothesis with a real "
            "edge keeps a high Deflated Sharpe, while the grid champion does not.\n\n"
            "We confirm **H₀ and H₁ on the null tape** and **H₂ via the positive control** — so the "
            "claim survives at full strength: searching manufactures a Sharpe, and the trap is "
            "measurable."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Harvey, Liu & Zhu (2016) argue the published cross-section of factors is so multiple-"
            "tested that the conventional *t* > 2 hurdle should be raised to ~3. Bailey & López de "
            "Prado give the portfolio-level analogue: a Sharpe reported without its trial count is "
            "*unfalsifiable*. The desk's contribution here is to run the demonstration on a tape with "
            "**zero ground-truth edge**, so the entire in-sample spread is overfitting we can attribute "
            "exactly — a positive control for the desk's own inference machinery."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Tape.** A geometric random walk, `edge=0` (null) and `edge=0.0003/day` (control), "
            f"{R['nd']} trading days, seed 344. One real tape: SPY total-return daily, 2000–2026.\n"
            "- **The zoo.** {N} long/flat MA crossovers `(fast, slow)`, `fast<slow`. One execution "
            "lag (signal at close of *t* earns *t+1*; a single `shift`). Costs one-way × NAV (here "
            "0 bps for the diagnostic; cost sweeps die faster).\n"
            "- **Selection.** IS = first 50%, OOS = last 50%; champion = max IS Sharpe.\n"
            "- **DSR.** Best Sharpe vs the expected maximum under the null over $N$ trials, with the "
            "Lo (2002) skew/kurtosis correction — `strategy.deflated_sharpe_ratio`.\n"
            "- **PBO.** CSCV with $S$ contiguous blocks; over all $\\binom{S}{S/2}$ symmetric "
            "partitions, the fraction where the IS champion lands below the OOS median — "
            "`strategy.pbo_cscv`.\n"
            "- **Inference.** Newey-West HAC *t* on the champion's OOS mean; circular block-bootstrap "
            "CI. The bar for a *real* selection edge: HAC *t* ≥ 2 attributable to timing, not beta."
            .replace("{N}", str(R["n_trials"]))
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The known-null tape — a Sharpe built from nothing\n\n"
            "The best of 1,000 rules in-sample, then live. The OOS HAC *t* confirms the OOS mean is "
            "indistinguishable from zero — correctly, since there is no edge."
        ),
        code(
            "prices, truth = data.synthetic_prices(n_days=ND, edge=0.0, seed=344)\n"
            "print('null tape fingerprint:', data.fingerprint(prices))\n"
            "mat = st.sweep(prices, n_trials=N_TRIALS, cost_bps=0.0)\n"
            "champ = st.in_sample_champion(mat, frac=0.5)\n"
            "t = st.hac_tstat(champ['oos_returns'])\n"
            "lo, hi = st.block_bootstrap_ci(champ['oos_returns'], n_boot=1000, seed=1)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "is_sr = mat.apply(st.sharpe)\n"
            "ax.hist(is_sr.values, bins=40, color=GREY, alpha=.8)\n"
            "ax.axvline(champ['is_sharpe'], c=RED, lw=2.5, label=f'selected champion ({champ[\"is_sharpe\"]:.2f})')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('full-sample Sharpe of each of 1,000 rules'); ax.set_ylabel('count')\n"
            "ax.set_title('1,000 noise rules — and the one we would have shipped'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'champion {champ[\"champion\"]}: IS {champ[\"is_sharpe\"]:.2f} -> OOS {champ[\"oos_sharpe\"]:+.2f}')\n"
            "print(f'OOS HAC t = {t[\"tstat\"]:+.2f} (n={t[\"n\"]}); OOS mean 95% block-bootstrap CI '\n"
            "      f'[{lo*1e4:+.1f}, {hi*1e4:+.1f}] bps/day')"
        ),
        md(
            f"> 💡 In plain words: the champion's OOS HAC *t* is **{R['null_oos_t']:+.2f}** — squarely "
            "inside the noise band, and the bootstrap CI straddles zero. The in-sample Sharpe of "
            f"**{R['null_is']:.2f}** was the right tail of a distribution centred on zero; we selected "
            "the tail and called it skill."
        ),
        md(
            "### 4b · The Deflated Sharpe Ratio — haircut for the trial count\n\n"
            "The expected maximum Sharpe under the null grows with $N$ (extreme-value asymptotics). "
            "The DSR re-expresses the best Sharpe as $\\Phi$ of its distance above that bar."
        ),
        code(
            "ns = [10, 50, 200, 1000]\n"
            "sr0s, dsrs, is_sh = [], [], []\n"
            "for n in ns:\n"
            "    m = st.sweep(prices, n_trials=n, cost_bps=0.0)\n"
            "    c = st.in_sample_champion(m, frac=0.5)\n"
            "    d = st.deflated_sharpe_ratio(m[c['champion']], n_trials=m.shape[1])\n"
            "    sr0s.append(d['sr0']); dsrs.append(d['dsr']); is_sh.append(d['sr_per_period'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(ns, sr0s, 'o-', c=GREY, lw=2, label='expected max Sharpe (null)')\n"
            "a1.plot(ns, is_sh, 's-', c=RED, lw=2, label='champion per-period Sharpe')\n"
            "a1.set_xscale('log'); a1.set_xlabel('N trials'); a1.set_ylabel('per-period Sharpe')\n"
            "a1.set_title('The luck bar (SR0) towers over the champion'); a1.legend()\n"
            "a2.plot(ns, dsrs, 'o-', c=RED, lw=2); a2.axhline(0.5, ls='--', c=GREY)\n"
            "a2.set_xscale('log'); a2.set_ylim(-0.02, 1.02)\n"
            "a2.set_xlabel('N trials'); a2.set_ylabel('Deflated Sharpe (probability)')\n"
            "a2.set_title('DSR collapses to 0 — the champion is luck')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SR0 by N :', [f'{s:.2f}' for s in sr0s])\n"
            "print('DSR by N :', [f'{d:.3f}' for d in dsrs])"
        ),
        md(
            f"> 💡 In plain words: the expected-max bar climbs {R['sweep_sr0'][0]:.1f} → "
            f"{R['sweep_sr0'][-1]:.1f} (per-period units shown; the annualised gap is larger) while "
            "the champion's per-period Sharpe sits far below it. The DSR — the probability the champion "
            "beats luck — is pinned near **0** at every N. A Sharpe with no trial count is uninterpretable."
        ),
        md(
            "### 4c · PBO via CSCV — the in-sample winner is an out-of-sample loser\n\n"
            "Over all symmetric block partitions, how often does the IS champion fall below the OOS "
            "median? That frequency is the Probability of Backtest Overfitting."
        ),
        code(
            "pbo = st.pbo_cscv(mat, n_splits=N_SPLITS)\n"
            "lam = np.array(pbo['lambdas'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "ax.hist(lam, bins=25, color=GREY, alpha=.85)\n"
            "ax.axvline(0, c=RED, lw=2.5, label='OOS median (λ=0)')\n"
            "ax.set_xlabel('logit of OOS relative rank of the IS champion (λ)')\n"
            "ax.set_ylabel('partitions')\n"
            "ax.set_title(f'PBO = {pbo[\"pbo\"]:.2f}  —  IS champion below OOS median that often'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'PBO = {pbo[\"pbo\"]:.3f} over {pbo[\"n_combos\"]} symmetric partitions; '\n"
            "      f'median OOS rank of champion = {pbo[\"median_oos_rank\"]:.0f}/{pbo[\"n_trials\"]}')"
        ),
        md(
            f"> 💡 In plain words: PBO = **{R['null_pbo']:.2f}**. More than half the time, the rule you "
            "would have *selected* from the in-sample tournament lands in the bottom half live. "
            "Selection here isn't merely useless — it is mildly *anti-predictive*, the fingerprint of "
            "overfitting on a no-edge tape."
        ),
        md(
            "### 4d · The positive control — the diagnostics discriminate\n\n"
            "Plant a real but un-timeable drift. The honest single hypothesis (buy-and-hold) must keep "
            "a high DSR; the grid-searched timing champion must not."
        ),
        code(
            "pe, _ = data.synthetic_prices(n_days=ND, edge=0.0003, seed=344)\n"
            "bh = st.buy_and_hold(pe); dbh = st.deflated_sharpe_ratio(bh, n_trials=1)\n"
            "mat_e = st.sweep(pe, n_trials=N_TRIALS, cost_bps=0.0)\n"
            "ch_e = st.in_sample_champion(mat_e, frac=0.5)\n"
            "d_e = st.deflated_sharpe_ratio(mat_e[ch_e['champion']], n_trials=mat_e.shape[1])\n"
            "p_e = st.pbo_cscv(mat_e, n_splits=N_SPLITS)\n"
            "print(f'buy-hold (N=1)            : Sharpe {st.sharpe(bh):.2f}  DSR {dbh[\"dsr\"]:.3f}')\n"
            "print(f'grid timing champion (N={mat_e.shape[1]}): IS {ch_e[\"is_sharpe\"]:.2f} -> '\n"
            "      f'OOS {ch_e[\"oos_sharpe\"]:+.2f}  DSR {d_e[\"dsr\"]:.3f}  PBO {p_e[\"pbo\"]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: buy-and-hold's DSR is **~{R['ctrl_bh_dsr']:.2f}** — the real drift is "
            "trusted *because it was one un-selected hypothesis*. The timing champion riding the same "
            f"drift collapses (IS {R['ctrl_champ_is']:.2f} → OOS {R['ctrl_champ_oos']:+.2f}, DSR "
            f"~{R['ctrl_dsr']:.0f}, PBO {R['ctrl_pbo']:.2f}). **The trap is selection, not the "
            "existence of an edge** — the machinery passes its own positive control."
        ),
        md(
            "### 4e · The real worked example — SPY 2000–2026\n\n"
            "The same grid-search on real SPY (IS 2000–2013, OOS 2013–2026). Here the champion's OOS "
            "Sharpe *held up* — and that is precisely the danger the tools are built for."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = data.load_real('SPY', fetch=False)\n"
            "    px = px[px.index < pd.Timestamp.today().normalize().to_period('M').to_timestamp()]\n"
            "    print('SPY', px.index[0].date(), '->', px.index[-1].date(), 'fp', data.fingerprint(px))\n"
            "    m = st.sweep(px, n_trials=N_TRIALS, cost_bps=0.0)\n"
            "    c = st.in_sample_champion(m, frac=0.5)\n"
            "    d = st.deflated_sharpe_ratio(m[c['champion']], n_trials=m.shape[1])\n"
            "    p = st.pbo_cscv(m, n_splits=N_SPLITS)\n"
            "    t = st.hac_tstat(c['oos_returns'])\n"
            "    bh = st.buy_and_hold(px); dbh = st.deflated_sharpe_ratio(bh, n_trials=1)\n"
            "    is_sh, oos_sh, oos_t = c['is_sharpe'], c['oos_sharpe'], t['tstat']\n"
            "    dsr, pbo_v, bh_sh, bh_dsr = d['dsr'], p['pbo'], st.sharpe(bh), dbh['dsr']\n"
            "    banner = 'REAL SPY TAPE (cached)'\n"
            "else:\n"
            f"    is_sh, oos_sh, oos_t = {R['spy_is']}, {R['spy_oos']}, {R['spy_oos_t']}\n"
            f"    dsr, pbo_v, bh_sh, bh_dsr = {R['spy_dsr']}, {R['spy_pbo']}, {R['spy_bh_sharpe']}, {R['spy_bh_dsr']}\n"
            "    banner = 'OFFLINE — frozen SPY numbers from docs/results.md'\n"
            "print(banner)\n"
            "print(f'champion: IS Sharpe {is_sh:.2f} -> OOS Sharpe {oos_sh:+.2f}  (OOS HAC t {oos_t:+.2f})')\n"
            "print(f'Deflated Sharpe {dsr:.3f}   PBO {pbo_v:.2f}')\n"
            "print(f'SPY buy-and-hold Sharpe {bh_sh:.2f}, DSR (N=1) {bh_dsr:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the SPY champion's OOS Sharpe **{R['spy_oos']:+.2f}** and HAC *t* "
            f"**{R['spy_oos_t']:+.2f}** look like a win — but it is *market beta* (a long-biased rule "
            f"rode the 2013–2026 bull, like buy-and-hold). The diagnostics refuse to be fooled: DSR "
            f"**≈{R['spy_dsr']:.2f}** (the IS Sharpe never cleared the {R['null_sr0']:.1f}-from-1,000 "
            f"bar) and PBO **{R['spy_pbo']:.2f}** (selection worse than a coin). Honest buy-and-hold "
            f"posts DSR **{R['spy_bh_dsr']:.3f}**. One good OOS window is not evidence the *selection* "
            "generalises — which is the entire lesson."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — on the null tape the champion's OOS HAC *t* is {R['null_oos_t']:+.2f} "
            f"(insignificant) and DSR ≈ {R['null_dsr']:.2f}; on SPY the apparent OOS success is beta, "
            "not selection skill, and DSR is still ≈ 0. No robust *t* ≥ 2 attributable to timing "
            "exists anywhere.\n"
            f"- **Tradability `MIRAGE`** — the in-sample Sharpe is a selection artefact; out of sample "
            "it is gone (null) or repackaged market exposure (SPY).\n"
            f"- **Does grid-search fake an edge? `CONFIRMED`** — IS Sharpe {R['null_is']:.2f} from "
            f"nothing; expected-max bar rises {R['sweep_sr0'][0]:.1f}→{R['sweep_sr0'][-1]:.1f} with N; "
            f"PBO {R['null_pbo']:.2f}; and the positive control spares honest buy-and-hold "
            f"(DSR {R['ctrl_bh_dsr']:.2f}). Report DSR and PBO, never a naked Sharpe."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the methodology dividend\n\n"
            "There is no edge to trade; the payoff is procedural. Two rules the desk now enforces "
            "everywhere because of this study:"
        ),
        code(
            "rows = [\n"
            "  ('Naked Sharpe (what gets pitched)', 'meaningless without N'),\n"
            f"  ('Deflated Sharpe @ N={R['n_trials']}', 'corrects for trials + moments'),\n"
            "  ('PBO via CSCV', 'P[IS winner is OOS loser]'),\n"
            "  ('OOS HAC t', 'is the live mean even non-zero?'),\n"
            "]\n"
            "import pandas as pd\n"
            "print(pd.DataFrame(rows, columns=['diagnostic', 'what it answers']).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: a backtest is a hypothesis you already peeked at. Disclose the trial "
            "count and deflate; cross-validate the *selection itself* with PBO; and never confuse a "
            "lucky out-of-sample window with a generalising edge. Do those three and most of the "
            "industry's dead strategies never get funded. That is the trade."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Walk-forward & CPCV.** Replace the single split with Combinatorial Purged "
            "Cross-Validation (López de Prado, *Advances in Financial Machine Learning*) to handle "
            "label leakage in overlapping windows.\n"
            "- **Cost-aware deflation.** Re-run the sweep at realistic one-way costs; the DSR falls "
            "further and the surviving-strategy count craters — overfitting and costs compound.\n"
            "- **Your own factor zoo.** Feed an (T×N) matrix of *your* candidate signals' returns into "
            "`deflated_sharpe_ratio` and `pbo_cscv` and read off whether selection added anything.\n\n"
            "*The uncomfortable corollary: the prettier your backtest, the more suspicious you should "
            "be — beauty is what selection optimises. Fork this, run your strategy through DSR and PBO, "
            "and let the numbers, not the equity curve, decide.*"
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
