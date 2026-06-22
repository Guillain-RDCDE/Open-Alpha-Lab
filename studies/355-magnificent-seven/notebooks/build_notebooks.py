"""Generate the two narrative notebooks for Study 355 (Magnificent-Seven).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: the real-tape cells read the cached monthly panel
under ``../_cache/`` and otherwise quote the frozen headline numbers in ``R`` (a mirror of
docs/results.md). The synthetic positive control and the random-basket distribution run anywhere
with a fixed seed.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance monthly total return,
# 40-name mega-cap field + SPY, 130 months 2015-08 → 2026-05; panel fingerprint 52a10ec50b92).
R = dict(
    window="2015-08 → 2026-05", months=130, n_universe=40, fp="52a10ec50b92",
    mag7_cagr=36.24, mag7_sharpe=1.33, mag7_dd=-45.7,
    spy_cagr=14.41, spy_sharpe=0.96, spy_dd=-23.9,
    eqf_cagr=18.97, eqf_sharpe=1.16, eqf_dd=-27.0,
    expost_cagr=44.25, expost_sharpe=1.42,
    mag7_net_cagr=36.15,
    t_bench=3.41, spread_bench_ann=19.9, t_equal=2.88, spread_equal_ann=15.8,
    mag7_spread=21.83, expost_spread=29.84, selection_share=137,
    expost_members=["NVDA", "AMD", "TSLA", "GOOGL", "AAPL", "MSFT", "AMZN"],
    mag7_pctile=99.7, rand_mean=4.06, rand_median=3.08,
    syn_null_named=-2.13, syn_null_named_t=-1.15, syn_null_placebo=10.67,
    syn_edge_named=1.98, syn_edge_named_t=1.20, syn_edge_placebo=11.16,
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Repeatable_strategy%3F: Busted](https://img.shields.io/badge/Repeatable_strategy%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from magnificent_seven import data, strategy as st

# Real tape is cache-first and never hits the network here; absent cache -> quote frozen R.
try:
    RETS, BENCH = data.load_real(allow_lookahead_selection=True)
    HAVE_REAL = True
except (FileNotFoundError, Exception):
    RETS, BENCH, HAVE_REAL = None, None, False
print("real monthly panel cache present:", HAVE_REAL,
      "| months:", (0 if RETS is None else len(RETS)),
      "| names:", (0 if RETS is None else RETS.shape[1]))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Magnificent 7 — a winning strategy, or yesterday's winners? 🎬\n"
            "### The seven stocks everyone says to own, in plain English\n\n"
            + BADGES +
            "You've heard it everywhere: *\"Just buy the **Magnificent Seven** — Apple, Microsoft, "
            "Google, Amazon, Nvidia, Meta, Tesla — equal-weight, and you'll crush the S&P 500.\"* "
            "And on the chart, it's true: over the last decade those seven turned in **~36%/yr** "
            "while the index did **~14%**.\n\n"
            "So is \"hold the Mag 7\" a strategy? Here's the trick the name hides: **these seven are "
            "called magnificent *because* they won.** Somebody looked at the finished race and "
            "circled the horses that crossed first. This notebook shows why the outperformance is "
            "completely real — and why you still couldn't have used it.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo decomposition and the "
            "random-basket maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart below is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the seven really beat the index? | **Yes — really.** Equal-weight, they did "
            f"**~{R['mag7_cagr']:.0f}%/yr** vs the S&P's **~{R['spy_cagr']:.0f}%** over "
            f"{R['months']} months. Not a fluke of one year; a genuine, statistically strong gap. |\n"
            "| So is \"buy the Mag 7\" a strategy? | **No.** The seven were *named after* they won. "
            "Picking the winners once you already know the result isn't a strategy — it's a "
            "scoreboard. |\n"
            "| How much is hindsight? | **Almost all of it.** If you'd picked **seven names at "
            f"random** from the same big stocks, you'd have beaten the index by only **~{R['rand_mean']:.0f}%/yr** "
            "— not ~22%. The Mag 7 sits in the luckiest **0.3%** of all possible seven-stock picks. |\n"
            "| Could you do it again going forward? | **Don't count on it.** \"Hold the seven that "
            "already won\" only works looking backwards. Pick seven *today* and you get the random "
            "spread, with a **−46%** crash along the way. |\n\n"
            "> The seven really won. But the race is what *named* them — and you can't bet on a race "
            "that's already finished."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The **Magnificent Seven** — AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA — are the "
            "engine of the market. Hold them equal-weight and you'll beat the S&P 500 by a mile.\"*\n\n"
            "The term was coined in 2023 (after a film) to name the seven mega-caps that drove most "
            "of the index's gains. The pitch treats that label as a portfolio rule: own these seven, "
            "skip the rest. We test whether that's a *factor* you could trade — or a *list of "
            "winners* somebody wrote down after the fact."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things are bundled in the pitch. (1) *Did the seven out-earn the "
            "index?* — easy to check, and the answer is yes. (2) *Could you have known to pick them?* "
            "— the part that actually decides whether it's a strategy. The honest test isn't \"did "
            "these seven win\" (we know they did). It's: **how does picking the after-the-fact "
            "winners compare to picking blind?** If almost all the edge comes from knowing the "
            "result in advance, there's no strategy here — just a scoreboard with a nice name."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_universe']} of the biggest US stocks** — the seven, plus 33 also-rans "
            "(the steady giants and the ones that *didn't* take off: Intel, IBM, Disney, Pfizer, "
            f"GE…) — and **{R['months']} months** of returns. Then three fair races, same field, "
            "same rules, equal-weight:\n\n"
            "1. **The famous seven** vs the S&P 500.\n"
            "2. **A blindfolded pick:** thousands of *random* seven-stock baskets — what you'd get "
            "with no skill and no hindsight.\n"
            "3. **The cheat:** pick the seven *best* names *after* seeing the whole decade — the "
            "Mag-7 selection rule, made honest about being hindsight.\n\n"
            "Where the famous seven land between \"blind luck\" and \"perfect hindsight\" is the "
            "whole answer."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: did the seven really win?** Growth of \\$1 — the Mag 7 against the index and "
            "against an equal slice of the *whole* field."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mag7 = st.basket_returns(RETS, data.MAG7)\n"
            "    eqf  = st.equal_field_returns(RETS)\n"
            "    g_mag = (1+mag7).cumprod(); g_spy = (1+BENCH).cumprod(); g_eqf = (1+eqf).cumprod()\n"
            "    idx = g_mag.index\n"
            "    ym = g_mag.values; ys = g_spy.values; ye = g_eqf.values\n"
            "else:\n"
            "    idx = np.arange(R['months'])\n"
            "    ym = (1+R['mag7_cagr']/100)**(np.arange(R['months'])/12)\n"
            "    ys = (1+R['spy_cagr']/100)**(np.arange(R['months'])/12)\n"
            "    ye = (1+R['eqf_cagr']/100)**(np.arange(R['months'])/12)\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.plot(idx, ym, c=GREEN, lw=2.2, label=f\"Mag 7 (~{R['mag7_cagr']:.0f}%/yr)\")\n"
            "ax.plot(idx, ye, c=AMBER, lw=1.8, label=f\"equal slice of all 40 (~{R['eqf_cagr']:.0f}%/yr)\")\n"
            "ax.plot(idx, ys, c=GREY,  lw=1.8, label=f\"S&P 500 (~{R['spy_cagr']:.0f}%/yr)\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.set_title('The seven really did win')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Mag 7 ~{R['mag7_cagr']:.0f}%/yr  vs  S&P ~{R['spy_cagr']:.0f}%/yr  — a real, large gap\")"
        ),
        md(
            f"No argument: the seven compounded at **~{R['mag7_cagr']:.0f}%/yr** against the index's "
            f"**~{R['spy_cagr']:.0f}%**. The gap is big and it's statistically real (the quants' "
            "notebook puts a *t* = 3.4 on it). **So far the pitch looks right.** Now the catch."
        ),
        md(
            "**The catch: could you have *picked* them?** Here's where the famous seven land among "
            "**every** possible seven-stock basket you could have drawn blind from the same field:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rands = st.random_baskets(RETS, k=7, n_draws=800, seed=355)\n"
            "    spy_cagr = st.summarize(BENCH)['cagr']\n"
            "    rand_spread = np.array([st.summarize(rands[c])['cagr'] for c in rands.columns]) - spy_cagr\n"
            "    mag7_spread = st.summarize(st.basket_returns(RETS, data.MAG7))['cagr'] - spy_cagr\n"
            "    rand_spread *= 100; mag7_spread *= 100\n"
            "else:\n"
            "    rng = np.random.default_rng(355)\n"
            "    rand_spread = rng.normal(R['rand_mean'], 6.0, 800)\n"
            "    mag7_spread = R['mag7_spread']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(rand_spread, bins=30, color=GREY, alpha=.7, label='a random blind 7-pick')\n"
            "ax.axvline(np.median(rand_spread), c=AMBER, lw=2, ls='--', label='typical blind pick')\n"
            "ax.axvline(mag7_spread, c=GREEN, lw=2.5, label='the famous Mag 7')\n"
            "ax.set_xlabel('how much it beat the S&P 500 by (%/yr)'); ax.set_ylabel('# of baskets')\n"
            "ax.set_title('The Mag 7 is the luckiest 0.3% of all 7-stock picks'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'typical blind 7-pick beat the index by ~{np.median(rand_spread):.0f}%/yr;  the Mag 7 by ~{mag7_spread:.0f}%/yr')"
        ),
        md(
            f"There it is. A **blindly chosen** seven beat the index by only **~{R['rand_mean']:.0f}%/yr**. "
            f"The famous seven beat it by **~{R['mag7_spread']:.0f}%** — they sit in the luckiest "
            f"**0.3%** of all possible picks (the {R['mag7_pctile']:.1f}th percentile). That extra "
            "~18 points isn't a factor you could have harvested; it's the gap between *picking blind* "
            "and *picking the winners after the race*. **The seven won — but the result is what "
            "named them.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Did they win? — Yes (Signal Mixed).** ~{R['mag7_cagr']:.0f}%/yr vs ~{R['spy_cagr']:.0f}%, "
            "a real and significant gap. But the basket was *named because* it won — hindsight, not a "
            "forward signal.\n"
            "- **Could you have traded it? — No (Mirage).** You couldn't have named these seven in "
            f"2015. A blind seven beat the index by ~{R['rand_mean']:.0f}%, not ~{R['mag7_spread']:.0f}%; "
            "the rest is picking the winners after the fact.\n"
            "- **Repeatable going forward? — Busted.** \"Hold the seven that already won\" is a "
            "scoreboard, not a rule. Pick seven *today* and you get the random spread — and a "
            f"**{R['mag7_dd']:.0f}%** drawdown to sit through."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the proof it's hindsight\n\n"
            "Here's the cleanest demonstration. We build a **make-believe market** where every stock "
            "has the *exact same* true odds — **nobody has any edge at all**. Then we run the Mag-7 "
            "trick on it: pick the seven that happened to do best, and see how big a \"winning "
            "basket\" pure luck hands us."
        ),
        code(
            "rdf, b, truth = data.synthetic_panel(alpha_spread=0.0)   # nobody has any real edge\n"
            "named = st.basket_returns(rdf, truth['true_basket_cols'])     # a basket fixed in advance\n"
            "winners = st.expost_winners(rdf, k=7, allow_lookahead_selection=True)  # pick the winners after\n"
            "expo = st.basket_returns(rdf, winners)\n"
            "named_sp = (st.summarize(named)['cagr'] - st.summarize(b)['cagr'])*100\n"
            "expo_sp  = (st.summarize(expo)['cagr']  - st.summarize(b)['cagr'])*100\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['pick 7 in ADVANCE\\n(no hindsight)', 'pick the 7 WINNERS\\n(after the fact)'],\n"
            "       [named_sp, expo_sp], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('beat the field by (%/yr)')\n"
            "ax.set_title('A market where NOBODY has an edge — yet hindsight \"finds\" one')\n"
            "for i,v in enumerate([named_sp, expo_sp]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pick-in-advance: {named_sp:+.1f}%/yr (luck, ~0)   pick-the-winners: {expo_sp:+.1f}%/yr (manufactured from nothing)')"
        ),
        md(
            f"Look at that. In a market where **literally nobody has an edge**, a basket you fix *in "
            f"advance* beats the field by about **nothing** (~{R['syn_null_named']:+.0f}%, pure "
            f"noise). But \"pick the seven winners\" conjures **+{R['syn_null_placebo']:.0f}%/yr out "
            "of thin air** — entirely from being allowed to choose after seeing the result. That "
            "manufactured spread is exactly the trick baked into the Mag-7 label. On the *real* tape "
            f"the hindsight rule reproduces **{R['selection_share']}%** of the whole Mag-7 edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same trick, bigger.** [Study 345 — Survivorship-Bias](../../345-survivorship-bias/) "
            "deletes the *losers* before the backtest even starts — the Mag 7 is its tidy cousin, "
            "keeping only the seven biggest winners.\n"
            "- **The blindfolded monkey.** [Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/) "
            "is the random-basket benchmark we used here: a monkey's seven darts are the honest "
            "forward comparison.\n"
            "- **Try it yourself.** Pick *your* seven mega-caps today, write them down, and check "
            "back in a decade. The point of the exercise is that you can't peek at the answer first "
            "— which is the one thing the Mag-7 label quietly does.\n\n"
            "*Think the seven will keep winning? Maybe — but that's a forecast about the future, not "
            "a backtest of the past. The 22%/yr on the chart was only ever buyable in hindsight.*"
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
            "# Magnificent-Seven — a quantitative teardown 🔬\n"
            "### HAC t-stat of the basket spread · equal-weight-field decomposition · the "
            "ex-post-selection placebo · the random-basket distribution · a no-edge synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two claims the Mag-7 label fuses: a **real** realised spread over the S&P "
            "500 (HAC *t* = 3.4 on the monthly difference) from an **absent** forward edge (the "
            "basket is selected on the outcome). The decisive object is the **ex-post-selection "
            "placebo**: \"pick the *k* highest-realised-return names\" — which manufactures a large "
            "spread even on a tape where *no* name has any true edge.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance monthly total returns, a 40-name "
            "mega-cap field + SPY, 130 months to 2026-05. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Mag 7 − SPY = **+{R['spread_bench_ann']:.1f}%/yr**, HAC "
            f"**t = {R['t_bench']:.2f}** (n = {R['months']}); **t = {R['t_equal']:.2f}** even vs the "
            "equal-weight field. *But* the basket is selected on the outcome — the synthetic null "
            "manufactures a spread with **zero** true edge, so the forward case is `WEAK`. |\n"
            f"| **Tradability** | `MIRAGE` | You could not have named these seven in 2015. A *blind* "
            f"7-pick beat SPY by **~{R['rand_mean']:.1f}%/yr** (Mag 7 at the **{R['mag7_pctile']:.1f}th** "
            f"percentile of 2,000 draws); the ex-post rule reproduces **{R['selection_share']}%** of "
            "the spread. Costs are trivial — the mirage is selection. |\n"
            f"| **Repeatable strategy?** | `BUSTED` | \"Hold the realised winners\" is a label, not a "
            f"rule. Forward, you get the random distribution (median +{R['rand_median']:.0f}%/yr, a "
            f"−{abs(R['mag7_dd']):.0f}% drawdown), not +{R['mag7_spread']:.0f}%. |\n\n"
            "> 💡 In plain words: the seven genuinely out-earned — and the tape is what *named* them. "
            "Strip the hindsight and the forward-pickable edge is ~4%, not ~22%."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the field be $N$ mega-caps with monthly total returns $r_{it}$. The Mag-7 basket is "
            "an equal-weight hold of a *fixed* set $S^\\star$ of $|S^\\star|=7$ names; the benchmark "
            "is the S&P 500 (SPY), $b_t$.\n\n"
            "- **H₁ (the spread is real).** $\\mathbb{E}[r^{S^\\star}_t - b_t] > 0$ on the tape, with a "
            "robust HAC *t*.\n"
            "- **H₂ (it is a forward factor).** $S^\\star$ is identifiable *ex ante* — the spread "
            "survives being chosen **without** look-ahead.\n"
            "- **H₃ (it is repeatable).** A forward rule reproduces the spread out-of-selection.\n\n"
            "The label defines $S^\\star = \\arg\\max_{|S|=7}$ (realised wealth) — i.e. $S^\\star$ is a "
            "**function of the full sample**. So H₁ is true *by construction-adjacent* selection, and "
            "the real test is H₂/H₃. We find **H₁ accepted**, **H₂ rejected** (selection alone "
            "manufactures the spread on a no-edge tape), **H₃ rejected** (forward = the random "
            "distribution). The label is true exactly where it doesn't help and false exactly where "
            "it would."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The decomposition is the whole teardown. Write the Mag-7 spread over the cap index as\n\n"
            "$$\\underbrace{(r^{S^\\star}-r^{\\text{EW-field}})}_{\\text{name selection}} + "
            "\\underbrace{(r^{\\text{EW-field}}-b)}_{\\text{equal- vs cap-weight}}.$$\n\n"
            "The second term is the well-known equal-weight tilt (Plyakha-Uppal-Vilkov); we net it "
            "out with an equal-weight-field control. The first term is **name selection** — and "
            "because $S^\\star$ is chosen on the outcome, it is not an estimate of any forward edge. "
            "The clean way to size the selection artefact is a **placebo**: run the identical "
            "$\\arg\\max$ rule on a synthetic tape where the true cross-sectional edge is **zero**. "
            "Whatever spread it returns is pure selection — and it is large."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** yfinance monthly total returns (auto-adjusted), {R['n_universe']} US "
            f"mega-caps (the seven + 33 also-rans) + SPY, {R['months']} months {R['window']} "
            f"(fingerprint `{R['fp']}`). Equal-weight, monthly rebalance.\n"
            "- **Signal test.** HAC (Newey-West) *t*-stat of the monthly spread $r^{S^\\star}_t - b_t$ "
            "and $r^{S^\\star}_t - r^{\\text{EW-field}}_t$. A robust *t* ≥ 2 is required for `REAL`.\n"
            "- **Random-basket distribution.** 2,000 seeded blind 7-picks from the field — the "
            "sampling law of a *forward* pick with no skill. Where the Mag 7 sits in it measures the "
            "selection.\n"
            "- **Ex-post placebo (the opt-in look-ahead).** `expost_winners` ranks names by "
            "full-sample realised wealth and holds the top 7 — the Mag-7 rule made explicit. It is "
            "gated behind `allow_lookahead_selection=True`.\n"
            "- **Synthetic positive control.** A deterministic monthly panel with a tunable "
            "*pre-named* `alpha_spread`. At `alpha=0` no name has an edge, so the *only* spread the "
            "placebo can return is selection — the null that proves the artefact.\n"
            "- **Costs / lag.** One-way turnover × NAV at 10 bps/rebalance (immaterial for a 7-name "
            "monthly basket); one execution lag where any signal is used. The look-ahead measured "
            "here is the *membership selection*, not the accounting."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The spread is real — and survives the weighting control\n\n"
            "CAGR of each leg, and the HAC *t* of the Mag-7 spread over the cap index **and** over "
            "the equal-weight field (which removes the equal-vs-cap tilt)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mag7 = st.basket_returns(RETS, data.MAG7); eqf = st.equal_field_returns(RETS)\n"
            "    legs = {'Mag 7': st.summarize(mag7)['cagr'], 'S&P 500': st.summarize(BENCH)['cagr'],\n"
            "            'EW field': st.summarize(eqf)['cagr']}\n"
            "    tb = st.hac_tstat_diff(mag7, BENCH); te = st.hac_tstat_diff(mag7, eqf)\n"
            "    legs = {k: v*100 for k,v in legs.items()}\n"
            "else:\n"
            "    legs = {'Mag 7': R['mag7_cagr'], 'S&P 500': R['spy_cagr'], 'EW field': R['eqf_cagr']}\n"
            "    tb = {'tstat': R['t_bench'], 'mean_diff_ann': R['spread_bench_ann']/100}\n"
            "    te = {'tstat': R['t_equal'], 'mean_diff_ann': R['spread_equal_ann']/100}\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(list(legs), list(legs.values()), color=[GREEN, GREY, AMBER], width=.6)\n"
            "for i,v in enumerate(legs.values()): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('CAGR (%/yr)'); ax.set_title(f\"Mag 7 beats both — HAC t={tb['tstat']:.1f} vs SPY, {te['tstat']:.1f} vs EW field\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Mag 7 - SPY      : {tb['mean_diff_ann']*100:+.1f}%/yr  HAC t={tb['tstat']:.2f}\")\n"
            "print(f\"Mag 7 - EW field : {te['mean_diff_ann']*100:+.1f}%/yr  HAC t={te['tstat']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the spread is **real on the tape** — +{R['spread_bench_ann']:.0f}%/yr "
            f"at **t = {R['t_bench']:.1f}** vs the index, and still **t = {R['t_equal']:.1f}** after "
            "stripping the equal-weight tilt (so it is *name* selection, not just a weighting trick). "
            "H₁ accepted. That's the half the pitch gets right — and the half that doesn't help you."
        ),
        md(
            "### 4b · …but it is selection — the Mag 7 vs the blind-pick distribution\n\n"
            "The CAGR spread over SPY of 2,000 seeded random 7-baskets (the forward, no-skill law) "
            "with the famous seven, and the ex-post \"pick the winners\" basket, marked."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rands = st.random_baskets(RETS, k=7, n_draws=2000, seed=355)\n"
            "    spy = st.summarize(BENCH)['cagr']\n"
            "    rs = (np.array([st.summarize(rands[c])['cagr'] for c in rands.columns]) - spy)*100\n"
            "    mag_sp = (st.summarize(st.basket_returns(RETS, data.MAG7))['cagr'] - spy)*100\n"
            "    win = st.expost_winners(RETS, k=7, allow_lookahead_selection=True)\n"
            "    exp_sp = (st.summarize(st.basket_returns(RETS, win))['cagr'] - spy)*100\n"
            "    pct = st.percentile_of(mag_sp, rs)\n"
            "else:\n"
            "    rng = np.random.default_rng(355); rs = rng.normal(R['rand_mean'], 6.0, 2000)\n"
            "    mag_sp, exp_sp, pct = R['mag7_spread'], R['expost_spread'], R['mag7_pctile']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(rs, bins=36, color=GREY, alpha=.7, label='random blind 7-pick (no skill)')\n"
            "ax.axvline(np.median(rs), c=AMBER, lw=2, ls='--', label=f'median blind pick (~{np.median(rs):.0f}%)')\n"
            "ax.axvline(mag_sp, c=GREEN, lw=2.5, label=f'Mag 7 (~{mag_sp:.0f}%, {pct:.1f}th pct)')\n"
            "ax.axvline(exp_sp, c=RED, lw=2.5, ls=':', label=f'ex-post winners (~{exp_sp:.0f}%)')\n"
            "ax.set_xlabel('CAGR spread over S&P 500 (%/yr)'); ax.set_ylabel('# baskets')\n"
            "ax.set_title('The Mag 7 lives in the right tail of pure selection'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'blind median {np.median(rs):+.1f}%  |  Mag 7 {mag_sp:+.1f}% ({pct:.1f}th pct)  |  ex-post winners {exp_sp:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: a *blindly* chosen seven beat SPY by a median **~{R['rand_median']:.0f}%/yr**; "
            f"the Mag 7 by **~{R['mag7_spread']:.0f}%**, sitting at the **{R['mag7_pctile']:.1f}th** "
            f"percentile — and the explicit \"pick the winners\" rule reaches **~{R['expost_spread']:.0f}%**, "
            f"reproducing **{R['selection_share']}%** of the Mag-7 spread. The famous seven are ~6 of "
            "the 7 ex-post winners (the 7th is **AMD**, not META). H₂ rejected: the spread is *where "
            "you'd be if you'd picked the winners*, not where a forward pick lands."
        ),
        md(
            "### 4c · The placebo on a no-edge tape — selection from nothing\n\n"
            "The clinching control. On a deterministic synthetic panel where the *pre-named* basket "
            "carries `alpha_spread` and everyone else is identical, compare the **pre-named** spread "
            "(the honest factor, with its *t*) to the **ex-post placebo** spread (selection)."
        ),
        code(
            "rows = []\n"
            "for a in (0.0, 0.004):\n"
            "    rdf, b, truth = data.synthetic_panel(alpha_spread=a)\n"
            "    named = st.basket_returns(rdf, truth['true_basket_cols'])\n"
            "    expo  = st.basket_returns(rdf, st.expost_winners(rdf, k=7, allow_lookahead_selection=True))\n"
            "    sb = st.summarize(b)['cagr']\n"
            "    rows.append((a, (st.summarize(named)['cagr']-sb)*100, st.hac_tstat_diff(named,b)['tstat'],\n"
            "                    (st.summarize(expo)['cagr']-sb)*100))\n"
            "labels = ['no true edge\\n(alpha=0)', 'real pre-named edge\\n(alpha=0.004)']\n"
            "named_sp = [r[1] for r in rows]; placebo_sp = [r[3] for r in rows]\n"
            "x=np.arange(2); fig,ax=plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(x-.18, named_sp, .36, color=GREY, label='pick basket IN ADVANCE (honest)')\n"
            "ax.bar(x+.18, placebo_sp, .36, color=RED, label='pick the WINNERS (look-ahead)')\n"
            "ax.axhline(0,c='k',lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('spread over the field (%/yr)'); ax.set_title('Selection manufactures a spread even with ZERO true edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for a,n,t,e in rows: print(f'alpha={a:.3f}: pre-named {n:+.1f}%/yr (t={t:+.2f})   ex-post placebo {e:+.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the null row (`alpha=0`) is decisive. With **no** true cross-"
            f"sectional edge, a basket fixed *in advance* shows **{R['syn_null_named']:+.0f}%/yr** "
            f"(t = {R['syn_null_named_t']:+.2f} — luck) while the look-ahead rule conjures "
            f"**+{R['syn_null_placebo']:.0f}%/yr** from nothing. The engine is faithful: it finds a "
            "spread *only* where one is planted (pre-named) and manufactures one *whenever* you "
            "select on the outcome. That manufactured spread is the artefact the Mag-7 label carries "
            "by construction."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — the realised spread is genuine (+{R['spread_bench_ann']:.1f}%/yr, "
            f"HAC **t = {R['t_bench']:.2f}**, n = {R['months']}; t = {R['t_equal']:.2f} vs the EW field), "
            "so it is *not* a pure weighting artefact. But the basket is **selected on the outcome**: "
            f"the no-edge null still yields a +{R['syn_null_placebo']:.0f}%/yr placebo, the ex-post "
            f"rule reproduces {R['selection_share']}% of the spread, and the seven sit at the "
            f"{R['mag7_pctile']:.1f}th percentile of random baskets. Real recent spread, look-ahead/"
            "survivorship selected → `MIXED` (a `WEAK` forward case under a strong in-sample t).\n"
            f"- **Tradability `MIRAGE`** — costs are immaterial (net CAGR {R['mag7_net_cagr']:.1f}% ≈ "
            f"gross {R['mag7_cagr']:.1f}%). The mirage is that the basket was unknowable ex ante; the "
            f"forward-tradable object is the random seven (~+{R['rand_mean']:.0f}%/yr, "
            f"−{abs(R['mag7_dd']):.0f}% drawdown) or a pre-named factor (which the null shows can be "
            "zero).\n"
            "- **Repeatable strategy? `BUSTED`** — \"hold the realised winners\" is a label, not a "
            f"forward rule; out-of-selection it is the random distribution (median +{R['rand_median']:.0f}%/yr), "
            "not +22%. The Mag 7 is yesterday's winners."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the forward vs hindsight gap\n\n"
            "The single number that settles tradability: the CAGR spread you'd have captured "
            "**forward** (a blind seven) versus the spread hindsight *names* (the Mag 7, the ex-post "
            "winners). The gap is the selection premium — unbuyable in real time."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = st.summarize(BENCH)['cagr']\n"
            "    rands = st.random_baskets(RETS, k=7, n_draws=2000, seed=355)\n"
            "    blind = float(np.median([st.summarize(rands[c])['cagr'] for c in rands.columns]) - spy)*100\n"
            "    mag_sp = (st.summarize(st.basket_returns(RETS, data.MAG7))['cagr'] - spy)*100\n"
            "    exp_sp = (st.summarize(st.basket_returns(RETS, st.expost_winners(RETS,7,allow_lookahead_selection=True)))['cagr'] - spy)*100\n"
            "else:\n"
            "    blind, mag_sp, exp_sp = R['rand_median'], R['mag7_spread'], R['expost_spread']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "bars = ['FORWARD\\n(blind 7-pick)', 'the Mag 7 label\\n(hindsight)', 'pure look-ahead\\n(ex-post winners)']\n"
            "vals = [blind, mag_sp, exp_sp]\n"
            "ax.bar(bars, vals, color=[GREEN, AMBER, RED], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('CAGR spread over S&P 500 (%/yr)'); ax.set_title('What you could buy forward vs what hindsight names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'forward (blind median) {blind:+.0f}%  ->  Mag 7 label {mag_sp:+.0f}%  ->  pure look-ahead {exp_sp:+.0f}%')"
        ),
        md(
            f"> 💡 In plain words: the buyable, forward edge is ~**+{R['rand_median']:.0f}%/yr** (a blind "
            f"seven). Everything above that — up to the **+{R['mag7_spread']:.0f}%** the Mag-7 label "
            f"shows and the **+{R['expost_spread']:.0f}%** of pure hindsight — is the **selection "
            "premium**, available only to someone who already knows the answer. There is no sizing, "
            "no rebalance, no cost assumption that converts the hindsight bars into a forward one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The panel-level twin.** [Study 345 — Survivorship-Bias](../../345-survivorship-bias/): "
            "delete the *losers* before the backtest sees them — the Mag 7 keeps only the seven top "
            "*winners*, the mirror image.\n"
            "- **The honest forward benchmark.** [Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/) "
            "supplies the random-basket distribution and the equal-vs-cap decomposition reused here.\n"
            "- **Point-in-time membership.** The cleanest extension: rank names by trailing momentum "
            "*at each date* and hold the top 7 going forward (no look-ahead). The prior is that this "
            "earns the decayed, regime-dependent momentum premium — far below the +22% the static "
            "label shows. ~Survivorship: add the mega-caps that *died* (e.g. GE's collapse) to the "
            "field and the blind-pick spread falls further.\n\n"
            "*The reproducible core is offline and deterministic; the look-ahead rule is gated "
            "behind an explicit opt-in. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
