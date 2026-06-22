"""Generate the two narrative notebooks for Study 357 (the Coffee-Can portfolio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The real-tape cells read the cached yfinance
monthly total-return panel under ``../_cache/coffee_can_monthly.csv`` and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic survivorship
positive control runs anywhere with no network (fixed seed).
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


# Frozen real-tape + synthetic headline numbers — mirror of docs/results.md.
# Real: yfinance monthly total-return, 2005-01 -> 2026-05 (256 monthly returns, 21.3y),
# basket KO/JNJ/PG/PEP/WMT/MMM/MO/XOM/IBM/GE vs SPY. Synthetic: 400-firm panel, same 8%
# drift, 3%/yr death hazard, seed 357.
R = dict(
    window="2005-01 → 2026-05", months=256, years=21.3,
    basket=["KO", "JNJ", "PG", "PEP", "WMT", "MMM", "MO", "XOM", "IBM", "GE"],
    can_cagr=9.49, can_vol=12.58, can_sharpe=0.786, can_mdd=-30.6,
    reb_cagr=10.28, reb_vol=12.76, reb_sharpe=0.834, reb_mdd=-31.9,
    spy_cagr=11.11, spy_vol=15.13, spy_sharpe=0.775, spy_mdd=-50.8,
    excess_ann=-1.84, t0=-0.81, t6=-0.83, p6=0.4059, n=256,
    can_vs_reb_ann=-0.75, can_vs_reb_t=-1.26,
    reb_cost0=10.296, reb_cost10=10.284, reb_cost20=10.272, entry_cost_pct=0.10,
    syn_firms=400, syn_dead=189, syn_dead_pct=47,
    syn_famous=17.08, syn_honest=8.92, syn_all=4.72,
    syn_gap_honest=8.16, syn_gap_all=12.36,
    nd_famous=18.15, nd_honest=12.19, nd_all=8.23, nd_gap_all=9.92,
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Survivorship%3F: Confirmed](https://img.shields.io/badge/Survivorship%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from coffee_can import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    BRET = data.monthly_returns(PRICES)[data.BASKET]
    SPY = data.monthly_returns(PRICES)["SPY"]
else:
    PRICES = BRET = SPY = None
print("real cache present:", HAVE_REAL,
      "| months:", (0 if SPY is None else len(SPY)))
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
            "# The Coffee-Can portfolio — buy great stocks and never sell? ☕\n"
            "### Robert Kirby's \"put them in a can and forget them\" idea, in plain English\n\n"
            + BADGES +
            "There's a beautiful old Wall Street story. A money manager named **Robert Kirby** "
            "noticed that one of his clients had quietly ignored every \"sell\" call for years — "
            "just bought the stocks Kirby recommended and stuck the certificates in a drawer. When "
            "the client died, that untouched account had *crushed* the ones Kirby actively managed. "
            "One forgotten holding alone had ballooned into a fortune. Kirby called it the **Coffee "
            "Can portfolio**: buy quality names, drop them in a can, and *never trade for a decade.*\n\n"
            "The pitch is irresistible: **do nothing, pay nothing, beat the pros.** Is it real? "
            "Partly — and the honest part isn't the part people repeat. This notebook pulls the two "
            "halves apart.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the cost model on NAV and the "
            "survivorship maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart is drawn by the code beside it; house style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a buy-and-hold \"can\" of quality names beat the index? | **Not on the money.** Our "
            f"can compounds at **{R['can_cagr']:.1f}%/yr** vs the S&P's **{R['spy_cagr']:.1f}%** over 21 "
            "years. It *lost* the race for total return. |\n"
            "| So the legend is bunk? | **Not quite.** The can rode out the storms far better: **half "
            f"the worst-case loss** ({R['can_mdd']:.0f}% vs {R['spy_mdd']:.0f}%) and lower wobble, so "
            "its *risk-adjusted* score basically ties the index. Smoother, not richer. |\n"
            "| Where does the 'magic' really come from? | **Two places — and only one is honest.** "
            "(1) *Low cost:* never trading means never paying the spread/tax — real, but tiny. "
            "(2) *Survivorship:* you pick the can's names **knowing** which ones became famous. That "
            "hindsight is most of the magic, and it's fake. |\n"
            "| Could you actually do it? | **Sort of — for the calm, not the returns.** The cost edge "
            "is real but small; the return edge vanishes once you can't cheat by knowing the winners. |\n\n"
            "> The discipline is real. The *out-performance* is mostly hindsight."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Buy a basket of quality companies. Put the certificates in a coffee can. Don't open "
            "it for ten years. You'll beat the managers who fuss over every tick — because you won't "
            "sell your winners too early, you won't churn, and you won't pay the costs and taxes that "
            "trading bleeds away.\"*\n\n"
            "Robert Kirby told this story in a 1984 essay (*The Coffee Can Portfolio*, Journal of "
            "Portfolio Management). It's been repeated ever since — by Charlie Munger's \"sit on your "
            "ass\" investing, by index-fund evangelists, by every \"just buy and hold\" thread online. "
            "The strongest version: **a zero-turnover basket of good companies beats both the active "
            "manager and, often, the index — at lower cost and lower stress.**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, it's the best deal in finance: market-beating returns for *zero effort and zero "
            "fees.* It would also say something deep — that **trading is net-negative**, that the "
            "discipline of *not* selling is itself the edge. But two very different things are bundled "
            "in the legend, and telling them apart is the whole game:\n\n"
            "1. **Cost.** Not trading saves the spread, the commission and (in a taxable account) the "
            "capital-gains tax. That's real money — but *how much?*\n"
            "2. **Survivorship.** Kirby's can looked brilliant **after the fact**, when one holding had "
            "become a monster. But the names you'd put in a can *today* are the ones you already know "
            "won. Pick winners with hindsight and *any* portfolio looks like genius."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a basket of **{len(R['basket'])} quality large-caps** — {', '.join(R['basket'])} — "
            f"and **{R['years']:.0f} years** of real monthly prices (dividends reinvested), and we run "
            "three portfolios side by side:\n\n"
            "1. **The Coffee Can.** Buy equal-weight once, then *never touch it.* Weights drift forever.\n"
            "2. **The rebalanced twin.** Same basket, but reset to equal weight every year — and "
            "*charged* the cost of that trading. The gap between the two is the pure cost tailwind.\n"
            "3. **The index (S&P 500).** What the \"pros\" are really being compared to.\n\n"
            "Then the honesty check: on a **made-up** stock market where every company has the *same* "
            "true return — so there is genuinely **no** skill to find — we build the can two ways: by "
            "**hindsight** (the names that ended up famous) and **honestly** (names picked at the "
            "start). If the hindsight can looks great anyway, that's survivorship, pure and measured."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: did the can beat the index?** Grow $1 in each over 21 years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    can = st.buy_and_hold(BRET); spy = SPY\n"
            "    reb = st.rebalanced(BRET, rebal=12, cost_bps=10.0)\n"
            "    eq = {'Coffee Can (never trade)': (1+can).cumprod(),\n"
            "          'Rebalanced yearly': (1+reb).cumprod(),\n"
            "          'S&P 500 (index)': (1+spy).cumprod()}\n"
            "    x = can.index\n"
            "else:\n"
            "    x = np.arange(R['months'])\n"
            "    def grow(c): return (1+c/100)**(np.arange(R['months'])/R['months']*R['years'])\n"
            "    eq = {'Coffee Can (never trade)': grow(R['can_cagr']),\n"
            "          'Rebalanced yearly': grow(R['reb_cagr']),\n"
            "          'S&P 500 (index)': grow(R['spy_cagr'])}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "for (lab, e), c in zip(eq.items(), [GREEN, AMBER, GREY]):\n"
            "    ax.plot(x, np.asarray(e), label=lab, lw=2, c=c)\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.set_title('21 years: the can rides smoother but finishes behind')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"final $: can ${np.asarray(list(eq.values())[0])[-1]:.2f}  rebal ${np.asarray(list(eq.values())[1])[-1]:.2f}  index ${np.asarray(list(eq.values())[2])[-1]:.2f}\")"
        ),
        md(
            f"The index wins the money race: **{R['spy_cagr']:.1f}%/yr** vs the can's "
            f"**{R['can_cagr']:.1f}%**. So far the legend is *not* delivering — \"never sell\" did not "
            "beat the market here."
        ),
        md(
            "**But look at the ride, not just the finish.** The can is far calmer — it lost much less "
            "in the crashes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [('Coffee Can', st.summary(can)), ('Rebalanced', st.summary(reb)), ('S&P 500', st.summary(spy))]\n"
            "    vols = [s['vol']*100 for _,s in rows]; mdds = [-s['mdd']*100 for _,s in rows]\n"
            "else:\n"
            "    vols = [R['can_vol'], R['reb_vol'], R['spy_vol']]; mdds = [-R['can_mdd'], -R['reb_mdd'], -R['spy_mdd']]\n"
            "labs = ['Coffee Can','Rebalanced','S&P 500']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "axes[0].bar(labs, vols, color=[GREEN, AMBER, GREY], width=.6); axes[0].set_title('Yearly wobble (volatility, %)')\n"
            "axes[1].bar(labs, mdds, color=[GREEN, AMBER, GREY], width=.6); axes[1].set_title('Worst peak-to-trough loss (%)')\n"
            "for ax, series in zip(axes, [vols, mdds]):\n"
            "    for i,v in enumerate(series): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"worst loss — can {R['can_mdd']:.0f}%  vs index {R['spy_mdd']:.0f}%\")"
        ),
        md(
            f"The can's worst drawdown was **{R['can_mdd']:.0f}%**; the index fell **{R['spy_mdd']:.0f}%** "
            "in 2008. *That's* the real coffee-can benefit: a smoother life. On a risk-adjusted basis "
            f"(Sharpe **{R['can_sharpe']:.2f}** vs **{R['spy_sharpe']:.2f}**) it's basically a tie. The "
            "can buys you calm — not extra money."
        ),
        md(
            "**Now the catch — where did the legend's 'magic' come from?** On a pretend market where "
            "*every* stock has the **same** true return (no skill exists), we build the can two ways: "
            "by **hindsight** (pick the names that ended up biggest) and **honestly** (pick at the "
            "start). Watch the hindsight version invent an edge out of thin air."
        ),
        code(
            "ps, _ = data.synthetic_panel(n_firms=400, n_months=240, mu_ann=0.08, death_hazard_ann=0.03, seed=357)\n"
            "g = st.survivorship_gap(ps, k=10, lookback=24)\n"
            "labs = ['Hindsight can\\n(famous winners)', 'Honest can\\n(picked at start)', 'Whole market\\n(the truth)']\n"
            "vals = [g['cagr_famous']*100, g['cagr_honest']*100, g['cagr_all']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(labs, vals, color=[RED, AMBER, GREY], width=.6)\n"
            "ax.axhline(8, ls='--', c='k', lw=1, label='true return everyone shares (8%)')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('return per year (%)'); ax.set_title('Survivorship: hindsight invents an edge where there is none'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"fake edge from hindsight: +{g['gap_vs_honest']*100:.1f} pp/yr vs an honest pick, +{g['gap_vs_all']*100:.1f} pp/yr vs the whole market\")"
        ),
        md(
            f"There it is. Every company in that market earns the **same 8%** — there is *no* skill to "
            f"find — yet the **hindsight** can shows **{R['syn_famous']:.0f}%/yr**, a fake "
            f"**+{R['syn_gap_honest']:.1f} points** over an honest pick (and +{R['syn_gap_all']:.1f} "
            "over the whole market, because the losers and the bankruptcies — the names you "
            "conveniently *don't* put in the can — drag the truth down to ~5%). **The coffee can's "
            "brilliance is mostly you, choosing the winners after the fact.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The can did **not** beat the index on return "
            f"({R['can_cagr']:.1f}% vs {R['spy_cagr']:.1f}%; the difference isn't statistically real). "
            "It *did* win on calm — half the drawdown, similar risk-adjusted score. Real benefit, but "
            "not the one the legend sells.\n"
            "- **Tradability — Fragile.** The one honest edge (low cost) is real but **tiny** at a "
            "sane trading pace; the headline out-performance leans on hindsight you can't have in "
            "advance.\n"
            "- **Survivorship? — Confirmed.** On a market with *no* skill, hindsight name-picking "
            f"manufactures **+{R['syn_gap_honest']:.0f} pp/yr** of fake edge. That's the engine under "
            "most coffee-can success stories."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually do it? — the cost tailwind, measured\n\n"
            "The *honest* coffee-can edge is cost: never trading means never paying. How big is it? "
            "Compare the never-trade can to a yearly-rebalanced twin charged a realistic cost each time "
            "it trades."
        ),
        code(
            "bps = [0, 5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    cagrs = [st.cagr(st.rebalanced(BRET, 12, b))*100 for b in bps]\n"
            "else:\n"
            "    cagrs = [R['reb_cost0'], 10.293, R['reb_cost10'], R['reb_cost20'], 10.248]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.plot(bps, cagrs, 'o-', c=AMBER, lw=2, label='rebalanced yearly, after cost')\n"
            "ax.axhline(R['can_cagr'], ls='--', c=GREEN, label=f\"never-trade can ({R['can_cagr']:.1f}%)\")\n"
            "ax.set_xlabel('trading cost per rebalance (basis points, one-way)'); ax.set_ylabel('return per year (%)')\n"
            "ax.set_title('Trading costs nibble — they do not feast (at a yearly pace)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"yearly rebalance costs only ~{(R['reb_cost0']-R['reb_cost20'])*100:.1f} hundredths of a %/yr even at 20bps — the cost edge is real but small\")"
        ),
        md(
            "> The cost saving from \"never trade\" is real but **tiny** at a yearly pace — a hundredth "
            "of a percent a year, not a percentage point. Where costs (and *taxes*) really bite is a "
            "high-churn active fund — which is the manager the legend beats, not the index. So the can "
            "is a fine, low-stress, tax-efficient way to **own the market** — just don't expect it to "
            "**beat** the market. The discipline is the product; the out-performance was hindsight."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The survivorship twin.** [Study 151 — Stocks-for-the-long-run](../../151-stocks-for-long-run/) "
            "and the dead-names problem: every \"great companies always win\" chart is drawn on the "
            "companies that *didn't* go bankrupt. Same bias, bigger canvas.\n"
            "- **Add the tax axis.** Our cost model charges a spread; in a *taxable* account the can's "
            "real advantage is deferring capital-gains tax for decades. Bolt a tax rate on and watch "
            "the honest edge grow (for high earners) — still small vs the survivorship illusion.\n"
            "- **Pick a worse basket.** Re-run with a *random* 2005 large-cap basket (including the "
            "ones that later cratered). The famous-names lift disappears — which is exactly the point.\n\n"
            "*Think your favourite \"buy and hold forever\" basket really beats the index? Re-pick it "
            "using only what you'd have known in 2005 — no peeking at who won — and see if the edge "
            "survives.*"
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
            "# The Coffee-Can portfolio — a quantitative teardown 🔬\n"
            "### Zero-turnover buy-and-hold vs a rebalanced twin vs the index · a paired HAC *t*-test "
            "of excess return · a cost model on NAV · a deterministic survivorship decomposition\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Kirby's "
            "Coffee Can fuses three claims: a **return** edge over the index, a **cost** edge from zero "
            "turnover, and — unspoken — a **survivorship** edge from choosing the basket with hindsight. "
            "We separate all three on a real monthly total-return tape and a deterministic synthetic "
            "control.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance monthly **total-return** (splits + "
            f"dividends), {R['window']} ({R['months']} monthly returns); basket = "
            f"{', '.join(R['basket'])} vs SPY. Offline core + synthetic survivorship control are "
            "deterministic (seed 357). Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Can CAGR **{R['can_cagr']:.2f}%** vs SPY **{R['spy_cagr']:.2f}%**; "
            f"paired HAC *t* of monthly excess = **{R['t6']:+.2f}** (n={R['n']}, p={R['p6']:.2f}) — "
            f"**no** return edge. But MDD **{R['can_mdd']:.0f}%** vs **{R['spy_mdd']:.0f}%** and Sharpe "
            f"**{R['can_sharpe']:.2f}** vs **{R['spy_sharpe']:.2f}** — a real *risk* edge. |\n"
            f"| **Tradability** | `FRAGILE` | The honest tailwind (zero turnover) is ~**1bp/yr** at an "
            f"annual rebalance; the can even trails its own rebalanced twin "
            f"({R['can_cagr']:.2f}% vs {R['reb_cagr']:.2f}%). No harvestable *return* edge survives. |\n"
            f"| **Survivorship?** | `CONFIRMED` | On a no-skill synthetic panel, the look-ahead "
            f"\"famous-names\" can fakes **+{R['syn_gap_honest']:.1f} pp/yr** over an ex-ante pick "
            f"(+{R['syn_gap_all']:.1f} vs the death-inclusive universe). |\n\n"
            "> 💡 In plain words: the can doesn't beat the index — it *de-risks* it. Its legendary "
            "out-performance is mostly hindsight name-selection, which we measure directly on a market "
            "engineered to have no edge at all."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Kirby (1984): equal-weight $k$ quality names at $t_0$, hold to $T$ with **zero** "
            "intervening trades. Let $r^{can}_t$ be the drift-weighted basket return, $r^{idx}_t$ the "
            "index, $c$ the one-way cost rate, $\\tau_t$ the rebalance turnover.\n\n"
            "- **H₁ (return edge).** $\\mathbb{E}[r^{can}_t - r^{idx}_t] > 0$, robust to HAC inference.\n"
            "- **H₂ (cost edge).** Avoiding $\\sum_t c\\,\\tau_t$ of rebalancing cost is materially "
            "positive: $\\text{CAGR}_{can} > \\text{CAGR}_{rebal,net}$.\n"
            "- **H₃ (no survivorship).** The basket's selection is not look-ahead; the same rule on an "
            "ex-ante universe gives the same edge.\n\n"
            "We find **H₁ rejected** (the excess is negative and insignificant; the win is in *risk*, "
            "not return), **H₂ true but immaterial** at sane turnover, and **H₃ decisively false** — "
            "the selection peeks at the winners, worth most of the apparent edge."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The terminal wealth of a buy-and-hold basket is $W_T = \\sum_i w_{i,0}\\prod_t(1+r_{i,t})$ "
            "— a *convex* function of each name's path, so the winners dominate and the weights drift "
            "toward them automatically (the can's \"let your winners run\"). That convexity is exactly "
            "what makes **selection** so dangerous: if the names are chosen on $\\prod_t(1+r_{i,t})$ "
            "itself (the terminal value), the basket inherits the right tail of the cross-section by "
            "construction. Formally the look-ahead estimator $\\mathbb{E}[r \\mid \\text{top-}k\\text{ "
            "terminal}]$ is biased up by an order-statistic gap that does **not** vanish with sample "
            "size. So measuring the can's return alone can't establish skill — you must hold the "
            "selection rule fixed and swap a look-ahead universe for an ex-ante one, which is what the "
            "synthetic control does."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** yfinance monthly **total-return** closes (`auto_adjust=True`: splits + "
            f"dividends), {R['window']}, {R['months']} monthly returns. Basket = {len(R['basket'])} "
            "quality large-caps; benchmark SPY. One execution convention: weights set once at $t_0$, "
            "returns realised contemporaneously (a buy-and-hold has no signal lag).\n"
            "- **Engines.** `buy_and_hold` (drift the weights, never trade) vs `rebalanced` (reset to "
            "equal weight every 12 months, charged $c\\cdot\\tau$ one-way on the L1 weight change) vs "
            "SPY.\n"
            "- **Signal test.** Paired *t* of $r^{can}_t - r^{idx}_t$ with a Newey-West SE "
            "(`hac_lag=6`) — the autocorrelation-robust statistic the Signal axis requires. Plus the "
            "risk decomposition (vol, MDD, Sharpe).\n"
            "- **Cost sweep.** CAGR of the rebalanced twin across $c \\in \\{0,5,10,20,40\\}$ bps — the "
            "size of the honest tailwind.\n"
            "- **Survivorship control.** A deterministic 400-firm panel where **every firm shares the "
            "same 8%/yr drift** (no skill exists) and carries a 3%/yr death hazard (delist → 95% wipe). "
            "Build the can by `famous_names` (top terminal value — look-ahead) vs `honest_names` (top "
            "early return — ex-ante). Any famous-minus-honest gap is **pure** survivorship; a "
            "no-death control isolates the look-ahead-selection floor.\n"
            "- **What would make us say 'mirage on the return claim.'** A negative or sub-2 *t* on the "
            "excess, *and* a famous-vs-honest gap that explains the apparent edge. Both hold."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · No return edge — paired HAC *t* of the can's excess over the index\n\n"
            "Monthly excess $r^{can}_t - r^{idx}_t$, mean annualised, with a Newey-West SE. The null is "
            "zero excess; a *positive* H₁ needs *t* ≥ 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    can = st.buy_and_hold(BRET); reb = st.rebalanced(BRET, 12, 10.0); spy = SPY\n"
            "    t0 = st.paired_t(can, spy, hac_lag=0); t6 = st.paired_t(can, spy, hac_lag=6)\n"
            "    tr = st.paired_t(can, reb, hac_lag=6)\n"
            "else:\n"
            "    t0 = {'mean_ann': R['excess_ann']/100, 't': R['t0']}\n"
            "    t6 = {'mean_ann': R['excess_ann']/100, 't': R['t6'], 'p': R['p6']}\n"
            "    tr = {'mean_ann': R['can_vs_reb_ann']/100, 't': R['can_vs_reb_t']}\n"
            "labels = ['can − index\\n(iid t)', 'can − index\\n(HAC t)', 'can − rebalanced\\n(HAC t)']\n"
            "ts = [t0['t'], t6['t'], tr['t']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, ts, color=[RED if v<2 else GREEN for v in ts], width=.55)\n"
            "ax.axhline(2, ls='--', c='k', label='significance bar (t = 2)'); ax.axhline(-2, ls='--', c='k')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i,v in enumerate(ts): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('t-statistic'); ax.set_title('The return edge is not there: every t is below +2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"can−index: mean {t6['mean_ann']*100:+.2f}%/yr, HAC t={t6['t']:+.2f} (p={t6.get('p', float('nan')):.2f})\")\n"
            "print(f\"can−rebalanced: mean {tr['mean_ann']*100:+.2f}%/yr, HAC t={tr['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the can's excess over the index is **{R['excess_ann']:+.1f}%/yr** with "
            f"HAC *t* = **{R['t6']:+.2f}** — not just shy of the bar, it's the *wrong sign*. H₁ "
            f"**rejected**. It even trails its own rebalanced twin (*t* = {R['can_vs_reb_t']:+.2f}). "
            "\"Never sell\" did not beat the market on this tape."
        ),
        md(
            "### 4b · The real win is risk, not return\n\n"
            "Where the can *does* deliver: lower volatility and a much shallower worst-case loss, for a "
            "comparable Sharpe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {'Coffee Can': st.summary(can), 'Rebalanced': st.summary(reb), 'S&P 500': st.summary(spy)}\n"
            "else:\n"
            "    rows = {'Coffee Can': dict(cagr=R['can_cagr']/100, vol=R['can_vol']/100, sharpe=R['can_sharpe'], mdd=R['can_mdd']/100),\n"
            "            'Rebalanced': dict(cagr=R['reb_cagr']/100, vol=R['reb_vol']/100, sharpe=R['reb_sharpe'], mdd=R['reb_mdd']/100),\n"
            "            'S&P 500': dict(cagr=R['spy_cagr']/100, vol=R['spy_vol']/100, sharpe=R['spy_sharpe'], mdd=R['spy_mdd']/100)}\n"
            "labs = list(rows); metrics = [('CAGR %', [rows[l]['cagr']*100 for l in labs]),\n"
            "    ('Vol %', [rows[l]['vol']*100 for l in labs]), ('Max drawdown %', [-rows[l]['mdd']*100 for l in labs]),\n"
            "    ('Sharpe', [rows[l]['sharpe'] for l in labs])]\n"
            "fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))\n"
            "for ax,(title,vals) in zip(axes, metrics):\n"
            "    ax.bar(labs, vals, color=[GREEN, AMBER, GREY], width=.6); ax.set_title(title)\n"
            "    ax.tick_params(axis='x', rotation=30)\n"
            "    for i,v in enumerate(vals): ax.annotate(f'{v:.1f}',(i,v),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "for l in labs: r=rows[l]; print(f\"{l:12s} CAGR={r['cagr']*100:5.2f}%  vol={r['vol']*100:5.2f}%  Sharpe={r['sharpe']:.3f}  MDD={r['mdd']*100:6.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: the can gives up **{R['spy_cagr']-R['can_cagr']:.1f} pp/yr** of return "
            f"to buy **{R['spy_vol']-R['can_vol']:.1f} pp** less volatility and a drawdown "
            f"**{abs(R['spy_mdd'])-abs(R['can_mdd']):.0f} pp shallower**. Sharpe ties "
            f"({R['can_sharpe']:.2f} vs {R['spy_sharpe']:.2f}). A defensive, low-beta basket of "
            "consumer-staples-heavy names — real, but it's *de-risking*, not alpha. This is why the "
            "Signal stamp is **MIXED**, not REAL."
        ),
        md(
            "### 4c · Survivorship control — the apparent edge, manufactured from nothing\n\n"
            "A deterministic panel where **every firm has the same 8%/yr drift** (no cross-sectional "
            "skill) and a 3%/yr death hazard. Build the can by look-ahead (famous terminal winners) vs "
            "ex-ante (honest early pick) vs the whole death-inclusive universe."
        ),
        code(
            "ps, alive = data.synthetic_panel(n_firms=400, n_months=240, mu_ann=0.08, sigma_ann=0.22, death_hazard_ann=0.03, seed=357)\n"
            "g = st.survivorship_gap(ps, k=10, lookback=24)\n"
            "ps_nd, _ = data.synthetic_panel(n_firms=400, n_months=240, death_hazard_ann=0.0, seed=357)\n"
            "gnd = st.survivorship_gap(ps_nd, k=10, lookback=24)\n"
            "cats = ['famous\\n(look-ahead)', 'honest\\n(ex-ante)', 'whole\\nuniverse']\n"
            "with_d = [g['cagr_famous']*100, g['cagr_honest']*100, g['cagr_all']*100]\n"
            "no_d = [gnd['cagr_famous']*100, gnd['cagr_honest']*100, gnd['cagr_all']*100]\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.18, with_d, .36, color=RED, label='with delistings (honest universe)')\n"
            "ax.bar(x+.18, no_d, .36, color=GREY, label='no delistings (control)')\n"
            "ax.axhline(8, ls='--', c='k', lw=1, label='true planted drift (8%) — no real edge exists')\n"
            "ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('No skill in the data — yet hindsight selection fakes a large edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"with deaths : famous {g['cagr_famous']*100:.2f}%  honest {g['cagr_honest']*100:.2f}%  all {g['cagr_all']*100:.2f}%  -> survivorship gap +{g['gap_vs_honest']*100:.2f}pp (vs honest), +{g['gap_vs_all']*100:.2f}pp (vs all)\")\n"
            "print(f\"no deaths   : all {gnd['cagr_all']*100:.2f}% ~ planted 8% -> engine unbiased; the residual famous-lift is pure look-ahead order-statistics\")"
        ),
        md(
            f"> 💡 In plain words: the planted truth is **8%/yr for everyone** — there is no skill to "
            f"find. Yet the look-ahead can shows **{R['syn_famous']:.0f}%**, an entirely fake "
            f"**+{R['syn_gap_honest']:.1f} pp** over an honest pick and **+{R['syn_gap_all']:.1f} pp** "
            "over the death-inclusive universe (the dead names, which never make it into a real coffee "
            f"can, drag the truth to {R['syn_all']:.1f}%). The no-death control returns "
            f"**{R['nd_all']:.1f}% ≈ 8%** for the whole universe — confirming the engine is unbiased "
            "and that the gap is selection, not a coding error. **H₃ decisively false.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — return excess over SPY **{R['excess_ann']:+.2f}%/yr**, HAC "
            f"*t* = **{R['t6']:+.2f}** (n={R['n']}): no edge, wrong sign. But a genuine **risk** edge "
            f"(MDD {R['can_mdd']:.0f}% vs {R['spy_mdd']:.0f}%, Sharpe {R['can_sharpe']:.2f} vs "
            f"{R['spy_sharpe']:.2f}). Real on risk · None on return → **MIXED**.\n"
            f"- **Tradability `FRAGILE`** — the honest cost tailwind is ~**1bp/yr** at a yearly "
            f"rebalance, and the can trails its own rebalanced twin ({R['can_cagr']:.2f}% vs "
            f"{R['reb_cagr']:.2f}%). The low-cost benefit is real but immaterial against the index; the "
            "headline out-performance needs hindsight.\n"
            f"- **Survivorship? `CONFIRMED`** — on a no-skill panel, look-ahead selection fabricates "
            f"**+{R['syn_gap_honest']:.1f} pp/yr** (vs honest) / **+{R['syn_gap_all']:.1f} pp/yr** (vs "
            "the death-inclusive universe). The legend's magic is mostly choosing winners after the "
            "fact."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost tailwind and its limits\n\n"
            "The only *defensible* coffee-can edge is cost: zero turnover pays no spread, no "
            "commission, and (untested here) defers tax. Sweep the rebalanced twin's cost to size it."
        ),
        code(
            "bps = [0, 5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    cagrs = [st.cagr(st.rebalanced(BRET, 12, b))*100 for b in bps]\n"
            "    can_c = st.cagr(can)*100\n"
            "else:\n"
            "    cagrs = [R['reb_cost0'], 10.293, R['reb_cost10'], R['reb_cost20'], 10.248]; can_c = R['can_cagr']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(bps, cagrs, 'o-', c=AMBER, lw=2, label='rebalanced yearly, net of cost')\n"
            "ax.axhline(can_c, ls='--', c=GREEN, label=f'never-trade can ({can_c:.2f}%)')\n"
            "ax.set_xlabel('one-way cost per rebalance (bps)'); ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('Cost erosion at a yearly pace is ~1bp/yr — real but not the edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('rebalanced CAGR by cost (bps):', {b: round(c,3) for b,c in zip(bps, cagrs)})\n"
            "print(f\"cost drag 0->20bps: {(cagrs[0]-cagrs[3]):.3f} pp/yr; entry cost paid once: {R['entry_cost_pct']:.2f}% of NAV\")"
        ),
        md(
            "> 💡 In plain words: at an annual rebalance the cost wedge is a *hundredth* of a percent a "
            "year — third-decimal stuff. Costs only \"feast\" on a high-churn active fund (turnover "
            "many ×/yr) or a taxable account realising gains constantly — i.e. the *active manager* the "
            "legend beats, **not** the index. So the can is a sensible low-stress, tax-efficient way to "
            "*own* market beta, with a defensive tilt — but there is no harvestable edge over the index "
            "once you strip the hindsight. **FRAGILE**, and only on the cost/risk margins."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The dead-names canvas.** [Study 151 — Stocks-for-the-long-run](../../151-stocks-for-long-run/): "
            "the same survivorship that inflates a coffee can inflates every \"equities always win\" "
            "long-horizon chart. Quantify the delisting drag on a real CRSP-style universe.\n"
            "- **The tax leg.** Extend the cost model to capital-gains deferral: in a taxable account "
            "the can's compounding-on-pre-tax-dollars is a *real* and larger edge for high earners — "
            "still bounded, and orthogonal to the survivorship illusion. Net-of-tax vs net-of-cost, "
            "labelled.\n"
            "- **Ex-ante basket construction.** Replace the hindsight basket with a *rules-based* "
            "quality screen applied with 2005 information only (point-in-time fundamentals); re-run the "
            "paired *t*. The prediction: the apparent edge collapses toward zero, the risk edge "
            "survives.\n\n"
            "*The reproducible core (synthetic survivorship control) is offline and deterministic "
            "(seed 357); the real tape is cached yfinance total-return. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
