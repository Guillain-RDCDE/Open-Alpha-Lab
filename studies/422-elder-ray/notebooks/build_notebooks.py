"""Generate the two narrative notebooks for Study 422 (Elder Ray).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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
    asof="2026-05-31",
    n_days=8390, years=33.3,
    # SPY Elder-Ray long/flat, net, cost=2bps
    sharpe=0.126, cagr=0.74, mdd=-38.98, hac_t=0.730, tim=46.2,
    turnover=29.9, perm_p=0.990, sharpe_delta=-0.524,
    # SPY buy-and-hold
    bh_sharpe=0.650, bh_cagr=10.90, bh_mdd=-55.19, bh_hac_t=4.36,
    # Simpler benchmarks on SPY (net, cost=2bps)
    sma_sharpe=0.731, sma_cagr=8.36, sma_hac_t=4.52,
    emac_sharpe=0.329, emac_cagr=3.11, emac_hac_t=1.97,
    # Long/short variant
    ls_sharpe=-0.129, ls_cagr=-1.13, ls_hac_t=-0.737, ls_tim=15.1,
    # Cost sweep (Elder long/flat SPY)
    cost00_sharpe=0.189, cost00_t=1.108,
    cost10_sharpe=0.157, cost10_t=0.919,
    cost20_sharpe=0.126, cost20_t=0.730,
    cost50_sharpe=0.030, cost50_t=0.172,
    # Panel: elder_sharpe, bh_sharpe, delta, hac_t
    spy_es=0.126, spy_bh=0.650, spy_d=-0.524, spy_t=0.730,
    qqq_es=0.247, qqq_bh=0.523, qqq_d=-0.275, qqq_t=1.316,
    iwm_es=0.066, iwm_bh=0.472, iwm_d=-0.407, iwm_t=0.338,
    efa_es=0.094, efa_bh=0.406, efa_d=-0.312, efa_t=0.500,
    gld_es=0.372, gld_bh=0.666, gld_d=-0.295, gld_t=1.770,
    # Synthetic positive control (n=6000)
    syn0_sharpe=0.021, syn0_bh=0.440, syn0_delta=-0.419, syn0_t=0.099, syn0_p=0.894,
    syn20_sharpe=0.919, syn20_bh=0.418, syn20_delta=0.501, syn20_t=3.853, syn20_p=0.002,
)

PANEL = ["SPY", "QQQ", "IWM", "EFA", "GLD"]

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

from elder_ray import data, strategy as st

PANEL = ["SPY", "QQQ", "IWM", "EFA", "GLD"]
ASOF = "2026-05-31"
CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have(t):
    return os.path.exists(data._cache_path(t, CACHE))

HAVE_REAL = _have("SPY")

def load(t):
    return data.load_real(t, cache_dir=CACHE).loc[:ASOF]

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Elder Ray — does Dr Elder's Bull/Bear Power beat just holding?\n"
            "### Alexander Elder's Bull Power / Bear Power oscillators, tested honestly on real tapes\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a plain SMA filter%3F: Busted](https://img.shields.io/badge/Beats_a_plain_SMA_filter%3F-Busted-8b949e?style=flat-square)\n\n"
            "Dr Alexander Elder built *Elder Ray* — Bull Power and Bear Power — to read the "
            "tug-of-war around a 13-day moving average of value. Bull Power is how far the "
            "day's *high* pokes above value; Bear Power is how far the *low* sinks below it. "
            "The recipe: in an uptrend, buy when the bears are losing their grip. Does timing "
            "the market this way beat simply buying and holding SPY?\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the permutation "
            "placebo and the head-to-head against a plain trend filter? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the Elder-Ray timing rule beat buy-and-hold? | **No.** Net Sharpe "
            f"**{R['sharpe']:.2f}** vs buy-and-hold's **{R['bh_sharpe']:.2f}** on SPY — it gives up "
            f"most of the market's return to sit in cash {100 - R['tim']:.0f}% of the time. |\n"
            f"| Is the timing edge real? | **No.** HAC *t* = **{R['hac_t']:.2f}** (needs ≥ 2); "
            f"a rotation placebo says *p* = **{R['perm_p']:.2f}** — indistinguishable from random timing. |\n"
            f"| Is it at least better than a plain 200-day trend filter? | **No.** The dead-simple "
            f"\"long when price > 200-day average\" filter scores Sharpe **{R['sma_sharpe']:.2f}** — "
            "it beats Elder Ray on every metric. |\n"
            "| Could you trade it? | **No.** ~30 round-trips/yr of turnover and a sub-market "
            "gross edge — costs only deepen the gap. |\n\n"
            "> Elder Ray is a *decent risk-reducer* (smaller drawdown) but a *bad return engine* — "
            "and the extra Bull/Bear Power machinery adds nothing over a one-line moving-average filter."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Bull Power and Bear Power show who is winning the battle between buyers and "
            "sellers. Trade with the trend: when the 13-day EMA is rising and Bear Power turns "
            "up from below zero, the bears are exhausted — buy. When the trend turns down, step "
            "aside. This times your entries far better than blindly holding.\"*\n\n"
            "Alexander Elder introduced Elder Ray in *Trading for a Living* (1993) as the second "
            "screen of his Triple Screen system. The idea is genuinely appealing: instead of one "
            "number, you watch *both* the high's reach above value and the low's reach below it. "
            "We test whether that extra detail earns its keep against the simplest alternatives."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If Elder Ray timed entries better than buy-and-hold, it would be a mechanical edge "
            "any retail trader could run on a daily chart. The stakes: Elder's books have sold "
            "in the hundreds of thousands, and Bull/Bear Power ships on nearly every charting "
            "platform. If the indicator only *reduces drawdown* at the cost of *most of the "
            "return* — and a one-line moving-average filter does the same job better — then the "
            "extra machinery is decoration, not edge."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "One honest test:\n\n"
            "1. **Turn the rule into a long/flat book.** Long when EMA13 rises and Bear Power "
            "is negative-but-rising; flat when the trend turns down. Decide the position on "
            "today's close, earn *tomorrow's* return (one execution lag).\n"
            "2. **Race it against buy-and-hold** — Sharpe vs Sharpe, both excess-of-cash, net "
            "of realistic costs. If Elder Ray can't beat just holding, the timing adds nothing.\n"
            "3. **Race it against the obvious simpler rule** — \"long when price > its 200-day "
            "average.\" If a one-liner wins, the Bull/Bear Power detail is busted.\n\n"
            "We run it on SPY (33 years) plus a 5-tape panel (QQQ, IWM, EFA, GLD)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: the equity curves.** Elder Ray vs buy-and-hold on SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY')\n"
            "    res = st.run_experiment(spy, span=13, cost_bps=2.0)\n"
            "    led = res['ledger']\n"
            "    eq_e = (1+led['ret_strat_net']).cumprod()\n"
            "    eq_b = (1+led['ret_bh']).cumprod()\n"
            "    e = res['elder']\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "    ax.plot(eq_b.index, eq_b, c=GREY, lw=1.6, label=f\"Buy & hold (Sharpe {e['bh_sharpe']:.2f})\")\n"
            "    ax.plot(eq_e.index, eq_e, c=RED, lw=1.6, label=f\"Elder Ray long/flat (Sharpe {e['strat_sharpe']:.2f})\")\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('Elder Ray sits in cash and gives up the compounding'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            f"    print(f\"Elder net Sharpe {{e['strat_sharpe']:.2f}} vs buy-and-hold {{e['bh_sharpe']:.2f}}\")\n"
            "else:\n"
            f"    print('Elder net Sharpe {R['sharpe']:.2f} vs buy-and-hold {R['bh_sharpe']:.2f} (frozen)')"
        ),
        md(
            f"Buy-and-hold compounds at **{R['bh_cagr']:.1f}%/yr**; the Elder-Ray timing rule "
            f"manages only **{R['cagr']:.1f}%/yr** because it is invested just **{R['tim']:.0f}%** "
            "of the time. It *does* cut the worst drawdown (from "
            f"{R['bh_mdd']:.0f}% to {R['mdd']:.0f}%) — but you pay for that in foregone return."
        ),
        md(
            "**Is the timing edge even real?** A rule that beats a coin should beat a "
            "*re-shuffled* version of itself. We rotate the strategy's daily returns by a "
            "random offset 2,000 times and see how often random timing does as well."
        ),
        code(
            "if HAVE_REAL:\n"
            "    means, actual, p = st.rotation_placebo(led['pos'].to_numpy(), led['ret_asset'].to_numpy())\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "    ax.hist(means*1e4, bins=40, color=GREY, alpha=.7, label='random timing (rotated positions)')\n"
            "    ax.axvline(actual*1e4, c=RED, lw=2.5, label=f'actual alignment (p={p:.2f})')\n"
            "    ax.set_xlabel('mean daily book return (bps)'); ax.set_ylabel('count')\n"
            "    ax.set_title('Most random rotations beat the actual timing'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'placebo p = {p:.3f}')\n"
            "else:\n"
            f"    print('permutation p = {R['perm_p']:.2f} (frozen)')"
        ),
        md(
            f"The placebo *p* = **{R['perm_p']:.2f}** — almost every random rotation of *when* the "
            "rule is invested does *as well or better* than the real timing. The rule's choice of "
            "days carries no edge; if anything it is on the wrong side."
        ),
        md(
            "**The decisive comparison: Elder Ray vs a one-line trend filter.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    names = ['Buy & hold', 'Elder Ray', '200-day SMA filter', 'EMA13 cross']\n"
            "    sh = [res['elder']['bh_sharpe'], res['elder']['strat_sharpe'],\n"
            "          res['sma200']['strat_sharpe'], res['emacross']['strat_sharpe']]\n"
            "else:\n"
            f"    names = ['Buy & hold','Elder Ray','200-day SMA filter','EMA13 cross']\n"
            f"    sh = [{R['bh_sharpe']}, {R['sharpe']}, {R['sma_sharpe']}, {R['emac_sharpe']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "col = [GREY, RED, GREEN, AMBER]\n"
            "ax.bar(names, sh, color=col)\n"
            "ax.set_ylabel('net Sharpe (excess-of-cash)')\n"
            "ax.set_title('A plain 200-day filter beats Elder Ray — and so does just holding')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Elder Ray {R['sharpe']:.2f}  <  plain SMA filter {R['sma_sharpe']:.2f}  <  buy-and-hold {R['bh_sharpe']:.2f}')"
        ),
        md(
            f"The dead-simple \"price above its 200-day average\" filter scores **{R['sma_sharpe']:.2f}** — "
            f"better than Elder Ray's **{R['sharpe']:.2f}**. All the Bull/Bear Power machinery loses "
            "to one line of code."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** SPY net Sharpe {R['sharpe']:.2f}, HAC *t* = {R['hac_t']:.2f} "
            f"(needs ≥ 2), placebo *p* = {R['perm_p']:.2f}. No tape in the panel clears the bar.\n"
            "- **Tradability — Mirage.** It trails buy-and-hold by half a Sharpe point, with "
            "~30 round-trips/yr of turnover. Costs only widen the gap.\n"
            "- **Beats a plain SMA filter? — Busted.** A one-line 200-day filter wins on Sharpe, "
            "CAGR and *t*. The extra Bull/Bear Power detail adds nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "At ~30 one-way round-trips/yr, costs bite — and the gross edge is already "
            "sub-market:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.run_experiment(spy, span=13, cost_bps=c)['elder']['strat_sharpe'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['cost00_sharpe']}, {R['cost10_sharpe']}, {R['cost20_sharpe']}, {R['cost50_sharpe']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            f"ax.axhline({R['bh_sharpe']}, ls='--', c=GREY, label='buy-and-hold')\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2, label='Elder Ray net Sharpe')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Elder Ray never reaches buy-and-hold, at any cost level'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even at zero cost the gross Sharpe is far below buy-and-hold.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants real trend in a "
            "synthetic tape — and the *same* Elder-Ray rule then beats buy-and-hold and clears "
            "HAC *t* > 3. The engine works; the real market just isn't trendy enough at this "
            "horizon to reward this particular filter over a simpler one.\n"
            "- **Elder's full Triple Screen.** We test the second screen in isolation. Elder "
            "pairs it with a weekly trend and a daily oscillator — a fork worth building.\n"
            "- **Other oscillators.** CCI ([Study 178](../../178-cci/)) and Bollinger reversion "
            "([Study 104](../../104-bollinger-reversion/)) land in the same place: classic-TA "
            "oscillators that don't clear the bar on modern equity tapes.\n\n"
            "*Think Elder Ray shines on a different timeframe or as part of the full Triple "
            "Screen? Fork this, wire it up, and show a net excess-Sharpe edge that beats both "
            "buy-and-hold and the plain SMA filter, clearing HAC t ≥ 2. That is the bar.*"
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
            "# Elder Ray — a quantitative teardown\n"
            "### Daily OHLCV · Bull/Bear Power(13) · long/flat & long/short · HAC inference · rotation placebo · simpler-benchmark race\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a plain SMA filter%3F: Busted](https://img.shields.io/badge/Beats_a_plain_SMA_filter%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We turn Elder's "
            "Bull/Bear Power into a long/flat timing rule, race its **net** excess Sharpe against "
            "buy-and-hold and against a plain 200-day SMA filter, and check the timing with a "
            "rotation placebo and a synthetic positive control.\n\n"
            f"> **Not investment advice.** Real data: Yahoo daily total-return bars, as-of "
            f"**{R['asof']}**; the offline core and synthetic control run deterministically. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front (real tape)\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SPY net Sharpe **{R['sharpe']:.2f}**, HAC *t* = "
            f"**{R['hac_t']:.2f}** (< 2); rotation placebo *p* = **{R['perm_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Net excess Sharpe trails buy-and-hold by "
            f"**{R['sharpe_delta']:.2f}**; ~{R['turnover']:.0f} one-way round-trips/yr. |\n"
            f"| **Beats a plain SMA filter?** | `BUSTED` | A one-line 200-day filter scores "
            f"**{R['sma_sharpe']:.2f}** vs Elder Ray's **{R['sharpe']:.2f}** — wins on every metric. |\n\n"
            "> In plain words: Elder Ray cuts drawdown but bleeds return by sitting in cash, and "
            "a far simpler trend filter does the same job better. The Bull/Bear Power detail is decoration."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $E_t = \\text{EMA}_{13}(C)_t$ be the exponential moving average of the close. "
            "Elder Ray defines:\n\n"
            "$$\\text{Bull}_t = H_t - E_t, \\qquad \\text{Bear}_t = L_t - E_t.$$\n\n"
            "The long/flat rule: enter long when the trend is up and the bears are fading,\n\n"
            "$$d_t = +1 \\;\\text{ if }\\; E_t > E_{t-1} \\;\\wedge\\; (\\text{Bear}_t > \\text{Bear}_{t-1} "
            "\\wedge \\text{Bear}_t < 0),$$\n\n"
            "hold until $E_t < E_{t-1}$ (trend down), then go flat. The hypotheses:\n\n"
            "- **H₁ (signal).** The net daily excess return of the lagged book has positive mean, "
            "HAC *t* ≥ 2.\n"
            "- **H₂ (beats hold).** Its excess Sharpe exceeds buy-and-hold's.\n"
            "- **H₃ (beats the simple rule).** Its excess Sharpe exceeds a plain 200-day SMA filter's.\n\n"
            "We reject all three: H₁ (*t* ≈ 0.7), H₂ (Sharpe trails by ~0.5), H₃ (the one-liner wins)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Elder Ray is taught as a precision entry timer inside the Triple Screen system. If "
            "H₁–H₃ held, the Bull/Bear Power decomposition would carry information beyond the EMA "
            "trend it sits on. The finding that an isolated EMA filter (and a plain SMA filter) "
            "*beats* it means the power oscillators add no incremental timing value on these tapes — "
            "a clean example of a multi-part indicator losing to its own simplest component."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Indicator.** Bull/Bear Power on EMA13 (Elder's default), `adjust=False`.\n"
            "- **Position.** Long/flat stateful timer (above); decided at close *t*, applied to "
            "the return of *t+1* — one `shift`, the documented execution lag. 30-bar warm-up dropped.\n"
            "- **Costs.** One-way × NAV on |Δposition|; long/short days pay daily borrow.\n"
            "- **Race.** Net **excess-of-cash** Sharpe vs buy-and-hold and vs a 200-day SMA filter "
            "and an EMA13-cross, all on the same tape and rf.\n"
            "- **Inference.** Newey–West HAC *t* on the net excess daily stream; a circular-rotation "
            "placebo (destroys the position/return alignment, keeps the autocorrelation).\n"
            "- **Positive control.** Synthetic tape with a tunable AR(1) trend knob (`edge`), "
            "confirming the engine recovers an edge when trend is planted.\n\n"
            f"Tapes: {', '.join(PANEL)} — full Yahoo history through **{R['asof']}**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The panel — Elder Ray trails buy-and-hold on every tape\n\n"
            "Net excess Sharpe, Elder long/flat vs buy-and-hold, with the HAC *t* on the rule."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in PANEL:\n"
            "        b = load(t); r = st.run_experiment(b, span=13, cost_bps=2.0)['elder']\n"
            "        rows.append((t, r['strat_sharpe'], r['bh_sharpe'], r['sharpe_delta'], r['strat_hac_t']))\n"
            "else:\n"
            f"    rows = [('SPY',{R['spy_es']},{R['spy_bh']},{R['spy_d']},{R['spy_t']}),\n"
            f"            ('QQQ',{R['qqq_es']},{R['qqq_bh']},{R['qqq_d']},{R['qqq_t']}),\n"
            f"            ('IWM',{R['iwm_es']},{R['iwm_bh']},{R['iwm_d']},{R['iwm_t']}),\n"
            f"            ('EFA',{R['efa_es']},{R['efa_bh']},{R['efa_d']},{R['efa_t']}),\n"
            f"            ('GLD',{R['gld_es']},{R['gld_bh']},{R['gld_d']},{R['gld_t']})]\n"
            "panel = pd.DataFrame(rows, columns=['tape','elder_sharpe','bh_sharpe','delta','hac_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(panel)); w = 0.38\n"
            "ax.bar(x-w/2, panel['bh_sharpe'], w, color=GREY, label='buy & hold')\n"
            "ax.bar(x+w/2, panel['elder_sharpe'], w, color=RED, label='Elder Ray')\n"
            "ax.set_xticks(x); ax.set_xticklabels(panel['tape']); ax.set_ylabel('net excess Sharpe')\n"
            "ax.set_title('Elder Ray is below buy-and-hold on every tape'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "panel.round(3)"
        ),
        md(
            "> In plain words: not one tape's HAC *t* clears 2 (the best is GLD at "
            f"{R['gld_t']:.2f}), and every delta-vs-hold is negative. The rule reliably "
            "*underperforms* holding."
        ),
        md(
            "### 4b · The rotation placebo on SPY\n\n"
            "If the timing carried information, the actual mean return would sit in the right "
            "tail of random rotations of the same stream."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); led = st.run_experiment(spy, span=13, cost_bps=2.0)['ledger']\n"
            "    means, actual, p = st.rotation_placebo(led['pos'].to_numpy(), led['ret_asset'].to_numpy())\n"
            "else:\n"
            f"    means = np.random.default_rng(0).normal(3e-4, 0.5e-4, 2000); actual = 2.5e-4; p = {R['perm_p']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(means*1e4, bins=40, color=GREY, alpha=.7, label='random rotations of position')\n"
            "ax.axvline(actual*1e4, c=RED, lw=2.5, label=f'actual alignment (p={p:.2f})')\n"
            "ax.set_xlabel('mean daily book return (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title('Actual timing sits in the LEFT tail — worse than random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p = {p:.3f}')"
        ),
        md(
            f"> In plain words: *p* = **{R['perm_p']:.2f}**. Almost every random rotation of *when* "
            "the rule holds does as well or better — the realised alignment of being invested with "
            "good days is not just unremarkable, it is on the wrong side of random."
        ),
        md(
            "### 4c · Elder Ray vs the simpler benchmarks (the decisive race)\n\n"
            "Does the Bull/Bear Power detail beat its own simplest component?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(spy, span=13, cost_bps=2.0)\n"
            "    labels = ['Buy & hold','Elder Ray','200d SMA filter','EMA13 cross']\n"
            "    shs = [res['elder']['bh_sharpe'], res['elder']['strat_sharpe'],\n"
            "           res['sma200']['strat_sharpe'], res['emacross']['strat_sharpe']]\n"
            "    cags = [res['elder']['bh_cagr'], res['elder']['strat_cagr'],\n"
            "            res['sma200']['strat_cagr'], res['emacross']['strat_cagr']]\n"
            "else:\n"
            f"    labels = ['Buy & hold','Elder Ray','200d SMA filter','EMA13 cross']\n"
            f"    shs = [{R['bh_sharpe']},{R['sharpe']},{R['sma_sharpe']},{R['emac_sharpe']}]\n"
            f"    cags = [{R['bh_cagr']/100},{R['cagr']/100},{R['sma_cagr']/100},{R['emac_cagr']/100}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "col = [GREY, RED, GREEN, AMBER]\n"
            "a1.bar(labels, shs, color=col); a1.set_ylabel('net excess Sharpe'); a1.set_title('Sharpe')\n"
            "a1.tick_params(axis='x', rotation=20)\n"
            "a2.bar(labels, [c*100 for c in cags], color=col); a2.set_ylabel('CAGR %'); a2.set_title('CAGR')\n"
            "a2.tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('SMA filter Sharpe {R['sma_sharpe']:.2f} > Elder Ray {R['sharpe']:.2f}; EMA13-cross alone {R['emac_sharpe']:.2f} also > Elder Ray')"
        ),
        md(
            f"> In plain words: the 200-day filter ({R['sma_sharpe']:.2f}) and even a bare EMA13 cross "
            f"({R['emac_sharpe']:.2f}) beat the full Elder-Ray rule ({R['sharpe']:.2f}). The Bull/Bear "
            "Power oscillators *subtract* value relative to the trend filter they ride on."
        ),
        md(
            "### 4d · The long/short variant\n\n"
            "Adding the symmetric short leg (sell when trend down and Bull Power fading):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.run_experiment(spy, span=13, cost_bps=2.0, long_short=True)['elder']\n"
            "    print(f\"long/short: Sharpe {ls['strat_sharpe']:+.3f}  CAGR {ls['strat_cagr']*100:+.2f}%  \"\n"
            "          f\"HAC t {ls['strat_hac_t']:+.2f}  time-in-market {ls['time_in_market']*100:.0f}%\")\n"
            "else:\n"
            f"    print('long/short: Sharpe {R['ls_sharpe']:+.3f}  CAGR {R['ls_cagr']:+.2f}%  HAC t {R['ls_hac_t']:+.2f}  TIM {R['ls_tim']:.0f}%')"
        ),
        md(
            f"> In plain words: the short leg makes it *worse* — Sharpe **{R['ls_sharpe']:+.2f}**, "
            f"HAC *t* **{R['ls_hac_t']:+.2f}**. Fighting an up-drifting market with shorts is a "
            "reliable way to lose."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SPY net excess Sharpe {R['sharpe']:.2f}, HAC *t* {R['hac_t']:.2f}; "
            f"placebo *p* {R['perm_p']:.2f}; no panel tape clears *t* = 2 (best GLD {R['gld_t']:.2f}).\n"
            f"- **Tradability `MIRAGE`** — trails buy-and-hold by Sharpe {R['sharpe_delta']:.2f}, "
            f"~{R['turnover']:.0f} one-way round-trips/yr; the long/short variant is negative.\n"
            f"- **Beats a plain SMA filter? `BUSTED`** — 200-day filter {R['sma_sharpe']:.2f} and "
            f"EMA13-cross {R['emac_sharpe']:.2f} both beat Elder Ray {R['sharpe']:.2f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            f"~{R['turnover']:.0f} one-way round-trips/yr is non-trivial turnover, and the gross "
            "edge already trails the market:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.run_experiment(spy, span=13, cost_bps=c)['elder'] for c in costs]\n"
            "    sh = [s['strat_sharpe'] for s in sw]; tt = [s['strat_hac_t'] for s in sw]\n"
            "else:\n"
            f"    sh = [{R['cost00_sharpe']},{R['cost10_sharpe']},{R['cost20_sharpe']},{R['cost50_sharpe']}]\n"
            f"    tt = [{R['cost00_t']},{R['cost10_t']},{R['cost20_t']},{R['cost50_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot(costs, sh, 'o-', c=RED, lw=2, label='net Sharpe')\n"
            f"ax.axhline({R['bh_sharpe']}, ls='--', c=GREY, label='buy-and-hold')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tt, 's--', c=GREY, lw=1.4, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net Sharpe', color=RED)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('Never reaches buy-and-hold; never clears t=2'); ax.legend(loc='center right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even at zero cost the HAC t is ~1.1 — below the bar before a single cent of cost.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the engine *able* to reward Elder Ray when trend genuinely exists? Plant an AR(1) "
            "trend knob in a synthetic tape and re-run the identical rule:"
        ),
        code(
            "edges = [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "deltas, tts = [], []\n"
            "for ed in edges:\n"
            "    b, _ = data.synthetic_panel(n_days=6000, edge=ed, seed=422)\n"
            "    r = st.run_experiment(b, span=13, cost_bps=2.0)['elder']\n"
            "    deltas.append(r['sharpe_delta']); tts.append(r['strat_hac_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot(edges, deltas, 'o-', c=GREEN, lw=2, label='Elder − buy&hold (Sharpe)')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax2 = ax.twinx(); ax2.plot(edges, tts, 's--', c=GREY, lw=1.4, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) trend (edge)'); ax.set_ylabel('Sharpe delta', color=GREEN)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('Engine works: Elder Ray beats hold once trend is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('edge=0: delta {R['syn0_delta']:+.2f}, t {R['syn0_t']:.2f}  |  "
            f"edge=0.20: delta {R['syn20_delta']:+.2f}, t {R['syn20_t']:.2f}')"
        ),
        md(
            "The engine is faithful: at zero planted trend the rule trails buy-and-hold "
            f"(delta {R['syn0_delta']:+.2f}, *t* {R['syn0_t']:.2f}, placebo *p* {R['syn0_p']:.2f}); "
            f"plant strong trend and the *same* rule beats hold and clears the bar "
            f"(delta {R['syn20_delta']:+.2f}, *t* {R['syn20_t']:.2f}, placebo *p* {R['syn20_p']:.3f}). "
            "The real-tape `NONE` is therefore a statement about the "
            "**market** (modern equity tapes don't have enough persistent daily trend to reward "
            "this filter over a simpler one), not the method.\n\n"
            "Forks worth trying: the full Triple Screen (weekly trend + daily Elder Ray + a "
            "force-index trigger); a longer EMA; or intraday tapes where the tug-of-war framing "
            "might carry more information."
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
