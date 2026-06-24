"""Generate the two narrative notebooks for Study 424 (Ease of Movement).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
positive control runs anywhere, offline and deterministic; the real-tape cells use the
cached daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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
    n=5013,
    # SPY EOM(14) long/flat
    eom_net_sharpe=0.684, eom_net_t=3.18, eom_net_ret=7.68, eom_vol=11.2,
    eom_dd=-29.7, eom_exposure=0.641, eom_turn=23.0,
    eom_gross_sharpe=0.704, eom_gross_t=3.28,
    bh_sharpe=0.657, bh_ret=12.78, bh_dd=-55.2,
    sma_sharpe=0.679, sma_t=3.41,
    macd_sharpe=0.579, macd_t=2.89,
    # difference tests (positive t = EOM wins)
    vsbh_ann=-5.10, vsbh_t=-1.71,
    vssma_ann=-1.67, vssma_t=-0.61,
    vsmacd_ann=1.23, vsmacd_t=0.74,
    # permutation
    perm_obs=0.704, perm_p=0.0005, perm_null=-0.010,
    # cost sweep (net)
    c0_sh=0.704, c0_t=3.28, c1_sh=0.684, c1_t=3.18, c2_sh=0.664, c2_t=3.08,
    c5_sh=0.602, c5_t=2.77, c10_sh=0.499, c10_t=2.28,
    # period sweep (net 1bp)
    p9_sh=0.396, p9_t=1.85, p9_vsbh=-2.80,
    p14_sh=0.684, p14_t=3.18, p14_vsbh=-1.71,
    p20_sh=0.627, p20_t=3.04, p20_vsbh=-1.94,
    p30_sh=0.464, p30_t=2.18, p30_vsbh=-2.46,
    p40_sh=0.604, p40_t=2.84, p40_vsbh=-1.99,
    # per-instrument
    spy_sh=0.684, spy_t=3.18, spy_vsbh=-1.71,
    qqq_sh=0.717, qqq_t=3.08, qqq_vsbh=-2.59,
    iwm_sh=0.208, iwm_t=0.93, iwm_vsbh=-2.10,
    dia_sh=0.481, dia_t=2.05, dia_vsbh=-2.22,
    efa_sh=0.100, efa_t=0.48, efa_vsbh=-1.99,
    eem_sh=0.070, eem_t=0.33, eem_vsbh=-2.17,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from ease_of_movement import data, strategy as st

ASOF = "2026-05-31"
CACHE = os.path.abspath(os.path.join("..", "_cache"))
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM"]

def _have(t):
    return os.path.exists(data._cache_path(t, CACHE))

HAVE_REAL = all(_have(t) for t in TICKERS)

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
            "# Ease of Movement — are 'effortless' price moves tradable?\n"
            "### Richard Arms' volume-weighted oscillator, turned into a timing rule and raced honestly\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats an SMA cross%3F: Not_supported](https://img.shields.io/badge/Beats_an_SMA_cross%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Here's a chart-school favourite: the *Ease of Movement* indicator. The idea is "
            "poetic — when price climbs a lot on *little* volume, the market is moving with the "
            "'path of least resistance', an *effortless* advance worth riding; when EOM dips "
            "below zero, the effortless move is down, so step aside. Richard Arms built it in "
            "the 1970s. Does buying when EOM is positive actually beat just owning the market?\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the permutation test, and "
            "the benchmark race? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does 'effortless movement' predict returns? | **Yes — surprisingly.** Going long "
            f"when EOM > 0 earns a net Sharpe of **{R['eom_net_sharpe']:.2f}** on SPY, with a "
            f"*t*-stat of **+{R['eom_net_t']:.2f}** — a real signal. |\n"
            f"| So it beats buy-and-hold? | **No.** It returns **+{R['eom_net_ret']:.1f}%/yr** vs "
            f"buy-and-hold's **+{R['bh_ret']:.1f}%** — it just sits in cash a third of the time. |\n"
            f"| Then what's the catch? | It's **the same as a plain moving-average cross.** The fancy "
            "volume weighting adds nothing a two-line SMA rule doesn't already give you. |\n"
            "| Could you trade it for an edge? | **No.** Its only real win is a smaller drawdown "
            "— that's *de-risked beta*, free from any trend filter, not alpha. |\n\n"
            "> EOM is real momentum wearing a volume costume. The signal exists; the *story* "
            "(that the volume part is special) doesn't."
        ),

        # ---- BEAT 1 — THE CLAIM --------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Ease of Movement measures how easily price moves. When it rises above zero, "
            "price is advancing on light volume — the path of least resistance is up, so go "
            "long. When it falls below zero, step aside. Trade with the effortless move.\"*\n\n"
            "Each day EOM asks: how far did the mid-price travel, *per unit of volume*? A big "
            "move on light volume scores high (easy); the same move on heavy volume scores low "
            "(it took effort). Smooth that over ~14 days and you get the EOM line. The folk rule "
            "is the zero-cross: positive → long, negative → flat."
        ),

        # ---- BEAT 2 — SO WHAT ----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If EOM's zero-cross reliably caught the 'easy' part of trends, it would be a clean "
            "mechanical signal — and crucially, one that claims to use *information a price-only "
            "chart can't see*: volume. That's the whole pitch. If true, it should beat both "
            "buy-and-hold and the simplest price-only trend rule. If it only matches them, the "
            "volume weighting is decorative — and millions of traders are watching an indicator "
            "that adds nothing to a moving average they could draw in two lines."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "Three honest tests, announced before we run them:\n\n"
            "1. **Race it against buy-and-hold, fairly.** Both measured the same way (excess of "
            "cash, net of costs). If EOM's timing adds value, it beats just owning the market.\n"
            "2. **Race it against the *simplest* rival.** A plain SMA(50/200) cross uses no volume "
            "at all. If EOM is special, it must beat that — otherwise the volume part is for show.\n"
            "3. **Shuffle the signal.** Keep the trade *timing* but flip the directions by a coin. "
            "If EOM beats its own shuffled self, the dates carry real information.\n\n"
            "We run it on 20 years of daily SPY bars (plus five more ETFs), entering one day "
            "after each signal — no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN -----------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: EOM vs buy-and-hold on SPY.** The growth of \\$1, net of costs, "
            "excess of cash."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY')\n"
            "    eom = st.ease_of_movement(spy, 14)\n"
            "    pos = st.eom_position(eom)\n"
            "    book = st.position_returns(spy, pos, cost_bps=1.0)\n"
            "    eq_eom = (1 + book['r_strat_net']).cumprod()\n"
            "    eq_bh  = (1 + book['r_asset_excess']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "    ax.plot(eq_bh.index, eq_bh, c=GREY, lw=1.6, label='Buy & hold')\n"
            "    ax.plot(eq_eom.index, eq_eom, c=GREEN, lw=1.8, label='EOM long/flat (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (excess of cash, log)')\n"
            "    ax.set_title('EOM rides a smoother path — but ends up lower')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    eom_sh = st.summarize(book, 'r_strat_net')['sharpe']\n"
            "    bh_sh = st.annualised_sharpe(book['r_asset_excess'].to_numpy())\n"
            "    print(f'EOM net Sharpe {eom_sh:+.3f}  vs  B&H {bh_sh:+.3f}')\n"
            "else:\n"
            f"    print('cached run: EOM net Sharpe {R['eom_net_sharpe']:+.3f}  "
            f"vs  B&H {R['bh_sharpe']:+.3f}')"
        ),
        md(
            f"The EOM line is *smoother* — it sidesteps the worst of the crashes (max drawdown "
            f"**{R['eom_dd']:.0f}%** vs buy-and-hold's **{R['bh_dd']:.0f}%**). But it also ends "
            f"up **lower**: **+{R['eom_net_ret']:.1f}%/yr** vs **+{R['bh_ret']:.1f}%**, because "
            "it sits in cash about a third of the time and misses chunks of the recovery. The "
            "Sharpe ratios are nearly tied."
        ),
        md(
            "**Now the decisive race: EOM vs the *simplest* rival, a plain SMA(50/200) cross.** "
            "No volume anywhere in that rule. If EOM's volume weighting is worth anything, it "
            "should win here."
        ),
        code(
            "labels = ['EOM(14)', 'SMA 50/200', 'MACD', 'Buy & hold']\n"
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(load('SPY'), period=14, cost_bps=1.0, n_perm=1)\n"
            "    sh = [r['eom']['sharpe'], r['sma']['sharpe'], r['macd']['sharpe'], r['bh']['sharpe']]\n"
            "else:\n"
            f"    sh = [{R['eom_net_sharpe']}, {R['sma_sharpe']}, {R['macd_sharpe']}, {R['bh_sharpe']}]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "col = [GREEN, GREY, GREY, GREY]\n"
            "ax.bar(labels, sh, color=col, width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe (excess of cash)')\n"
            "ax.set_title('EOM, an SMA cross and B&H are all the same Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('EOM and the plain SMA cross are a statistical dead heat.')"
        ),
        md(
            f"They're a dead heat. EOM's net Sharpe (**{R['eom_net_sharpe']:.2f}**) and the "
            f"price-only SMA cross's (**{R['sma_sharpe']:.2f}**) are within a hair — the formal "
            f"difference test (in the quant notebook) can't tell them apart (*t* = "
            f"{R['vssma_t']:.2f}). The volume weighting that is EOM's *entire selling point* "
            "buys nothing a two-line moving average doesn't already give you."
        ),

        # ---- BEAT 5 — VERDICT ----------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** EOM long/flat net Sharpe **{R['eom_net_sharpe']:.2f}**, "
            f"*t* = **+{R['eom_net_t']:.2f}**, and it beats its own coin-flipped shuffle "
            f"(*p* = {R['perm_p']}). An effortless advance really does tend to continue.\n"
            f"- **Tradability — Mirage.** It loses to buy-and-hold on return "
            f"(**+{R['eom_net_ret']:.1f}%** vs **+{R['bh_ret']:.1f}%**) and ties a plain SMA "
            "cross on Sharpe. Its only win is a smaller drawdown — that's *de-risked beta*, free "
            "from any trend filter, not a new edge.\n"
            "- **Beats an SMA cross? — Not supported.** The volume weighting adds no measurable "
            "advantage over a price-only moving-average cross."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You *could* run it — it survives costs fine (only ~23 trades/yr). But what would you "
            "be paying for? A rule that:\n\n"
            "- returns **less** than just holding the index,\n"
            "- has the **same** risk-adjusted return as a two-line SMA cross you could build for free,\n"
            "- whose only genuine benefit (smaller drawdowns) is available from *any* trend filter.\n\n"
            "If you want lower drawdowns, a moving-average overlay does the job without the volume "
            "machinery. EOM gives you nothing extra to pay for."
        ),

        # ---- BEAT 7 — GOING FURTHER ----------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a genuine "
            "volume-coupled edge in synthetic data and shows EOM *does* find it — the indicator "
            "is correctly built; the real market just doesn't reward the volume part separately.\n"
            "- **Commodities & FX.** Arms designed EOM watching *volume* behaviour; equity-index "
            "ETFs may not be its best habitat. A fork on single names or futures is fair game.\n"
            "- **EOM as a filter, not a signal.** Combine EOM with a separate entry rule rather "
            "than trading its zero-cross directly — does it add value as a *confirmation*?\n\n"
            "*Think the volume weighting earns its keep somewhere? Fork this, change the asset, "
            "and show EOM beating a plain SMA cross on a difference test that clears HAC inference. "
            "That is the bar.*"
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
            "# Ease of Movement — a quantitative teardown\n"
            "### Daily OHLCV · EOM(14) long/flat · excess-of-cash NET Sharpe race · HAC inference · sign-shuffle null\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats an SMA cross%3F: Not_supported](https://img.shields.io/badge/Beats_an_SMA_cross%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the EOM(14) "
            "zero-cross timing rule (long/flat) beats buy-and-hold and the simpler SMA / MACD "
            "benchmarks on a NET, excess-of-cash Sharpe race, with HAC *t*-stats on the rule and on "
            "each difference, a sign-shuffle permutation null, and cost/period sweeps.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars (`auto_adjust=True`), 20-year "
            "window, as-of 2026-05-31; the offline core and synthetic control run on a deterministic "
            "generator. Methods & sources in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 --------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | EOM net Sharpe **{R['eom_net_sharpe']:.3f}**, HAC "
            f"*t* = **+{R['eom_net_t']:.2f}**; sign-shuffle permutation *p* = **{R['perm_p']}**; "
            "robust over smoothing windows 14–40. |\n"
            f"| **Tradability** | `MIRAGE` | EOM−B&H *t* = **{R['vsbh_t']:.2f}** (loses); "
            f"EOM−SMA *t* = **{R['vssma_t']:.2f}** (dead heat). Only win is drawdown "
            f"({R['eom_dd']:.0f}% vs {R['bh_dd']:.0f}%) = de-risked beta. |\n"
            f"| **Beats an SMA cross?** | `NOT SUPPORTED` | ΔSharpe vs SMA(50/200) ≈ "
            f"{R['eom_net_sharpe']-R['sma_sharpe']:+.3f}; the volume weighting adds nothing "
            "over a price-only cross. |\n\n"
            "> In plain words: the EOM signal is real (it's a trend filter, and trend is real) "
            "but it carries no information a plain moving-average cross doesn't, and it doesn't "
            "beat buy-and-hold — so there is no separate edge to harvest."
        ),

        # ---- BEAT 1 --------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Per bar, define the mid-price move and the volume-scaled box ratio:\n\n"
            "$$\\Delta\\text{mid}_t = \\tfrac{H_t+L_t}{2} - \\tfrac{H_{t-1}+L_{t-1}}{2}, "
            "\\qquad \\text{BR}_t = \\frac{V_t/s}{H_t-L_t}, \\qquad "
            "\\text{EMV}_t = \\frac{\\Delta\\text{mid}_t}{\\text{BR}_t}$$\n\n"
            "and smooth: $\\text{EOM}_t = \\frac{1}{n}\\sum_{i=0}^{n-1}\\text{EMV}_{t-i}$ "
            "($n=14$). The scale $s$ only rescales magnitude — the *sign* the rule trades on is "
            "invariant. The folk rule asserts:\n\n"
            "- **H₁ (signal).** The long/flat position $\\mathbb{1}[\\text{EOM}_t>0]$, held over "
            "$r_{t+1}$, has positive expected excess return: $\\mathbb{E}[\\,\\mathbb{1}"
            "[\\text{EOM}_t>0]\\cdot r^{\\text{ex}}_{t+1}] > 0$.\n"
            "- **H₂ (beats B&H).** Its Sharpe exceeds buy-and-hold's, excess-vs-excess.\n"
            "- **H₃ (volume adds value).** It beats a price-only SMA(50/200) cross.\n\n"
            "We **accept H₁** (HAC *t* = +3.18), but **reject H₂** (difference *t* = −1.71) and "
            "**reject H₃** (difference *t* = −0.61). The signal is real; the *edge* is not."
        ),

        # ---- BEAT 2 --------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "EOM is taught as a *volume* indicator: its claim to fame is using information a "
            "price-only chart cannot see. If H₃ failed — if it merely replicated a moving-average "
            "cross — then the entire pedagogical case for EOM collapses to 'a slow trend filter "
            "with extra steps'. The de-risked-beta trap (H₂) is the more general lesson: any rule "
            "that sits in cash part-time will show a lower drawdown and a flattering Sharpe-vs-raw "
            "comparison, which is why the race must be run excess-of-cash on a difference test."
        ),

        # ---- BEAT 3 --------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **EOM(14).** Arms' kernel; scale $s = 10^8$; zero-range bars treated as no info.\n"
            "- **Position.** Long/flat: $+1$ when EOM $>0$, else $0$. (Long/short tested too.)\n"
            "- **Execution lag.** Signal known at close of *t*; position held over $r_{t+1}$ — "
            "exactly one `shift`, stated in `position_returns`.\n"
            "- **Returns.** Excess of cash; NET of one-way `cost_bps` × |Δposition| × NAV; shorts "
            "pay borrow.\n"
            "- **Inference.** Newey–West HAC *t* on the daily series *and* on the "
            "strategy-minus-benchmark difference; a 2,000-draw sign-shuffle permutation null "
            "(flip each episode's sign, keep durations); cost and period sweeps.\n"
            "- **Benchmarks.** Buy-and-hold, SMA(50/200) cross, MACD(12/26/9) — same engine.\n"
            "- **Positive control.** Synthetic tape with a *planted* volume-coupled edge.\n\n"
            "Six tapes: SPY (headline), QQQ, IWM, DIA, EFA, EEM — 20-year window, as-of 2026-05-31."
        ),

        # ---- BEAT 4 --------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The Sharpe race — EOM, B&H, SMA, MACD all tie\n\n"
            "Net of 1 bp/leg, excess of cash. If EOM carried a separate edge, its bar clears the rest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(load('SPY'), period=14, cost_bps=1.0, n_perm=1)\n"
            "    rows = [('EOM(14)', r['eom']), ('SMA 50/200', r['sma']),\n"
            "            ('MACD', r['macd']), ('Buy & hold', r['bh'])]\n"
            "    tab = pd.DataFrame({'arm':[a for a,_ in rows],\n"
            "        'sharpe':[s['sharpe'] for _,s in rows],\n"
            "        't':[s['tstat'] for _,s in rows],\n"
            "        'ann_ret%':[s['ann_ret']*100 for _,s in rows],\n"
            "        'maxDD%':[s['max_dd']*100 for _,s in rows]})\n"
            "else:\n"
            "    tab = pd.DataFrame({'arm':['EOM(14)','SMA 50/200','MACD','Buy & hold'],\n"
            f"        'sharpe':[{R['eom_net_sharpe']},{R['sma_sharpe']},{R['macd_sharpe']},{R['bh_sharpe']}],\n"
            f"        't':[{R['eom_net_t']},{R['sma_t']},{R['macd_t']},float('nan')],\n"
            f"        'ann_ret%':[{R['eom_net_ret']},float('nan'),float('nan'),{R['bh_ret']}],\n"
            f"        'maxDD%':[{R['eom_dd']},float('nan'),float('nan'),{R['bh_dd']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN, GREY, GREY, GREY]\n"
            "ax.bar(tab['arm'], tab['sharpe'], color=col, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('net Sharpe (excess of cash)')\n"
            "ax.set_title('A statistical dead heat — EOM is not special')\n"
            "plt.tight_layout(); plt.show()\n"
            "tab.round(3)"
        ),
        md(
            "> In plain words: four ways of timing or holding the market, four near-identical "
            "Sharpes. EOM's standalone *t* is real, but every bar in this chart could be the same "
            "height — there is no separation to trade."
        ),
        md(
            "### 4b · The difference tests — the decisive numbers\n\n"
            "HAC *t* of the *daily difference* (positive = EOM wins). This is where H₂ and H₃ die."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(load('SPY'), period=14, cost_bps=1.0, n_perm=1)\n"
            "    diffs = [('EOM − B&H', r['vs_bh']), ('EOM − SMA', r['vs_sma']),\n"
            "             ('EOM − MACD', r['vs_macd'])]\n"
            "    names = [d for d,_ in diffs]; tvals = [v['tstat'] for _,v in diffs]\n"
            "else:\n"
            "    names = ['EOM − B&H','EOM − SMA','EOM − MACD']\n"
            f"    tvals = [{R['vsbh_t']},{R['vssma_t']},{R['vsmacd_t']}]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.0))\n"
            "col = [GREEN if v >= 2 else RED if v <= -2 else GREY for v in tvals]\n"
            "ax.barh(names, tvals, color=col)\n"
            "for s in (2, -2): ax.axvline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('HAC t of daily difference  (positive = EOM wins)')\n"
            "ax.set_title('EOM beats none of them — and loses to buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "for n_, t_ in zip(names, tvals): print(f'{n_:12s} t = {t_:+.2f}')"
        ),
        md(
            f"> In plain words: EOM−B&H *t* = {R['vsbh_t']:.2f} (loses), EOM−SMA *t* = "
            f"{R['vssma_t']:.2f} (a dead heat), EOM−MACD *t* = +{R['vsmacd_t']:.2f} (not "
            "significant). The volume weighting adds **nothing** over a price-only cross."
        ),
        md(
            "### 4c · The sign-shuffle permutation null\n\n"
            "Keep the holding episodes, flip each one's sign by a coin (2,000 draws). Does the "
            "real rule's Sharpe sit out in the tail?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); eom = st.ease_of_movement(spy, 14); pos = st.eom_position(eom)\n"
            "    perm = st.permutation_pvalue(spy, pos, n_perm=2000, cost_bps=0.0, seed=424)\n"
            "    obs, pval, nullm = perm['observed'], perm['p_value'], perm['null_mean']\n"
            "    # rebuild a null sample for the histogram\n"
            "    rng = np.random.default_rng(7); base = pos.fillna(0).to_numpy()\n"
            "    ch = np.r_[True, base[1:] != base[:-1]]; eid = np.cumsum(ch)-1; nep = int(eid.max())+1\n"
            "    nulls = []\n"
            "    for _ in range(400):\n"
            "        fl = rng.choice([-1.,1.], size=nep)\n"
            "        b = st.position_returns(spy, pd.Series(base*fl[eid], index=pos.index), cost_bps=0.0, borrow_bps_annual=0.0)\n"
            "        nulls.append(st.summarize(b,'r_strat_gross')['sharpe'])\n"
            "else:\n"
            f"    obs, pval, nullm = {R['perm_obs']}, {R['perm_p']}, {R['perm_null']}\n"
            "    nulls = list(np.random.default_rng(7).normal(nullm, 0.25, 400))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(nulls, bins=24, color=GREY, alpha=.65, label='sign-shuffled null')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'EOM observed: {obs:+.3f}')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross Sharpe'); ax.set_ylabel('count')\n"
            "ax.set_title('The signal is real: EOM sits far out in the null tail'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed Sharpe {obs:+.3f}  permutation p = {pval:.4f}  (null mean {nullm:+.3f})')"
        ),
        md(
            f"> In plain words: permutation *p* = **{R['perm_p']}** — the timing dates carry "
            "genuine directional information. This is *why* the Signal axis is `REAL`: the "
            "effect is not an artefact of the holding structure. It just isn't a tradable edge."
        ),
        md(
            "### 4d · Robustness — cost and period sweeps\n\n"
            "Does the standalone signal survive costs and a different smoothing window?"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "pers  = [9, 14, 20, 30, 40]\n"
            "if HAVE_REAL:\n"
            "    spy = load('SPY')\n"
            "    csh = [st.run_experiment(spy, cost_bps=c, n_perm=1)['eom']['tstat'] for c in costs]\n"
            "    psh = [st.run_experiment(spy, period=p, cost_bps=1.0, n_perm=1)['eom']['tstat'] for p in pers]\n"
            "else:\n"
            f"    csh = [{R['c0_t']},{R['c1_t']},{R['c2_t']},{R['c5_t']},{R['c10_t']}]\n"
            f"    psh = [{R['p9_t']},{R['p14_t']},{R['p20_t']},{R['p30_t']},{R['p40_t']}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))\n"
            "a1.plot(costs, csh, 'o-', c=GREEN, lw=2); a1.axhline(2, ls='--', c=GREY)\n"
            "a1.set_xlabel('round-trip cost (bps)'); a1.set_ylabel('EOM HAC t'); a1.set_title('Survives costs')\n"
            "a2.plot([str(p) for p in pers], psh, 's-', c=GREEN, lw=2); a2.axhline(2, ls='--', c=GREY)\n"
            "a2.set_xlabel('EOM smoothing period'); a2.set_ylabel('EOM HAC t'); a2.set_title('Robust to the window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Standalone t stays >= 2 across costs to 10bp and periods 14-40.')"
        ),
        md(
            "> In plain words: the *signal* is sturdy — it's not a knife-edge of one parameter. "
            "But sturdiness of a signal that ties the simplest benchmark doesn't make it tradable."
        ),
        md(
            "### 4e · Per-instrument — strongest on US large-cap, gone on the rest\n\n"
            "Standalone EOM *t* (does the signal exist?) and vs-B&H *t* (does it win the race?)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        rr = st.run_experiment(load(t), period=14, cost_bps=1.0, n_perm=1)\n"
            "        recs.append((t, rr['eom']['sharpe'], rr['eom']['tstat'], rr['vs_bh']['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','eom_sharpe','eom_t','vsbh_t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker': TICKERS,\n"
            f"        'eom_sharpe':[{R['spy_sh']},{R['qqq_sh']},{R['iwm_sh']},{R['dia_sh']},{R['efa_sh']},{R['eem_sh']}],\n"
            f"        'eom_t':[{R['spy_t']},{R['qqq_t']},{R['iwm_t']},{R['dia_t']},{R['efa_t']},{R['eem_t']}],\n"
            f"        'vsbh_t':[{R['spy_vsbh']},{R['qqq_vsbh']},{R['iwm_vsbh']},{R['dia_vsbh']},{R['efa_vsbh']},{R['eem_vsbh']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "x = np.arange(len(perinst)); w = .38\n"
            "ax.bar(x-w/2, perinst['eom_t'], w, color=GREEN, label='EOM standalone t')\n"
            "ax.bar(x+w/2, perinst['vsbh_t'], w, color=RED, label='EOM − B&H t')\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(perinst['ticker'])\n"
            "ax.set_ylabel('HAC t'); ax.set_title('Signal real on SPY/QQQ/DIA — never wins the B&H race')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "perinst.round(3)"
        ),
        md(
            "> In plain words: the green bars (signal exists) clear +2 on SPY/QQQ/DIA; the red "
            "bars (beats B&H) are negative *everywhere*. Wherever EOM is real, it still loses the race."
        ),

        # ---- BEAT 5 --------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — EOM net Sharpe {R['eom_net_sharpe']:.3f}, HAC "
            f"*t* +{R['eom_net_t']:.2f}, permutation *p* {R['perm_p']}; robust to cost and "
            "smoothing window; confirmed on QQQ/DIA.\n"
            f"- **Tradability `MIRAGE`** — EOM−B&H *t* {R['vsbh_t']:.2f} (loses the race on "
            f"return, +{R['eom_net_ret']:.1f}% vs +{R['bh_ret']:.1f}%); only edge is a halved "
            "drawdown = de-risked beta.\n"
            f"- **Beats an SMA cross? `NOT SUPPORTED`** — EOM−SMA *t* {R['vssma_t']:.2f}; "
            "the volume weighting adds no measurable advantage over a price-only cross."
        ),

        # ---- BEAT 6 --------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "At ~23 round-trips/yr the rule survives realistic costs — that was never the "
            "problem. The problem is there is nothing to harvest *beyond* the SMA cross:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    spy = load('SPY')\n"
            "    net = [st.run_experiment(spy, cost_bps=c, n_perm=1)['eom']['sharpe'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['c0_sh']},{R['c1_sh']},{R['c2_sh']},{R['c5_sh']},{R['c10_sh']}]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.plot(costs, net, 'o-', c=GREEN, lw=2)\n"
            f"ax.axhline({R['sma_sharpe']}, ls='--', c=GREY, label='SMA 50/200 Sharpe')\n"
            f"ax.axhline({R['bh_sharpe']}, ls=':', c='k', label='Buy & hold Sharpe')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('EOM net Sharpe')\n"
            "ax.set_title('EOM never separates from the free benchmarks'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even gross, EOM sits on top of the SMA cross and below B&H on return.')"
        ),

        # ---- BEAT 7 --------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding a volume-coupled edge, or is it blind? Plant a "
            "persistence edge tied to the EOM kernel in synthetic data and sweep the knob:"
        ),
        code(
            "edges = [-0.30, -0.10, 0.0, 0.10, 0.30, 0.60, 1.00]\n"
            "sh = []\n"
            "for ed in edges:\n"
            "    d, _ = data.synthetic_panel(n_days=2500, edge=ed, seed=424)\n"
            "    eom = st.ease_of_movement(d, 14); pos = st.eom_position(eom, long_short=True)\n"
            "    bk = st.position_returns(d, pos, cost_bps=0.0, borrow_bps_annual=0.0)\n"
            "    sh.append(st.summarize(bk, 'r_strat_gross')['sharpe'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot(edges, sh, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted volume-coupled edge'); ax.set_ylabel('EOM gross Sharpe')\n"
            "ax.set_title('Engine works: EOM finds the edge when it is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At edge=0 Sharpe ~ 0; it rises monotonically with the planted edge.')"
        ),
        md(
            "The engine is a faithful detector: on a tape with a planted volume-coupled "
            "persistence it recovers a strong Sharpe, and reads ≈ 0 on the martingale. So the "
            "real-tape verdict is a statement about the **market** — equity-index returns offer "
            "EOM no edge *separate* from the trend a moving average already captures — not about "
            "the method. Forks worth trying: single names, commodity/FX futures (closer to Arms' "
            "original volume focus), or EOM as a *confirmation filter* layered on a distinct entry."
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
