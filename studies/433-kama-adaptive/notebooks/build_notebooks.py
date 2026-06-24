"""Generate the two narrative notebooks for Study 433 (Kaufman Adaptive MA).

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
    asof="2026-05-31",
    spy_n=8389,
    # Headline SPY, long/flat, net 1bp
    kama_sh=0.203, sma_sh=0.530, bh_sh=0.650,
    kama_cagr=1.66, sma_cagr=5.32, bh_cagr=10.90,   # in %
    kama_mdd=-58.4, sma_mdd=-34.8, bh_mdd=-55.2,    # in %
    kama_turn=1269, sma_turn=765,
    ks_bps=-1.387, ks_t=-2.98,
    kbh_bps=-3.884, kbh_t=-4.61,
    perm_p=0.9945,
    # Panel KS_t
    panel=dict(SPY=(0.203, 0.530, 0.650, -2.98), QQQ=(0.181, 0.521, 0.523, -2.79),
               IWM=(-0.089, 0.243, 0.472, -2.86), EFA=(0.168, 0.421, 0.406, -2.23),
               GLD=(0.464, 0.436, 0.666, 0.07), AAPL=(0.579, 0.766, 0.628, -2.49)),
    # Cost sweep (cost, kama_sh, sma_sh, ks_bps, ks_t)
    costs=[(0.0, 0.237, 0.551, -1.327, -2.87), (0.5, 0.220, 0.540, -1.357, -2.92),
           (1.0, 0.203, 0.530, -1.387, -2.98), (2.0, 0.169, 0.509, -1.447, -3.09),
           (5.0, 0.067, 0.445, -1.627, -3.41)],
    # Synthetic control (edge, kama_sh, sma_sh, ks_bps, ks_t) at cost 2bp
    syn=[(0.0, 0.100, 0.008, 0.451, 0.81), (2.0, 3.658, 4.036, -0.643, -3.38),
         (4.0, 6.881, 6.795, 0.272, 1.15), (8.0, 9.589, 9.180, 1.120, 2.86)],
    # Long/short SPY
    ls_kama=-0.411, ls_sma=-0.032, ls_t=-3.01,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#3498db"

from kama_adaptive import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))
ASOF = "2026-05-31"
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "AAPL"]

def have_cache():
    return os.path.exists(data._cache_path("SPY", CACHE))

HAVE_REAL = have_cache()

def get(t):
    b = data.load_real(t, cache_dir=CACHE)
    return b[b.index <= ASOF]

print("real daily cache present:", HAVE_REAL)
"""


def _panel_lit():
    rows = "".join(
        f"        '{t}': ({k}, {s}, {bh}, {tt}),\n"
        for t, (k, s, bh, tt) in R["panel"].items()
    )
    return rows


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# KAMA — does a moving average that *adapts* beat a plain one?\n"
            "### Kaufman's Adaptive Moving Average, raced head-to-head against a fixed SMA\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a fixed SMA%3F: Busted](https://img.shields.io/badge/Beats_a_fixed_SMA%3F-Busted-8b949e?style=flat-square)\n\n"
            "A plain moving average has one speed: it lags in trends and whipsaws in chop. "
            "Perry Kaufman's clever fix (1995) — the **Adaptive Moving Average (KAMA)** — "
            "measures how *efficient* the recent price path is (a clean straight line vs a "
            "sawtooth) and speeds up in trends, freezes in chop. The pitch: a buy-when-price-"
            "crosses-above rule on KAMA should beat the same rule on a fixed SMA — fewer "
            "whipsaws, faster entries. We test that *exact* claim.\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the permutation placebo, "
            "the cost and parameter sweeps? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first (real tape)\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does KAMA beat a plain SMA? | **No.** On SPY the KAMA book's net Sharpe is "
            f"**+{R['kama_sh']:.2f}** vs the SMA's **+{R['sma_sh']:.2f}** — the adaptation is "
            f"*worse*, significantly so (HAC *t* = **{R['ks_t']:.2f}**). |\n"
            f"| Fewer whipsaws? | **No — more.** KAMA's turnover is **{R['kama_turn']}** vs the "
            f"SMA's **{R['sma_turn']}**. The 'fewer whipsaws' pitch is backwards here. |\n"
            f"| Beat just holding? | **No.** Buy-and-hold nets **+{R['bh_sh']:.2f}** Sharpe; "
            "both filters lose to doing nothing. |\n"
            "| Could you trade it? | **No.** It loses to both the simpler rule and to holding, "
            "and costs make its higher churn worse. |\n\n"
            "> The Efficiency-Ratio idea is genuinely clever — it just doesn't help a "
            "price-cross rule on real index tapes. Numbers frozen as-of "
            f"{R['asof']}; see [docs/results.md](../docs/results.md)."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A fixed moving average is a compromise: too slow for trends, too fast for "
            "chop. KAMA fixes this — it measures the Efficiency Ratio (how much net distance "
            "the price covered vs how much it wiggled) and adapts its speed. In a clean trend "
            "it tracks like a fast EMA; in noise it freezes like a slow one. So a price-cross "
            "rule on KAMA gives fewer whipsaws and faster entries than the same rule on a "
            "plain SMA.\"*\n\n"
            "Kaufman introduced this in *Smarter Trading* (1995). It is the headline 'adaptive' "
            "indicator on most charting platforms. We test whether the adaptation actually "
            "earns its keep — *measured against the simple thing it claims to beat.*"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If adapting to volatility genuinely beat a fixed window, it would be a free "
            "upgrade: same idea, smarter execution, less getting chopped up. Millions of "
            "traders pick indicators on exactly this 'smarter version' logic. But if the "
            "adaptation *doesn't* help — or actively hurts — then the extra machinery is just "
            "more knobs to overfit, and the plain moving average was the better tool all along."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "We make the race fair:\n\n"
            "1. **Same rule, two filters.** Long when the close is above the filter, flat "
            "otherwise. Run it once with KAMA(10, 2, 30) and once with a fixed **SMA(30)** of "
            "matched length. The *difference* is the adaptation's contribution.\n"
            "2. **One execution lag.** The signal at today's close earns *tomorrow's* return — "
            "one shift, no look-ahead.\n"
            "3. **Charge real costs**, count turnover, and compare everything to **buy-and-"
            "hold** (because any trend filter that sits in cash part-time must be raced "
            "excess-vs-excess).\n\n"
            "**The mirage line:** if KAMA − SMA isn't reliably *positive*, the adaptation adds "
            "nothing. We run it on six liquid tapes (SPY, QQQ, IWM, EFA, GLD, AAPL)."
        ),

        # ---- BEAT 4 — TEARDOWN -----------------------------------------------
        md(
            "## 4 · The teardown — the head-to-head on SPY\n\n"
            "Three equity curves on the same tape: hold, the plain SMA timing rule, and KAMA."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = get('SPY')['close']\n"
            "    k = st.kama(p); s = st.sma(p, 30)\n"
            "    bk = st.book_excess(p, st.ma_positions(p, k), cost_bps=1.0)\n"
            "    bs = st.book_excess(p, st.ma_positions(p, s), cost_bps=1.0)\n"
            "    bh = st.buy_hold_excess(p)\n"
            "    eqk = (1+bk).cumprod(); eqs = (1+bs).cumprod(); eqh = (1+bh).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "    ax.plot(eqh.index, eqh, c=GREY, lw=1.6, label=f'buy & hold (Sharpe +{st.sharpe(bh):.2f})')\n"
            "    ax.plot(eqs.index, eqs, c=GREEN, lw=1.6, label=f'fixed SMA(30) (Sharpe +{st.sharpe(bs):.2f})')\n"
            "    ax.plot(eqk.index, eqk, c=RED, lw=1.6, label=f'KAMA (Sharpe +{st.sharpe(bk):.2f})')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (excess, net, log)')\n"
            "    ax.set_title('KAMA trails the plain SMA — and both trail just holding'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    print('cached tape missing — frozen: KAMA +{R['kama_sh']:.2f}, "
            f"SMA +{R['sma_sh']:.2f}, hold +{R['bh_sh']:.2f} (net Sharpe)')"
        ),
        md(
            f"The plain SMA (Sharpe **+{R['sma_sh']:.2f}**) beats the adaptive KAMA "
            f"(**+{R['kama_sh']:.2f}**) — and **buy-and-hold (+{R['bh_sh']:.2f}**) beats both. "
            "The 'smarter' filter is the worst of the three."
        ),
        md("**Does the 'fewer whipsaws' promise hold? Look at turnover.**"),
        code(
            "labels = ['fixed SMA(30)', 'KAMA (adaptive)']\n"
            "if HAVE_REAL:\n"
            "    turns = [st.ma_positions(p, s).diff().abs().sum(),\n"
            "             st.ma_positions(p, k).diff().abs().sum()]\n"
            "else:\n"
            f"    turns = [{R['sma_turn']}, {R['kama_turn']}]\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            "ax.bar(labels, turns, color=[GREEN, RED], width=.55)\n"
            "ax.set_ylabel('lifetime turnover (one-way × NAV)')\n"
            "ax.set_title('KAMA trades MORE, not less — the opposite of the pitch')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SMA turnover: {turns[0]:.0f}   KAMA turnover: {turns[1]:.0f}')"
        ),
        md(
            f"KAMA's lifetime turnover (**{R['kama_turn']}**) is *higher* than the plain SMA's "
            f"(**{R['sma_turn']}**). The squared smoothing constant keeps the filter creeping "
            "across the price even in chop — so the adaptation generates *extra* crossings, the "
            "exact reverse of the marketing."
        ),
        md("**Is it just SPY? The whole basket.**"),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        r = st.run_experiment(get(t)['close'], cost_bps=1.0)\n"
            "        rows.append((t, r['kama_sharpe'], r['sma_sharpe'], r['diff_ks_t']))\n"
            "else:\n"
            "    lit = {\n" + _panel_lit() + "    }\n"
            "    rows = [(t, v[0], v[1], v[3]) for t, v in lit.items()]\n"
            "tab = pd.DataFrame(rows, columns=['ticker','KAMA Sharpe','SMA Sharpe','KAMA-SMA t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "col = [GREEN if v > 0 else RED for v in tab['KAMA-SMA t']]\n"
            "ax.bar(tab['ticker'], tab['KAMA-SMA t'], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('KAMA - SMA  (HAC t)')\n"
            "ax.set_title('Five of six tapes: KAMA significantly WORSE than the plain SMA')\n"
            "plt.tight_layout(); plt.show()\n"
            "tab.round(3)"
        ),
        md(
            "Only GLD ties; every equity tape is *significantly* behind the plain SMA, and "
            "none clears the +2 bar that 'KAMA beats SMA' would require."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** KAMA − SMA = **{R['ks_bps']:+.2f} bps/day**, HAC "
            f"*t* = **{R['ks_t']:.2f}** (significant, wrong side). Five of six tapes worse.\n"
            f"- **Tradability — Mirage.** Loses to the SMA *and* to buy-and-hold; "
            f"turnover {R['kama_turn']} > {R['sma_turn']}, so costs widen the gap.\n"
            "- **Beats a fixed SMA? — Busted.** The adaptation makes the rule worse and "
            "raises churn — the reverse of the 'fewer whipsaws' claim."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There is nothing to deploy: the adaptive filter loses to the *simpler* filter, "
            "which loses to *holding*. And because KAMA trades more, every basis point of cost "
            "hurts it more than the SMA — the cost sweep below tilts the race further against it."
        ),
        code(
            "costs = [c[0] for c in " + repr(R['costs']) + "]\n"
            "if HAVE_REAL:\n"
            "    kss = [st.run_experiment(get('SPY')['close'], cost_bps=c)['diff_ks_t'] for c in costs]\n"
            "else:\n"
            "    kss = [c[4] for c in " + repr(R['costs']) + "]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(costs, kss, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(-2, ls='--', c=GREY)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('KAMA - SMA (HAC t)')\n"
            "ax.set_title('More costs → KAMA falls further behind the plain SMA')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The KAMA-minus-SMA t-stat only gets more negative as costs rise.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook shows a synthetic tape where "
            "trends and chop alternate sharply: there KAMA *does* beat the SMA (the harness "
            "works). The real market just isn't that regime.\n"
            "- **Other adaptive filters.** [Supertrend](../../106-supertrend/) (ATR-adaptive) "
            "is KAMA's cousin — same family, same honest treatment.\n"
            "- **The simpler benchmark wins.** This echoes [Bollinger](../../104-bollinger-reversion/) "
            "and [CCI](../../178-cci/): elaborate technical machinery rarely beats the plain version.\n\n"
            "*Think KAMA shines on a different asset, timeframe, or rule (e.g. KAMA-slope, not "
            "price-cross)? Fork this, swap it in, and show a KAMA − SMA difference that clears "
            "*t* = +2. That is the bar.*"
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
            "# KAMA — a quantitative teardown\n"
            "### Daily closes · KAMA(10,2,30) vs SMA(30) long/flat · HAC inference · permutation placebo · planted-edge control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a fixed SMA%3F: Busted](https://img.shields.io/badge/Beats_a_fixed_SMA%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the KAMA(10,2,30) price-cross timing book beats the matched SMA(30) book, "
            "excess-of-cash and net of costs, across six liquid daily tapes.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, `auto_adjust=True` "
            f"(total-return-ish), full history as-of {R['asof']}; the offline core and the "
            "positive control run on a deterministic synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front (real tape)\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | KAMA − SMA daily out-performance **{R['ks_bps']:+.2f} "
            f"bps/day**, HAC *t* = **{R['ks_t']:.2f}** (significant, wrong side); permutation "
            f"*p* = **{R['perm_p']:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | KAMA net Sharpe **+{R['kama_sh']:.2f}** < SMA "
            f"**+{R['sma_sh']:.2f}** < buy-hold **+{R['bh_sh']:.2f}**; turnover "
            f"{R['kama_turn']} > {R['sma_turn']}. |\n"
            f"| **Beats a fixed SMA?** | `BUSTED` | Worse net Sharpe *and* higher churn — the "
            "reverse of the 'fewer whipsaws' claim. |\n\n"
            "> In plain words: the Efficiency-Ratio adaptation is a clever idea that, on a "
            "price-cross timing rule over real index tapes, costs you rather than pays you."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $C_t$ be the close. The Efficiency Ratio over window $n$:\n\n"
            "$$\\text{ER}_t = \\frac{|C_t - C_{t-n}|}{\\sum_{i=1}^{n} |C_{t-i+1} - C_{t-i}|}"
            "\\in [0, 1]$$\n\n"
            "The adaptive smoothing constant interpolates between a fast and slow EMA, squared:\n\n"
            "$$\\text{SC}_t = \\left[\\text{ER}_t\\left(\\tfrac{2}{f+1} - \\tfrac{2}{s+1}\\right) "
            "+ \\tfrac{2}{s+1}\\right]^2,\\qquad "
            "\\text{KAMA}_t = \\text{KAMA}_{t-1} + \\text{SC}_t\\,(C_t - \\text{KAMA}_{t-1})$$\n\n"
            "with Kaufman's defaults $n=10,\\,f=2,\\,s=30$. The timing rule: long if "
            "$C_t > \\text{KAMA}_t$, else flat. The claims:\n\n"
            "- **H₁ (adaptation helps).** $\\mathbb{E}[r^{\\text{KAMA}}_t - r^{\\text{SMA}}_t] > 0$ "
            "for the matched fixed SMA($s$).\n"
            "- **H₂ (it's not noise).** That difference survives a position-shuffle permutation null.\n"
            "- **H₃ (tradable).** It survives costs and beats buy-and-hold.\n\n"
            "We reject H₁ (the difference is significantly *negative*), H₂ (permutation "
            "$p \\approx 1$), and H₃ (loses to both SMA and hold)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "KAMA is the flagship 'adaptive' indicator taught as a strict upgrade over the "
            "moving average. If H₁–H₃ held, adapting smoothing to realized efficiency would be "
            "a free improvement to any MA-based system. The finding that the adaptation is "
            "*negatively* contributive — and *raises* turnover — matters: it is a clean example "
            "of added parameters degrading an out-of-sample rule rather than improving it, the "
            "over-fitting failure mode (Sullivan-Timmermann-White 1999)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **KAMA(10, 2, 30)** and matched **SMA(30)**, both as a long/flat price-cross.\n"
            "- **Execution lag.** Position at close $t$ earns return $t{+}1$ — one `shift`.\n"
            "- **Excess-of-cash, net.** rf subtracted on the invested fraction; one-way "
            "`cost_bps` × |Δpos| turnover; shorts (in the L/S variant) pay borrow.\n"
            "- **Arbiter.** HAC (Newey-West) *t* on the daily KAMA − SMA difference vs 0.\n"
            "- **Placebo.** Shuffle KAMA's daily positions, re-book, repeat 2,000× → permutation *p*.\n"
            "- **Sweeps.** Costs {0…5 bp}; ER period {5,10,20}; window {20,30,50}; long/short.\n"
            "- **Positive control.** Synthetic regime-switching tape with a planted edge knob.\n\n"
            "Six tapes: SPY, QQQ, IWM, EFA, GLD, AAPL — full Yahoo history to "
            f"{R['asof']}."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The head-to-head difference, with its HAC *t*\n\n"
            "The decisive number: is KAMA − SMA reliably positive? (It is reliably *negative*.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(get('SPY')['close'], er_period=10, fast=2, slow=30,\n"
            "                          sma_window=30, cost_bps=1.0, n_perm=2000)\n"
            "    ks_bps, ks_t, perm = r['diff_ks_mean_bps'], r['diff_ks_t'], r['perm_p']\n"
            "    kbh_t = r['diff_kbh_t']\n"
            "else:\n"
            f"    ks_bps, ks_t, perm, kbh_t = {R['ks_bps']}, {R['ks_t']}, {R['perm_p']}, {R['kbh_t']}\n"
            "print(f'KAMA - SMA      : {ks_bps:+.3f} bps/day   HAC t = {ks_t:+.2f}')\n"
            "print(f'KAMA - buy&hold : HAC t = {kbh_t:+.2f}')\n"
            "print(f'permutation p (KAMA book carries timing info?): {perm:.4f}')"
        ),
        md(
            f"> In plain words: the adaptation subtracts ~**{abs(R['ks_bps']):.1f} bps every "
            f"day** from the plain SMA, at HAC *t* = **{R['ks_t']:.2f}** — a statistically "
            "reliable *loss*. And the position-shuffle placebo says KAMA's own timing is "
            f"indistinguishable from a random reshuffle (*p* = {R['perm_p']:.3f})."
        ),
        md(
            "### 4b · Net Sharpe race — KAMA < SMA < buy-and-hold\n\n"
            "All excess-of-cash, net of 1 bp."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rr = st.run_experiment(get('SPY')['close'], cost_bps=1.0)\n"
            "    vals = [rr['kama_sharpe'], rr['sma_sharpe'], rr['bh_sharpe']]\n"
            "else:\n"
            f"    vals = [{R['kama_sh']}, {R['sma_sh']}, {R['bh_sh']}]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['KAMA','fixed SMA(30)','buy & hold'], vals, color=[RED, GREEN, GREY], width=.55)\n"
            "ax.set_ylabel('net Sharpe (excess-of-cash)')\n"
            "ax.set_title('The adaptive filter is the worst of the three')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'KAMA {vals[0]:+.3f} | SMA {vals[1]:+.3f} | hold {vals[2]:+.3f}')"
        ),
        md(
            "### 4c · Panel — five of six tapes significantly behind the SMA\n\n"
            "HAC *t* on KAMA − SMA per tape. None clears +2."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        rr = st.run_experiment(get(t)['close'], cost_bps=1.0)\n"
            "        rows.append((t, rr['kama_sharpe'], rr['sma_sharpe'], rr['bh_sharpe'], rr['diff_ks_t']))\n"
            "else:\n"
            "    lit = {\n" + _panel_lit() + "    }\n"
            "    rows = [(t,)+v for t, v in lit.items()]\n"
            "tab = pd.DataFrame(rows, columns=['ticker','KAMA Sh','SMA Sh','hold Sh','KAMA-SMA t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "ax.bar(tab['ticker'], tab['KAMA-SMA t'],\n"
            "       color=[GREEN if v>0 else RED for v in tab['KAMA-SMA t']])\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('KAMA - SMA (HAC t)'); ax.set_ylim(-3.5, 3.5)\n"
            "ax.set_title('No tape clears +2; five clear -2 (wrong side)')\n"
            "plt.tight_layout(); plt.show()\n"
            "tab.round(3)"
        ),
        md(
            "> In plain words: only gold (GLD) ties. On every equity index and on AAPL the "
            "adaptation is a *significant* drag versus the plain SMA."
        ),
        md(
            "### 4d · Cost sweep — costs widen the gap (KAMA churns more)\n\n"
            "If the 'fewer whipsaws' story were true, higher costs would *favour* KAMA. They don't."
        ),
        code(
            "rows = " + repr(R['costs']) + "\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for c in [0.0, 0.5, 1.0, 2.0, 5.0]:\n"
            "        rr = st.run_experiment(get('SPY')['close'], cost_bps=c)\n"
            "        rows.append((c, rr['kama_sharpe'], rr['sma_sharpe'], rr['diff_ks_mean_bps'], rr['diff_ks_t']))\n"
            "cs = [r[0] for r in rows]; tt = [r[4] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(cs, tt, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(-2, ls='--', c=GREY)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('KAMA - SMA (HAC t)')\n"
            "ax.set_title('More costs → KAMA falls further behind')\n"
            "plt.tight_layout(); plt.show()\n"
            "pd.DataFrame(rows, columns=['cost_bps','KAMA Sh','SMA Sh','KAMA-SMA bps','t']).round(3)"
        ),
        md(
            "### 4e · Robustness — the sign never flips, and long/short is a disaster\n\n"
            "ER window and MA length sweeps; plus the long/short framing."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = get('SPY')['close']\n"
            "    grid = []\n"
            "    for slow in (20,30,50):\n"
            "        rr = st.run_experiment(p, slow=slow, sma_window=slow, cost_bps=1.0)\n"
            "        grid.append((f'slow={slow}', rr['kama_sharpe'], rr['sma_sharpe'], rr['diff_ks_t']))\n"
            "    for er in (5,10,20):\n"
            "        rr = st.run_experiment(p, er_period=er, cost_bps=1.0)\n"
            "        grid.append((f'er={er}', rr['kama_sharpe'], rr['sma_sharpe'], rr['diff_ks_t']))\n"
            "    ls = st.run_experiment(p, cost_bps=1.0, long_short=True)\n"
            "    grid.append(('long/short', ls['kama_sharpe'], ls['sma_sharpe'], ls['diff_ks_t']))\n"
            "else:\n"
            "    grid = [('slow=20',0.178,0.415,-2.68),('slow=30',0.203,0.530,-2.98),\n"
            "            ('slow=50',0.200,0.488,-2.42),('er=5',0.149,0.530,-2.83),\n"
            "            ('er=10',0.203,0.530,-2.98),('er=20',0.388,0.530,-1.76),\n"
            f"            ('long/short',{R['ls_kama']},{R['ls_sma']},{R['ls_t']})]\n"
            "pd.DataFrame(grid, columns=['variant','KAMA Sh','SMA Sh','KAMA-SMA t']).round(3)"
        ),
        md(
            f"> In plain words: across every parameter the SMA wins; the longest ER window "
            "narrows the gap but never closes it; and the long/short version is outright "
            f"negative (KAMA {R['ls_kama']:+.2f} Sharpe)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — KAMA − SMA {R['ks_bps']:+.2f} bps/day, HAC "
            f"*t* {R['ks_t']:.2f}; five of six tapes < −2; permutation *p* {R['perm_p']:.3f}.\n"
            f"- **Tradability `MIRAGE`** — KAMA +{R['kama_sh']:.2f} < SMA +{R['sma_sh']:.2f} "
            f"< hold +{R['bh_sh']:.2f}; turnover {R['kama_turn']} > {R['sma_turn']}.\n"
            "- **Beats a fixed SMA? `BUSTED`** — worse Sharpe and more churn.\n\n"
            "> **Survivorship**, named on the Signal axis: the basket is liquid surviving "
            "ETFs / a surviving mega-cap. The bias tilts toward *holding* being good — which is "
            "exactly what beats both filters, so it reinforces (does not manufacture) the verdict."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — nothing survives\n\n"
            "The adaptive filter loses to the simpler filter, which loses to holding, and its "
            "higher turnover means costs punish it more. There is no version, parameter, or "
            "instrument in the grid where KAMA clears the bar for beating the plain SMA."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of crowning KAMA when it deserves it? Plant a regime-"
            "switching tape: strong clean trends (efficiency ≈ 1) alternating with whippy chop "
            "(efficiency ≈ 0), at a realistic 2 bp cost, and sweep the edge knob:"
        ),
        code(
            "rows = " + repr(R['syn']) + "\n"
            "# always re-run the synthetic — it is deterministic & offline\n"
            "rows = []\n"
            "for edge in (0.0, 2.0, 4.0, 8.0):\n"
            "    d, _ = data.synthetic_panel(n_days=6000, edge=edge, seed=433)\n"
            "    rr = st.run_experiment(d['close'], cost_bps=2.0)\n"
            "    rows.append((edge, rr['kama_sharpe'], rr['sma_sharpe'], rr['diff_ks_mean_bps'], rr['diff_ks_t']))\n"
            "es = [r[0] for r in rows]; tt = [r[4] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(es, tt, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted regime-switch edge'); ax.set_ylabel('KAMA - SMA (HAC t)')\n"
            "ax.set_title('Engine works: at a strong planted edge, KAMA finally beats the SMA')\n"
            "plt.tight_layout(); plt.show()\n"
            "pd.DataFrame(rows, columns=['edge','KAMA Sh','SMA Sh','KAMA-SMA bps','t']).round(3)"
        ),
        md(
            "At `edge = 0` the difference is noise (*t* ≈ +0.8); at a strong planted "
            "trend/chop contrast KAMA's whipsaw avoidance beats the SMA at *t* ≈ +2.9. "
            "**The harness is faithful** — it crowns KAMA when the regime genuinely rewards "
            "adaptation. The real-tape verdict is therefore a statement about the **market** "
            "(real index tapes are not that sharply regime-switching at this filter scale), "
            "not the method. Forks worth trying: KAMA-*slope* signals instead of price-cross; "
            "FX or commodity futures where regime switches are sharper; or pairing the ER "
            "directly as a position-sizing overlay rather than a binary cross."
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
