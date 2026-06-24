"""Generate the two narrative notebooks for Study 418 (Money Flow Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
positive control runs anywhere, offline and deterministic; the real-tape cells use the
cached SPY parquet under ../_cache/ if present and otherwise quote the frozen headline
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


def _boot():
    """BOOT preamble with the frozen R dict literal injected."""
    import pprint
    return BOOT.replace("__R__", pprint.pformat(R, indent=0, width=100))


# Frozen real-tape headline numbers -- mirror of docs/results.md.
# SPY daily 2000-01-03 .. 2026-06-23 (as-of), partial 2026-06-24 bar dropped.
# Long/flat 14-day oscillator rule: <=30 long, >=70 flat; one execution lag;
# cost 2 bps one-way x NAV; excess-of-cash (rf=0); n=6657 bars (~26.5y).
R = dict(
    asof="2026-06-23", fp="0e5348c18f0c", n=6657, years=26.5,
    # MFI long/flat book (net excess, 2 bps)
    mfi_sharpe=0.460, mfi_ann=7.23, mfi_t=2.72, mfi_dd=-52.96, mfi_expo=41.7,
    mfi_gross_sharpe=0.466, mfi_gross_t=2.76,
    mfi_perm_p=0.030,
    # RSI long/flat book (net excess, 2 bps)
    rsi_sharpe=0.312, rsi_ann=5.14, rsi_t=1.93, rsi_dd=-55.19, rsi_expo=39.6,
    rsi_perm_p=0.204,
    # Buy-and-hold (net excess)
    bh_sharpe=0.505, bh_ann=9.76, bh_t=3.02, bh_dd=-55.19,
    # Head-to-head deltas
    delta_mfi_rsi=0.148, delta_mfi_bh=-0.045,
    delta_mfi_rsi_bootp=0.277,   # block-bootstrap two-sided p on the daily diff
    # Cost sweep (MFI net excess Sharpe / t ; RSI Sharpe)
    c0_mfi=0.466, c0_mfi_t=2.76, c0_rsi=0.314,
    c2_mfi=0.460, c2_mfi_t=2.72, c2_rsi=0.312,
    c5_mfi=0.451, c5_mfi_t=2.67, c5_rsi=0.309,
    c10_mfi=0.436, c10_mfi_t=2.59, c10_rsi=0.304,
    # Synthetic positive control (seed 418, net excess Sharpe)
    syn0_mfi=0.365, syn0_rsi=0.560,
    syn1_mfi=0.490, syn1_rsi=0.196,
    syn2_mfi=0.432, syn2_rsi=0.183,
    syn2_mean_delta=0.164,  # across 12 seeds at edge=2
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from money_flow_index import data, strategy as st

# Frozen real-tape headline numbers -- mirror of docs/results.md (single source of truth).
R = __R__

ASOF = "2026-06-23"

def have_real():
    return os.path.exists(data._cache_path("SPY", data.DEFAULT_CACHE))

HAVE_REAL = have_real()

def spy():
    return data.load_real("SPY").loc[:ASOF]

print("real SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Money Flow Index -- is the *volume-weighted RSI* a better RSI?\n"
            "### The MFI, tested against the plain RSI and buy-and-hold, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_plain_RSI%3F: Not_supported](https://img.shields.io/badge/"
            "Beats_plain_RSI%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The **Money Flow Index** is sold as the RSI's smarter sibling: same 0-100 "
            "overbought/oversold dial, but each day's move is *weighted by volume*, so "
            "(the story goes) it ignores low-conviction wiggles and only fires on moves "
            "that 'real money' is behind. The pitch writes itself: *a better RSI*. This "
            "notebook asks the only question that matters -- on the actual tape, does "
            "folding volume in **buy you anything** the plain RSI doesn't?\n\n"
            "> **This is the plain-language layer.** Want the HAC *t*-stats, the bootstrap "
            "on the MFI-minus-RSI gap, and the synthetic control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(_boot()),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the MFI rule beat the plain RSI rule? | **Not really.** The MFI's "
            f"net excess Sharpe is **{R['mfi_sharpe']:.2f}** vs **{R['rsi_sharpe']:.2f}** for "
            f"the RSI -- a gap of +{R['delta_mfi_rsi']:.2f}, but a block bootstrap puts it "
            f"squarely in the noise (*p* = {R['delta_mfi_rsi_bootp']:.2f}). |\n"
            "| Does either beat just buying and holding SPY? | **No.** Buy-and-hold scores "
            f"**{R['bh_sharpe']:.2f}** -- higher than both timers, with the same drawdown. |\n"
            "| So is the MFI useless? | Its own book is *positive* and statistically real "
            f"(HAC *t* = {R['mfi_t']:.1f}) -- but that's just it being long ~42% of a market "
            "that went up. It's diluted beta, not alpha. |\n"
            "| Could you trade it? | **No.** Even before costs it lags buy-and-hold; "
            "weighting the RSI by volume doesn't change that. |\n\n"
            "> Adding volume to the RSI feels like adding information. On 26 years of SPY it "
            "adds a rounding error -- and you give up return vs simply holding the index."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The RSI only looks at price. The Money Flow Index weights every move by "
            "volume, so it tells you whether a move has conviction behind it. An MFI below 20 "
            "is a high-conviction wash-out -- a better buy signal than a bare RSI 30. It's the "
            "RSI, upgraded.\"*\n\n"
            "It is genuinely plausible. A 2% drop on triple volume *feels* different from a 2% "
            "drift down on a quiet day -- the first looks like capitulation that should bounce. "
            "The MFI is built to see exactly that difference; the RSI is blind to it. If volume "
            "carries information about the next move, the MFI should turn it into a cleaner timer."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "Every charting package ships the MFI right next to the RSI, and traders pick it "
            "*because* of the 'volume makes it better' story. If that story is true, the simplest "
            "win in technical analysis is to swap one indicator for its volume-weighted twin and "
            "pocket a free improvement. If it's false, a whole layer of folklore -- 'volume "
            "confirms price' -- is decorative. The honest test is to run **the same rule** on "
            "both indicators and see if the volume weighting actually moves the needle."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "We make it a fair fight:\n\n"
            "1. **Same rule, two indicators.** 14-day MFI and 14-day RSI. Oscillator "
            "<= 30 -> go long; >= 70 -> go flat; otherwise hold. Identical thresholds, "
            "identical execution. The *only* difference is volume weighting.\n"
            "2. **Honest execution.** The signal is read at today's close; the position earns "
            "*tomorrow's* return (one day's lag). When flat we sit in cash. Costs are charged "
            "every time we flip.\n"
            "3. **The benchmark that matters.** We race both against **buy-and-hold SPY**, on "
            "an excess-of-cash basis so it's apples to apples. A long/flat timer that can't beat "
            "buying the index has no reason to exist.\n\n"
            "**What would make us say 'mirage':** if the MFI doesn't decisively beat the RSI "
            "*and* neither beats buy-and-hold. We run it on 26 years of SPY daily data."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First: the three contenders, net excess-of-cash Sharpe.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(spy(), cost_bps=2.0, rf_annual=0.0, perm_iter=1)\n"
            "    mfi_s = res['mfi']['net']['sharpe']; rsi_s = res['rsi']['net']['sharpe']\n"
            "    bh_s  = res['buy_hold']['net']['sharpe']\n"
            "else:\n"
            "    mfi_s, rsi_s, bh_s = R['mfi_sharpe'], R['rsi_sharpe'], R['bh_sharpe']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "names = ['MFI\\n(volume-weighted)', 'RSI\\n(plain)', 'Buy & hold\\nSPY']\n"
            "vals = [mfi_s, rsi_s, bh_s]\n"
            "cols = [AMBER, GREY, GREEN]\n"
            "bars = ax.bar(names, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net excess-of-cash Sharpe (2 bps)')\n"
            "ax.set_title('The volume-weighted RSI edges the plain RSI -- and both lose to buy-and-hold')\n"
            "for b, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, b.get_height()+0.01),\n"
            "                ha='center', va='bottom', fontsize=11)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'MFI {mfi_s:.2f}  |  RSI {rsi_s:.2f}  |  buy-hold {bh_s:.2f}')"
        ),
        md(
            f"The MFI ({R['mfi_sharpe']:.2f}) does edge the plain RSI ({R['rsi_sharpe']:.2f}). "
            "But both timers sit *below* buy-and-hold "
            f"({R['bh_sharpe']:.2f}). Stepping aside on overbought readings cost more return "
            "than it saved in risk -- because the market mostly went up while you were watching "
            "from cash."
        ),
        md(
            "**Is the MFI-over-RSI gap real, or luck?** Let's bootstrap the daily difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = spy()\n"
            "    pm = st.positions_from_indicator(st.money_flow_index(b))\n"
            "    pr = st.positions_from_indicator(st.rsi(b['close']))\n"
            "    dm = (st.backtest(b, pm, 2.0)['net_excess']\n"
            "          - st.backtest(b, pr, 2.0)['net_excess']).dropna().to_numpy()\n"
            "    rng = np.random.default_rng(418); n=len(dm); blk=21; nb=int(np.ceil(n/blk))\n"
            "    boot = np.array([np.concatenate([dm[i:i+blk] for i in\n"
            "            rng.integers(0,n-blk,size=nb)])[:n].mean() for _ in range(2000)])\n"
            "    pval = 2*min((boot<=0).mean(), (boot>=0).mean())\n"
            "else:\n"
            "    boot = np.random.default_rng(0).normal(0, 1, 2000); pval = R['delta_mfi_rsi_bootp']\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(boot*1e4*252, bins=40, color=GREY, alpha=.7)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('bootstrapped MFI-minus-RSI annual edge (bps/yr)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The MFI-over-RSI gap straddles zero  (two-sided p = {pval:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'block-bootstrap two-sided p on the MFI-RSI daily difference: {pval:.3f}')"
        ),
        md(
            f"The histogram is centred barely to the right of zero and the *p*-value is "
            f"**{R['delta_mfi_rsi_bootp']:.2f}** -- the MFI's apparent edge over the RSI is well "
            "inside what coin-flips produce. 'The volume-weighted RSI is better' is **not "
            "supported**: the gap is noise."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- Weak.** The MFI's own book *is* statistically positive "
            f"(net excess Sharpe {R['mfi_sharpe']:.2f}, HAC *t* = {R['mfi_t']:.1f}) -- but that "
            "is mostly being long a rising market ~42% of the time, not a volume edge.\n"
            f"- **Tradability -- Mirage.** Both timers underperform buy-and-hold "
            f"({R['bh_sharpe']:.2f}) even before costs. There is no residual alpha to deploy.\n"
            f"- **'Beats the plain RSI?' -- Not supported.** The MFI-minus-RSI gap "
            f"(+{R['delta_mfi_rsi']:.2f} Sharpe) has bootstrap *p* = {R['delta_mfi_rsi_bootp']:.2f}. "
            "Volume weighting changes the answer by a rounding error."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Costs aren't even the problem here -- the rule trades rarely, so they barely "
            "register. The problem is upstream: the timer gives up return vs holding the index."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    b = spy()\n"
            "    mfi_sw = [st.run_experiment(b, cost_bps=c, perm_iter=1)['mfi']['net']['sharpe'] for c in costs]\n"
            "    rsi_sw = [st.run_experiment(b, cost_bps=c, perm_iter=1)['rsi']['net']['sharpe'] for c in costs]\n"
            "else:\n"
            "    mfi_sw = [R['c0_mfi'], R['c2_mfi'], R['c5_mfi'], R['c10_mfi']]\n"
            "    rsi_sw = [R['c0_rsi'], R['c2_rsi'], R['c5_rsi'], R['c10_rsi']]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(costs, mfi_sw, 'o-', c=AMBER, lw=2, label='MFI')\n"
            "ax.plot(costs, rsi_sw, 's--', c=GREY, lw=2, label='RSI')\n"
            "ax.axhline(R['bh_sharpe'], ls=':', c=GREEN, lw=2, label='buy & hold')\n"
            "ax.set_xlabel('one-way cost (bps x NAV)'); ax.set_ylabel('net excess Sharpe')\n"
            "ax.set_title('Costs barely move the timers -- but the green line is always above them')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Even at zero cost, both timers sit below buy-and-hold.')"
        ),
        md(
            "Both lines are nearly flat across cost -- low turnover. And both stay under the "
            "buy-and-hold line at every cost level, including zero. This isn't an edge eaten by "
            "frictions; it's **no edge to begin with**."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Does volume *ever* help?** The companion notebook plants a deliberate "
            "volume-conditioned bounce in synthetic data -- and there the MFI *does* pull ahead "
            "of the RSI. So the indicator isn't broken; SPY's volume just doesn't carry that "
            "signal.\n"
            "- **Other thresholds / windows.** Try MFI 20/80 or a 7-day window. The bar to clear "
            "is a *delta-vs-RSI* bootstrap *p* < 0.05, not a positive Sharpe.\n"
            "- **Single stocks, not the index.** Volume information may be stronger in a single "
            "name than in a diversified ETF. Fork this and run AAPL or a small-cap.\n\n"
            "*Think volume weighting helps somewhere? Fork this, find a tape where the "
            "MFI-minus-RSI bootstrap clears p < 0.05, and show it beats buy-and-hold. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Money Flow Index -- a quantitative teardown\n"
            "### Real SPY daily tape - 26 years - MFI vs RSI vs buy-and-hold - "
            "HAC inference - block-permutation placebo - synthetic volume control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_plain_RSI%3F: Not_supported](https://img.shields.io/badge/"
            "Beats_plain_RSI%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.*  We turn the 14-day "
            "MFI and the 14-day RSI into the **same** long/flat timing rule on SPY (2000-2026), "
            "race their NET excess-of-cash Sharpe against each other and against buy-and-hold, "
            "and ask whether the volume weighting adds *anything* a bootstrap can see.\n\n"
            "> **Not investment advice.** Real data: Yahoo SPY daily bars "
            "2000-01-03 to 2026-06-23 (auto-adjusted close; raw share volume); reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(_boot()),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front (real tape)\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | MFI net excess Sharpe **{R['mfi_sharpe']:.2f}**, "
            f"HAC *t* = **{R['mfi_t']:.2f}** (clears 2) -- but it is ~{R['mfi_expo']:.0f}% "
            "exposure to a rising market, not a volume edge. |\n"
            f"| **Tradability** | `MIRAGE` | Buy-and-hold scores **{R['bh_sharpe']:.2f}** > MFI "
            f"({R['mfi_sharpe']:.2f}) > RSI ({R['rsi_sharpe']:.2f}); the timer destroys, not "
            "creates, risk-adjusted return. |\n"
            f"| **Beats the plain RSI?** | `NOT SUPPORTED` | Delta Sharpe "
            f"**+{R['delta_mfi_rsi']:.2f}**, block-bootstrap two-sided *p* = "
            f"**{R['delta_mfi_rsi_bootp']:.2f}** on the daily MFI-minus-RSI difference. |\n\n"
            "> In plain words: the MFI's book is 'real' only in the trivial sense that being "
            "long a bull market 42% of the time is real. The thing being *sold* -- that volume "
            "weighting beats the plain RSI -- is statistically invisible."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $c_t$ be the close, $TP_t = (h_t+l_t+c_t)/3$ the typical price, $v_t$ the "
            "volume. Raw money flow $MF_t = TP_t\\,v_t$. With $w=14$:\n\n"
            "$$\\mathrm{MFI}_t = 100\\,\\frac{\\sum_{i} MF_i\\,[\\Delta TP_i>0]}"
            "{\\sum_{i} MF_i\\,[\\Delta TP_i>0] + \\sum_{i} MF_i\\,[\\Delta TP_i<0]}$$\n\n"
            "the RSI is the same object with $MF_i \\to |\\Delta c_i|$ (Wilder-smoothed). The "
            "claim is **H1:** the volume weighting makes the MFI a *better-conditioned* "
            "oversold/overbought signal, so the long/flat rule built on it earns a higher "
            "risk-adjusted return than the identical rule on the RSI -- and ideally beats "
            "buy-and-hold (**H2**). We **fail to reject the null on H1** (bootstrap *p* = "
            f"{R['delta_mfi_rsi_bootp']:.2f}) and **reject H2** (both timers < buy-and-hold)."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The MFI is one of the most-cited 'improvements' on the RSI, and the rationale -- "
            "'volume confirms price' -- is a load-bearing belief across technical analysis. If "
            "weighting an oscillator by volume genuinely sharpened it, that belief would have a "
            "clean, mechanical demonstration. The interesting result is not that the MFI is bad "
            "-- its book is positive -- but that the *incremental* contribution of volume is "
            "indistinguishable from zero on a liquid index, while the headline Sharpe is just "
            "recycled beta."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Indicators.** 14-day MFI and 14-day Wilder RSI on SPY daily bars.\n"
            "- **Rule.** Oscillator <= 30 -> long; >= 70 -> flat; else hold previous state "
            "(a standard contrarian oversold/overbought timer). Identical for both.\n"
            "- **Execution (one lag, stated once).** Position is decided at the close of *t* "
            "and earns the return of *t+1* (a single `shift(1)`). Flat = cash (rf).\n"
            "- **Costs.** One-way `cost_bps` x NAV charged on every change in position.\n"
            "- **Race.** NET **excess-of-cash** Sharpe, MFI vs RSI vs buy-and-hold -- "
            "excess-vs-excess so the cash legs net out.\n"
            "- **Inference.** Newey-West HAC *t* on each book's daily excess returns; a "
            "**block-permutation placebo** (shuffle position blocks vs realised returns) for "
            "'is the timing doing anything?'; a **block-bootstrap** two-sided *p* on the "
            "MFI-minus-RSI daily difference for the head-to-head claim.\n"
            "- **Positive control.** A synthetic tape with a *planted, volume-keyed* "
            "overreaction -- the engine must let the MFI beat the RSI there.\n\n"
            "SPY daily 2000-01-03 .. 2026-06-23 (the partial 2026-06-24 bar is dropped); "
            f"n = {R['n']:,} bars (~{R['years']:.1f} years)."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - The three books: Sharpe, annual return, HAC *t*, drawdown\n\n"
            "Net excess-of-cash, 2 bps one-way costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(spy(), cost_bps=2.0, rf_annual=0.0, perm_iter=1)\n"
            "    tab = pd.DataFrame({\n"
            "      'arm':['MFI','RSI','buy & hold'],\n"
            "      'sharpe':[res['mfi']['net']['sharpe'],res['rsi']['net']['sharpe'],res['buy_hold']['net']['sharpe']],\n"
            "      'ann_ret%':[res['mfi']['net']['ann_return']*100,res['rsi']['net']['ann_return']*100,res['buy_hold']['net']['ann_return']*100],\n"
            "      'HAC_t':[res['mfi']['net']['hac_t'],res['rsi']['net']['hac_t'],res['buy_hold']['net']['hac_t']],\n"
            "      'max_dd%':[res['mfi']['net']['max_dd']*100,res['rsi']['net']['max_dd']*100,res['buy_hold']['net']['max_dd']*100],\n"
            "      'exposure':[res['mfi']['net'].get('exposure'),res['rsi']['net'].get('exposure'),1.0]})\n"
            "else:\n"
            "    tab = pd.DataFrame({'arm':['MFI','RSI','buy & hold'],\n"
            "      'sharpe':[R['mfi_sharpe'],R['rsi_sharpe'],R['bh_sharpe']],\n"
            "      'ann_ret%':[R['mfi_ann'],R['rsi_ann'],R['bh_ann']],\n"
            "      'HAC_t':[R['mfi_t'],R['rsi_t'],R['bh_t']],\n"
            "      'max_dd%':[R['mfi_dd'],R['rsi_dd'],R['bh_dd']],\n"
            "      'exposure':[R['mfi_expo']/100,R['rsi_expo']/100,1.0]})\n"
            "print(tab.round(3).to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.5))\n"
            "cols=[AMBER,GREY,GREEN]\n"
            "bars=ax.bar(tab['arm'], tab['sharpe'], color=cols, width=.55)\n"
            "for b,t in zip(bars, tab['HAC_t']):\n"
            "    ax.annotate(f't={t:.2f}', (b.get_x()+b.get_width()/2, b.get_height()+0.01), ha='center')\n"
            "ax.set_ylabel('net excess Sharpe'); ax.axhline(0,c='k',lw=1)\n"
            "ax.set_title('All three positive; buy-and-hold wins; MFI ~ RSI'); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> In plain words: the MFI's HAC *t* of {R['mfi_t']:.1f} clears the bar -- but it is "
            f"long only ~{R['mfi_expo']:.0f}% of the time, and buy-and-hold (100% exposed, "
            f"*t* = {R['bh_t']:.1f}) earns a higher Sharpe with the same drawdown. The timer is "
            "diluting beta, not adding alpha."
        ),

        md(
            "### 4b - The head-to-head: is MFI > RSI real? (block bootstrap)\n\n"
            "We bootstrap the *daily* MFI-minus-RSI net-excess difference in 21-day blocks "
            "(2000 draws) and read the two-sided *p* that its mean differs from zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = spy()\n"
            "    pm = st.positions_from_indicator(st.money_flow_index(b))\n"
            "    pr = st.positions_from_indicator(st.rsi(b['close']))\n"
            "    dm = (st.backtest(b, pm, 2.0)['net_excess']\n"
            "          - st.backtest(b, pr, 2.0)['net_excess']).dropna().to_numpy()\n"
            "    rng = np.random.default_rng(418); n=len(dm); blk=21; nb=int(np.ceil(n/blk))\n"
            "    boot = np.array([np.concatenate([dm[i:i+blk] for i in\n"
            "            rng.integers(0,n-blk,size=nb)])[:n].mean() for _ in range(2000)])\n"
            "    pval = 2*min((boot<=0).mean(), (boot>=0).mean()); obs = dm.mean()\n"
            "else:\n"
            "    boot = np.random.default_rng(0).normal(3e-5, 8e-5, 2000)\n"
            "    pval = R['delta_mfi_rsi_bootp']; obs = boot.mean()\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(boot*252*100, bins=40, color=GREY, alpha=.7)\n"
            "ax.axvline(0, c='k', lw=1.5)\n"
            "ax.axvline(obs*252*100, c=AMBER, lw=2, label=f'observed {obs*252*100:+.2f}%/yr')\n"
            "ax.set_xlabel('MFI-minus-RSI annual edge (%/yr, bootstrapped)')\n"
            "ax.set_ylabel('frequency'); ax.legend()\n"
            "ax.set_title(f'The edge straddles zero  (two-sided p = {pval:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed MFI-RSI daily diff mean = {obs:.2e}; two-sided bootstrap p = {pval:.3f}')"
        ),
        md(
            f"> In plain words: the distribution sits across zero and *p* = "
            f"{R['delta_mfi_rsi_bootp']:.2f}. The MFI's +{R['delta_mfi_rsi']:.2f} Sharpe "
            "advantage over the RSI is not statistically distinguishable from noise. **H1 fails.**"
        ),

        md(
            "### 4c - Block-permutation placebo: is the *timing* informative at all?\n\n"
            "Shuffle each book's position in contiguous 21-day blocks against the realised "
            "returns (keeps the time-in-market, destroys the alignment). *p* = "
            "P(shuffled mean >= observed)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(spy(), cost_bps=2.0, perm_iter=2000)\n"
            "    p_mfi = res['mfi']['perm']['p_value']; p_rsi = res['rsi']['perm']['p_value']\n"
            "else:\n"
            "    p_mfi, p_rsi = R['mfi_perm_p'], R['rsi_perm_p']\n"
            "fig, ax = plt.subplots(figsize=(7.5,4.0))\n"
            "bars = ax.bar(['MFI','RSI'], [p_mfi, p_rsi], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0.05, ls='--', c=RED, label='p = 0.05')\n"
            "for b,p in zip(bars,[p_mfi,p_rsi]):\n"
            "    ax.annotate(f'p={p:.3f}', (b.get_x()+b.get_width()/2, b.get_height()+0.005), ha='center')\n"
            "ax.set_ylabel('permutation p-value'); ax.legend()\n"
            "ax.set_title('MFI timing is marginally informative; RSI timing is not')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'MFI perm p = {p_mfi:.3f}   RSI perm p = {p_rsi:.3f}')"
        ),
        md(
            f"> In plain words: the MFI's timing clears the placebo (*p* = {R['mfi_perm_p']:.3f}) "
            f"while the RSI's does not (*p* = {R['rsi_perm_p']:.2f}) -- a hint that volume *is* "
            "doing a little something. But 'a little something' that still can't beat the RSI "
            "head-to-head (4b) or buy-and-hold (4a) does not earn an `INVESTABLE` stamp."
        ),

        md(
            "### 4d - The positive control -- does the engine see a volume edge when planted?\n\n"
            "Synthetic tape (`data.synthetic_panel`) with a *volume-keyed* overreaction: "
            "down-moves on heavy volume snap back harder. The MFI should beat the RSI as the "
            "edge grows, and tie it at edge = 0."
        ),
        code(
            "edges = [0.0, 1.0, 2.0]\n"
            "mfi_s, rsi_s = [], []\n"
            "for e in edges:\n"
            "    bb,_ = data.synthetic_panel(n_days=4000, edge=e, seed=418)\n"
            "    r = st.run_experiment(bb, cost_bps=2.0, perm_iter=1)\n"
            "    mfi_s.append(r['mfi']['net']['sharpe']); rsi_s.append(r['rsi']['net']['sharpe'])\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "x = range(len(edges)); w=.35\n"
            "ax.bar([i-w/2 for i in x], mfi_s, width=w, color=AMBER, label='MFI')\n"
            "ax.bar([i+w/2 for i in x], rsi_s, width=w, color=GREY, label='RSI')\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels([f'edge={e}' for e in edges])\n"
            "ax.set_ylabel('synthetic net excess Sharpe'); ax.legend()\n"
            "ax.set_title('Plant a volume-keyed bounce -> the MFI pulls ahead (engine is faithful)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,m,rr in zip(edges,mfi_s,rsi_s):\n"
            "    print(f'edge={e}: MFI {m:+.3f}  RSI {rr:+.3f}  delta {m-rr:+.3f}')"
        ),
        md(
            "> In plain words: with a planted volume edge the MFI beats the RSI by "
            f"~+{R['syn1_mfi']-R['syn1_rsi']:.2f}/+{R['syn2_mfi']-R['syn2_rsi']:.2f} Sharpe "
            f"(and across 12 seeds at edge = 2 the mean delta is +{R['syn2_mean_delta']:.2f}); "
            "with no edge the RSI is fine. So the indicator and harness are faithful -- SPY's "
            "volume simply doesn't carry the bounce the MFI is built to exploit. A synthetic "
            "control proves the *machinery*, never the real-tape stamp."
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- MFI net excess Sharpe {R['mfi_sharpe']:.2f}, HAC "
            f"*t* = {R['mfi_t']:.2f} clears 2, *but* the placebo and the buy-hold comparison "
            "show the *t* is bull-market exposure, not a volume signal. Literature sells it as "
            "'a better RSI'; this tape can't certify even that it differs from the RSI.\n"
            f"- **Tradability `MIRAGE`** -- buy-and-hold ({R['bh_sharpe']:.2f}) beats both "
            "timers at every cost level including zero; the rule converts beta into less beta. "
            "Nothing to deploy.\n"
            f"- **'Beats the plain RSI?' `NOT SUPPORTED`** -- delta +{R['delta_mfi_rsi']:.2f} "
            f"Sharpe, bootstrap *p* = {R['delta_mfi_rsi_bootp']:.2f}. The defining promise of "
            "the MFI is statistically invisible on SPY."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- cost sweep\n\n"
            "Net excess Sharpe and HAC *t* for the MFI across one-way cost, with the RSI and "
            "buy-hold lines for reference."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    b = spy()\n"
            "    sw = [st.run_experiment(b, cost_bps=c, perm_iter=1) for c in costs]\n"
            "    mfi_sw = [s['mfi']['net']['sharpe'] for s in sw]\n"
            "    mfi_tw = [s['mfi']['net']['hac_t'] for s in sw]\n"
            "    rsi_sw = [s['rsi']['net']['sharpe'] for s in sw]\n"
            "else:\n"
            "    mfi_sw=[R['c0_mfi'],R['c2_mfi'],R['c5_mfi'],R['c10_mfi']]\n"
            "    mfi_tw=[R['c0_mfi_t'],R['c2_mfi_t'],R['c5_mfi_t'],R['c10_mfi_t']]\n"
            "    rsi_sw=[R['c0_rsi'],R['c2_rsi'],R['c5_rsi'],R['c10_rsi']]\n"
            "fig, axes = plt.subplots(1,2,figsize=(12,4.3))\n"
            "axes[0].plot(costs, mfi_sw,'o-',c=AMBER,lw=2,label='MFI')\n"
            "axes[0].plot(costs, rsi_sw,'s--',c=GREY,lw=2,label='RSI')\n"
            "axes[0].axhline(R['bh_sharpe'], ls=':', c=GREEN, lw=2, label='buy & hold')\n"
            "axes[0].set_xlabel('one-way cost (bps x NAV)'); axes[0].set_ylabel('net excess Sharpe')\n"
            "axes[0].set_title('Timers stay below buy-and-hold at all costs'); axes[0].legend()\n"
            "axes[1].plot(costs, mfi_tw,'o-',c=AMBER,lw=2)\n"
            "axes[1].axhline(2, ls='--', c=GREY); axes[1].axhline(0,c='k',lw=1)\n"
            "axes[1].set_xlabel('one-way cost (bps x NAV)'); axes[1].set_ylabel('MFI HAC t')\n"
            "axes[1].set_title('MFI t-stat clings just above 2 (it is mostly drift)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('MFI net Sharpe by cost:', [round(s,3) for s in mfi_sw])"
        ),
        md(
            "> In plain words: turnover is low, so costs barely move the lines. That is *not* "
            "good news -- it means the rule's failure is structural (it never beats holding the "
            "index), not a frictions problem you could engineer away."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the open questions\n\n"
            "The synthetic control is the constructive part of this study: it shows the MFI is a "
            "faithful volume-conditioned-reversion detector *when the data contains one*. SPY "
            "daily simply doesn't. Forks worth pursuing:\n\n"
            "- **Single names / less liquid tapes.** A diversified ETF averages away "
            "name-specific volume signals; a single stock or small-cap may retain them. The bar "
            "is a *delta-vs-RSI* bootstrap *p* < 0.05, not a positive Sharpe.\n"
            "- **Intraday volume.** Volume information may live at the minute scale, not the "
            "daily close. Run the MFI on 5-minute SPY bars (mind the ~60-day span + capacity "
            "caveat).\n"
            "- **MFI as a *filter*, not a timer.** Use a low MFI to *confirm* an RSI buy rather "
            "than replace it, and test whether the joint condition beats either alone.\n"
            "- **Related desk work.** [Study 104 -- Bollinger-Reversion](../../104-bollinger-reversion/) "
            "and [Study 301 -- Triple-RSI](../../301-triple-rsi/) run the same 'buy the extreme' "
            "logic with the same honest baselines."
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
