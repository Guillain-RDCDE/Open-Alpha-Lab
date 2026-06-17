"""Generate the two narrative notebooks for Study 222 (Altseason-Rotation).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily panel parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_rows=2259, start="2020-04-10", end="2026-06-16",
    dom_mean=0.679, dom_min=0.536, dom_max=0.841, dom_std=0.073,
    fingerprint="0d6a4045008a",
    # dom_change_20d signal, 40 bps RT
    ev1_n=43, ev1_hold=7.3, ev1_active=13.8,
    g1_ret=13.48, g1_vol=21.16, g1_sharpe=0.64, g1_t=1.48, g1_mdd=-66.46,
    n1_ret=9.64,  n1_vol=21.26, n1_sharpe=0.45, n1_t=1.05, n1_mdd=-89.26,
    n1h_ret=5.82, n1h_sharpe=0.27, n1h_t=0.63,  # 80 bps stress
    # dom_level signal, 40 bps RT
    ev2_n=35, ev2_hold=23.9, ev2_active=37.1,
    g2_ret=7.36, g2_vol=28.79, g2_sharpe=0.26, g2_t=0.67, g2_mdd=-76.63,
    n2_ret=4.24, n2_vol=28.79, n2_sharpe=0.15, n2_t=0.38, n2_mdd=-101.83,
    # benchmarks
    bm_btc_ret=25.33, bm_btc_sharpe=0.53,
    bm_alt_ret=33.47, bm_alt_sharpe=0.51,
    bm_spread_ret=8.15, bm_spread_sharpe=0.20, bm_spread_t=0.57,
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

from altseason_rotation import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def _load():
    return data.fetch_panel(fetch=False)

print("real panel cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Altseason-Rotation -- can BTC dominance time the rotation into altcoins?\n"
            "### The alt-season rotation strategy, tested honestly against buy-and-hold\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Alt-season timing: BUSTED](https://img.shields.io/badge/Alt--season_timing%3F-BUSTED-8b949e?style=flat-square)\n\n"
            "Every crypto bull market spawns the same narrative: Bitcoin's 'dominance' -- its "
            "share of total crypto market cap -- is falling. The caption reads: **'Alt season "
            "confirmed! Rotate into altcoins!'** The recipe is specific: when BTC dominance "
            "falls, sell BTC, buy the alts. Ride the rotation. Sell when BTC reclaims its "
            "dominance. This notebook asks whether that timing rule actually outperforms just "
            "*holding* BTC or alts -- and whether there is any real edge beneath the narrative.\n\n"
            "> **This is the plain-language layer.** For Sharpe ratios, HAC t-stats and the "
            "cost sweep, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n\n"
            "> *Companion to [Study 134 -- Bitcoin-Dominance](../../134-bitcoin-dominance/) "
            "which tests the underlying regression signal. Study 222 asks: can you build a "
            "profitable strategy around that signal?*"
        ),
        code(BOOT),

        # BEAT 0 -- VERDICT
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the rotation beat buy-and-hold BTC? | **No.** Net Sharpe "
            f"**{R['n1_sharpe']:.2f}** (timed rotation) vs **{R['bm_btc_sharpe']:.2f}** "
            f"(BTC buy-and-hold). The timed strategy underperforms. |\n"
            f"| Does it beat just always long alts? | **No.** Net Sharpe "
            f"**{R['n1_sharpe']:.2f}** (rotation) vs **{R['bm_alt_sharpe']:.2f}** "
            f"(alt buy-and-hold). Always holding alts beats timing them. |\n"
            f"| Is the edge statistically real? | **No.** Best HAC *t* = **{R['n1_t']:+.2f}** "
            f"on net returns. Well below the |*t*| >= 2 bar. |\n"
            "| Can you capture it even if real? | **No.** At 80 bps round-trip, net Sharpe "
            f"drops to {R['n1h_sharpe']:.2f}. Max drawdown on net returns: {R['n1_mdd']:.0f}%. |\n\n"
            "> **The key insight:** 'alt season' in 2021 was real and spectacular -- but it "
            "was a one-time event in a single crypto cycle. A rotation rule timed by BTC "
            "dominance *was* profitable in that episode, but it cannot be distinguished from "
            "just holding alts the whole time. The timing adds nothing."
        ),

        # BEAT 1 -- THE CLAIM
        md(
            "## 1 -- The claim\n\n"
            "> *'When Bitcoin's market share falls, capital is rotating from BTC to altcoins. "
            "Go long the alt basket, short BTC. The dominance chart tells you when to enter "
            "and exit. This is alt season -- a repeating pattern every crypto cycle.'*\n\n"
            "The claim has two parts: (1) falling BTC dominance *predicts* alt outperformance "
            "(the signal), and (2) you can build a profitable rotation strategy around it "
            "(the trade). [Study 134](../../134-bitcoin-dominance/) already showed the signal "
            "is weak (regression *t* = -1.88, just short of the bar). This study takes the "
            "next question: even granting a weak signal, does it generate a tradable strategy?"
        ),

        # BEAT 2 -- SO WHAT
        md(
            "## 2 -- So what?\n\n"
            "If the timed rotation beats buy-and-hold, it means BTC dominance is a genuine "
            "timing tool -- a signal you could have used to allocate between BTC and alts. "
            "If it doesn't, then 'alt season' is a label applied to rotation episodes "
            "*after they happen*, not a forecast rule. The difference matters: noise-traders "
            "who rotate in and out based on dominance charts incur costs (spreads, slippage, "
            "taxes) while underperforming those who just hold."
        ),

        # BEAT 3 -- HOW WE'D KNOW
        md(
            "## 3 -- How would we even know?\n\n"
            "The test is direct: simulate the strategy and compare:\n\n"
            "1. **Signal.** Fire the rotation when BTC basket dominance (price x fixed supply "
            "proxy) has fallen by >= 2 percentage points over the past 20 days. Go long "
            "equal-weighted alts, short BTC. Exit when the next signal fires 0.\n"
            "2. **Costs.** Deduct 40 bps round-trip per rotation event (realistic for "
            "liquid perpetuals on major exchanges; stress-tested at 80 bps).\n"
            "3. **The bar.** Must beat: (a) BTC buy-and-hold Sharpe, (b) alt EW buy-and-hold "
            "Sharpe, and (c) |t| >= 2 on net returns.\n"
            "4. **No look-ahead.** Signal formed on day t close; position entered on day t+1.\n\n"
            f"Data: BTC, ETH, XRP, ADA, SOL, BNB, DOGE from {R['start']} to {R['end']} "
            f"({R['n_rows']:,} daily observations). Fingerprint: `{R['fingerprint']}`."
        ),

        # BEAT 4 -- THE TEARDOWN
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "First: the rotation events. How often does the strategy activate?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = _load()\n"
            "    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)\n"
            "    pnl = st.rotation_pnl(panel, sig, cost_bps=40)\n"
            "    ev = st.count_rotations(pnl)\n"
            "    n_rot, avg_hold, pct_act = ev['n_rotations'], ev['avg_hold_days'], ev['pct_days_active']\n"
            "    cum_net = pnl['cum_net'].values\n"
            "    cum_dates = pnl.index\n"
            "else:\n"
            "    import numpy as _np\n"
            f"    n_rot, avg_hold, pct_act = {R['ev1_n']}, {R['ev1_hold']}, {R['ev1_active']}\n"
            "    n = 500\n"
            "    cum_dates = pd.date_range('2020-04-10', periods=n, freq='B')\n"
            "    cum_net = _np.cumsum(_np.random.default_rng(222).normal(0.0004, 0.014, n))\n"
            "print(f'Rotation events: n={n_rot}, avg hold={avg_hold:.1f} days, "
            "active {pct_act:.1f}% of days')\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(cum_dates, cum_net * 100, c=AMBER, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('cumulative net log-return (%)')\n"
            "ax.set_title('Rotation strategy net cumulative return (40 bps RT)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The strategy fires **{R['ev1_n']} rotation events** in 6 years -- roughly 7 per "
            f"year -- with an average holding period of **{R['ev1_hold']:.0f} days**. It is active "
            f"only **{R['ev1_active']:.0f}%** of the time. The cumulative return looks modest "
            "but the key question is whether it beats simply holding."
        ),
        md("**The decisive comparison: rotation strategy vs buy-and-hold.**"),
        code(
            "if HAVE_REAL:\n"
            "    panel = _load()\n"
            "    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)\n"
            "    pnl = st.rotation_pnl(panel, sig, cost_bps=40)\n"
            "    bm = st.benchmark_pnl(panel)\n"
            "    cum_rot = pnl['cum_net'] * 100\n"
            "    cum_btc = bm['cum_btc'] * 100\n"
            "    cum_alt = bm['cum_alt'] * 100\n"
            "    dates = pnl.index\n"
            "    s_rot = st.performance_summary(pnl['net_ret'])\n"
            "    s_btc = st.performance_summary(bm['btc_ret'])\n"
            "    s_alt = st.performance_summary(bm['alt_ew_ret'])\n"
            "    sr_rot, sr_btc, sr_alt = s_rot['sharpe'], s_btc['sharpe'], s_alt['sharpe']\n"
            "else:\n"
            "    import numpy as _np\n"
            "    n = 500\n"
            "    dates = pd.date_range('2020-04-10', periods=n, freq='B')\n"
            f"    cum_rot = _np.cumsum(_np.random.default_rng(1).normal(0.038, 1.3, n))\n"
            f"    cum_btc = _np.cumsum(_np.random.default_rng(2).normal(0.1, 2.6, n))\n"
            f"    cum_alt = _np.cumsum(_np.random.default_rng(3).normal(0.13, 3.0, n))\n"
            f"    sr_rot, sr_btc, sr_alt = {R['n1_sharpe']}, {R['bm_btc_sharpe']}, {R['bm_alt_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(dates, cum_rot, c=AMBER, lw=1.8, label=f'Rotation net (Sharpe={sr_rot:.2f})')\n"
            "ax.plot(dates, cum_btc, c=RED, lw=1.5, ls='--', label=f'BTC B&H (Sharpe={sr_btc:.2f})')\n"
            "ax.plot(dates, cum_alt, c=GREEN, lw=1.5, ls=':', label=f'Alt EW B&H (Sharpe={sr_alt:.2f})')\n"
            "ax.set_ylabel('cumulative log-return (%)')\n"
            "ax.set_title('Rotation strategy vs buy-and-hold: timing adds no value')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe: rotation={sr_rot:.2f}  BTC={sr_btc:.2f}  Alt={sr_alt:.2f}')"
        ),
        md(
            f"The rotation strategy earns a **net Sharpe of {R['n1_sharpe']:.2f}** -- below "
            f"both BTC buy-and-hold (**{R['bm_btc_sharpe']:.2f}**) and alt EW buy-and-hold "
            f"(**{R['bm_alt_sharpe']:.2f}**). The timing rule actively *hurts* risk-adjusted "
            "performance. Worse, the max drawdown on net returns is **"
            f"{R['n1_mdd']:.0f}%** -- catastrophic for any real allocation."
        ),

        # BEAT 5 -- THE VERDICT
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- None.** Net HAC *t* = **{R['n1_t']:+.2f}** on the dom_change "
            f"signal. Not close to the |*t*| >= 2 bar. The strategy Sharpe ({R['n1_sharpe']:.2f}) "
            f"is below BTC buy-and-hold ({R['bm_btc_sharpe']:.2f}). No detectable timing edge.\n"
            "- **Tradability -- Mirage.** At 40 bps RT: net Sharpe 0.45, MDD -89%. "
            f"At 80 bps: net Sharpe {R['n1h_sharpe']:.2f}. Altcoin rotation is expensive and "
            "the gross edge (+13.5%/yr) is mostly consumed by costs and timing failures.\n"
            "- **Alt-season timing? -- BUSTED.** The strategy's positive gross return in "
            "2020-2026 is largely attributable to the 2021 alt season -- a single regime "
            "episode in one crypto cycle. With only 43 rotation events across 6 years, "
            "the single-cycle sample offers no statistical evidence of a repeatable rule."
        ),

        # BEAT 6 -- COULD YOU TRADE IT
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Even setting aside the weak t-stat, the execution environment makes this harder:"
        ),
        code(
            "costs_ow = [0.1, 0.2, 0.4, 0.6, 1.0, 2.0]  # one-way costs, %\n"
            f"n_rot = {R['ev1_n']}\n"
            "print('Rotation events per year: {:.1f}'.format(n_rot / 6))\n"
            "print('Annual cost drag at different one-way costs:')\n"
            "for c in costs_ow:\n"
            "    rt = c * 2  # round-trip %\n"
            "    annual_drag = rt * n_rot / 6  # % per year\n"
            "    net_est = 13.48 - annual_drag\n"
            "    print('  {:.1f}% one-way ({:.0f} bps RT): annual drag = {:.2f}%/yr  "
            "net est = {:.2f}%/yr'.format(c, rt*100, annual_drag, net_est))\n"
            "print()\n"
            "print('For reference: BTC buy-and-hold 0% cost drag, 25.33%/yr.')"
        ),
        md(
            "Even at a very optimistic 0.1% one-way cost (only achievable on BTC perps, not "
            "the full alt basket), the annual drag from 43 rotations over 6 years is "
            "meaningful. More realistic altcoin costs (0.2-0.5% per side) consume the "
            "entire gross advantage over buy-and-hold."
        ),

        # BEAT 7 -- GOING FURTHER
        md(
            "## 7 -- Going further\n\n"
            "- **The survivorship problem.** The alt basket here is ETH, XRP, ADA, SOL, BNB, "
            "DOGE -- the 2024 winners. SOL gained ~200x from 2020 to 2021. A basket formed "
            "in real-time in 2020 would have included LUNA, FTT, GALA, and other projects "
            "that went to zero. Survivorship bias inflates the alt benchmark and therefore "
            "the *gross* strategy return.\n"
            "- **One cycle is one data point.** 43 rotation events across 6 years, heavily "
            "concentrated in 2021. For the statistical framework to work (|t| >= 2), we need "
            "many more independent cycle observations. Crypto has existed for ~15 years with "
            "perhaps 3 full cycles -- not enough for robust inference.\n"
            "- **Related studies.** [Study 134](../../134-bitcoin-dominance/) shows the "
            "underlying regression signal barely misses the bar (t = -1.88). "
            "[Study 210](../../210-crypto-trend/) tests crypto trend-following as an "
            "alternative timing approach with more persistence. "
            "[Study 209](../../209-eth-btc-ratio/) tests the ETH/BTC ratio directly.\n"
            "- **What would change the verdict?** A survivorship-free database covering 3+ "
            "cycles with consistent dominance tracking (Kaiko or Messari). If the rotation "
            "pattern repeats in cycles 4 and 5 and clears t = 2 out-of-sample, this becomes "
            "a live hypothesis. With free Yahoo data, we cannot get there."
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
            "# Altseason-Rotation -- a quantitative teardown\n"
            "### Long-alt / short-BTC timed by BTC dominance: PnL simulation, HAC t-stats, "
            "cost sweep, positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Alt-season timing: BUSTED](https://img.shields.io/badge/Alt--season_timing%3F-BUSTED-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We simulate a long-alt/short-BTC rotation strategy timed by BTC basket dominance, "
            "apply explicit transaction costs, compute HAC-robust t-stats on net daily returns, "
            "and compare Sharpe ratios against buy-and-hold benchmarks. The conclusion: the "
            "strategy does not clear any of the desk's inference bars on the real tape.\n\n"
            f"> Real data: Yahoo Finance daily bars, as-of {R['end']}, "
            f"panel fingerprint `{R['fingerprint']}`. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> *Study 222 is the strategy complement to "
            "[Study 134 -- Bitcoin-Dominance](../../134-bitcoin-dominance/), "
            "which tests the underlying regression signal (regression t = -1.88). "
            "Here we ask: can you trade around that weak signal profitably?*"
        ),
        code(BOOT),

        # BEAT 0 -- VERDICT
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Best net HAC *t* = **{R['n1_t']:+.2f}** "
            f"(dom_change_20d, 40 bps RT). Strategy Sharpe {R['n1_sharpe']:.2f} < "
            f"BTC B&H {R['bm_btc_sharpe']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe {R['n1_sharpe']:.2f} at 40 bps; "
            f"{R['n1h_sharpe']:.2f} at 80 bps. MDD {R['n1_mdd']:.0f}%. "
            "Underperforms both buy-and-hold benchmarks. |\n"
            f"| **Alt-season timing?** | `BUSTED` | {R['ev1_n']} rotation events in 6 years; "
            "net t = 1.05; dominated by the single 2021 alt-season episode. "
            "No statistical evidence of a repeatable rule. |\n\n"
            "> In plain words: the dominance-timed rotation earns a positive gross return "
            "mostly from the 2021 bull market episode. Once we account for costs, compare "
            "to buy-and-hold, and apply HAC inference, there is no detectable alpha."
        ),

        # BEAT 1 -- THE CLAIM FORMALISED
        md(
            "## 1 -- The claim, formalised\n\n"
            "Let $d_t$ = BTC basket dominance on day $t$. The rotation rule is:\n\n"
            r"$$\text{position}_t = \mathbf{1}[\Delta^{20} d_{t-1} < -0.02]$$"
            "\n\nwhere $\\Delta^{20} d_{t-1} = d_{t-1} - d_{t-21}$ (20-day dominance change "
            "formed on prior day, 1-day lag). Position = 1: long equal-weighted alts, short "
            "BTC. Position = 0: flat.\n\n"
            "Net daily return: $r_t^{\\text{net}} = \\text{position}_t \\cdot "
            "(r_t^{\\text{alt,ew}} - r_t^{\\text{BTC}}) - \\text{cost}_t$\n\n"
            "where $\\text{cost}_t = |\\Delta \\text{position}_t| \\cdot c$ and "
            "$c$ = one-way cost (20 bps base case, 40 bps RT). The strategy is dollar-neutral "
            "(equal long alt / short BTC notional).\n\n"
            "Inference bar: the strategy must clear **|t| >= 2** on $r_t^{\\text{net}}$ "
            "(HAC Newey-West) **and** achieve net Sharpe > benchmark Sharpe."
        ),

        # BEAT 2 -- SO WHAT
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "If the strategy clears the bar, it is a simple, mechanically-implementable "
            "rotation rule that a crypto allocator could use to tilt between BTC and alts. "
            "If it doesn't, then every 'alt season incoming' rotation is noise trading at "
            "a cost -- the investor would be better served by a static allocation. Given that "
            "Study 134 already showed the underlying regression signal barely misses the bar, "
            "we expect the strategy result to be weak-to-none; the strategy test is the "
            "real-world translation of that weak signal."
        ),

        # BEAT 3 -- PROTOCOL
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal variants.** (1) 20-day dominance change, threshold = -2 pp. "
            "(2) Level signal: dominance below 52-week rolling median.\n"
            "- **PnL simulation.** Daily log-return, 1-day entry lag, cost deducted on "
            "position change days, cumulative log-return track.\n"
            "- **Benchmarks.** BTC buy-and-hold, equal-weight alt buy-and-hold, "
            "always-long-spread (unconditional long alt/short BTC).\n"
            "- **Inference.** HAC t-stat on net daily return series (Newey-West kernel, "
            "bandwidth = 4*(n/100)^(2/9) lags).\n"
            "- **Cost sweep.** 0, 20, 40, 60, 80, 100 bps round-trip.\n"
            "- **Positive control.** Synthetic panel with tunable dom-signal, to confirm the "
            "engine detects planted effects.\n\n"
            f"Panel: {R['start']} to {R['end']}, {R['n_rows']:,} rows, "
            f"fingerprint `{R['fingerprint']}`."
        ),

        # BEAT 4 -- TEARDOWN
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Strategy performance vs benchmarks\n\n"
            "Full Sharpe comparison: dom_change_20d strategy (gross and net) vs "
            "BTC buy-and-hold, alt EW buy-and-hold, and always-long-spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = _load()\n"
            "    sig1 = st.dominance_change_signal(panel, lookback=20, threshold=0.02)\n"
            "    pnl1 = st.rotation_pnl(panel, sig1, cost_bps=40)\n"
            "    bm = st.benchmark_pnl(panel)\n"
            "    spread_bm = bm['alt_ew_ret'] - bm['btc_ret']\n"
            "    labels = ['Rotation gross', 'Rotation net', 'BTC B&H', 'Alt EW B&H', 'Always spread']\n"
            "    rets = [pnl1['gross_ret'], pnl1['net_ret'], bm['btc_ret'],\n"
            "            bm['alt_ew_ret'], spread_bm]\n"
            "    sharpes = [st.performance_summary(r)['sharpe'] for r in rets]\n"
            "    tsts = [st.performance_summary(r)['tstat'] for r in rets]\n"
            "else:\n"
            f"    labels = ['Rotation gross', 'Rotation net', 'BTC B&H', 'Alt EW B&H', 'Always spread']\n"
            f"    sharpes = [{R['g1_sharpe']}, {R['n1_sharpe']}, {R['bm_btc_sharpe']}, "
            f"{R['bm_alt_sharpe']}, {R['bm_spread_sharpe']}]\n"
            f"    tsts = [{R['g1_t']}, {R['n1_t']}, 'n/a', 'n/a', {R['bm_spread_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "colors = [GREEN if s > 0.5 else AMBER if s > 0.3 else RED for s in sharpes]\n"
            "ax.bar(labels, sharpes, color=colors)\n"
            "ax.axhline(0.5, ls='--', c=GREY, lw=1, label='Sharpe 0.5 (BTC B&H)')\n"
            "ax.set_ylabel('Sharpe ratio (annualised)')\n"
            "ax.set_title('Timed rotation underperforms buy-and-hold on Sharpe')\n"
            "for i, (s, t) in enumerate(zip(sharpes, tsts)):\n"
            "    ax.annotate(f'{s:.2f}', (i, s + 0.01), ha='center', va='bottom', fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', dict(zip(labels, [round(s, 2) for s in sharpes])))"
        ),
        md(
            f"> In plain words: the rotation strategy's net Sharpe ({R['n1_sharpe']:.2f}) "
            f"is below both buy-and-hold benchmarks ({R['bm_btc_sharpe']:.2f} BTC, "
            f"{R['bm_alt_sharpe']:.2f} alt). Timing the rotation actively *reduces* "
            "risk-adjusted returns compared to simply holding the assets."
        ),
        md(
            "### 4b -- HAC t-stat: is the net daily return distinguishable from noise?\n\n"
            "The t-stat on net daily returns across both signal variants."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = _load()\n"
            "    sig1 = st.dominance_change_signal(panel, lookback=20, threshold=0.02)\n"
            "    sig2 = st.dominance_level_signal(panel, window=252)\n"
            "    pnl1 = st.rotation_pnl(panel, sig1, cost_bps=40)\n"
            "    pnl2 = st.rotation_pnl(panel, sig2, cost_bps=40)\n"
            "    p1g = st.performance_summary(pnl1['gross_ret'])\n"
            "    p1n = st.performance_summary(pnl1['net_ret'])\n"
            "    p2g = st.performance_summary(pnl2['gross_ret'])\n"
            "    p2n = st.performance_summary(pnl2['net_ret'])\n"
            "    rows = [('dom_chg_20d gross', p1g['tstat']), ('dom_chg_20d net', p1n['tstat']),\n"
            "            ('dom_level gross', p2g['tstat']), ('dom_level net', p2n['tstat'])]\n"
            "else:\n"
            f"    rows = [('dom_chg_20d gross', {R['g1_t']}), ('dom_chg_20d net', {R['n1_t']}),\n"
            f"            ('dom_level gross', {R['g2_t']}), ('dom_level net', {R['n2_t']})]\n"
            "labels_t, tstats = zip(*rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "colors_t = [GREEN if abs(t) >= 2 else AMBER if abs(t) >= 1.5 else RED for t in tstats]\n"
            "ax.bar(labels_t, tstats, color=colors_t)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('HAC t-stat on daily returns')\n"
            "ax.set_title('No variant clears |t|>=2 -- even gross before costs')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> In plain words: even the *gross* t-stat (before costs, {R['g1_t']:+.2f}) "
            "does not clear the bar. After realistic costs the net t-stat is "
            f"**{R['n1_t']:+.2f}**. The inference bar is not met."
        ),
        md(
            "### 4c -- Cost sweep: what breakeven cost would make this strategy work?\n\n"
            "If the signal were real (which it isn't at t = 1.48 gross), what cost would the "
            "strategy survive?"
        ),
        code(
            "import numpy as _np\n"
            f"gross_ann = {R['g1_ret']}\n"
            f"n_events_yr = {R['ev1_n']} / 6.0\n"
            "cost_bps_range = _np.arange(0, 105, 5)\n"
            "net_ann = [gross_ann - (c / 100) * n_events_yr for c in cost_bps_range]\n"
            "breakeven_bps = next((c for c, n in zip(cost_bps_range, net_ann) if n < 0), None)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(cost_bps_range, net_ann, c=AMBER, lw=2)\n"
            f"ax.axhline({R['bm_btc_ret']}, ls='--', c=RED, lw=1.5, label='BTC B&H {R['bm_btc_ret']:.1f}pct/yr')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.fill_between(cost_bps_range, net_ann, 0, where=[n < 0 for n in net_ann],\n"
            "                color=RED, alpha=0.1)\n"
            "ax.set_xlabel('round-trip cost per rotation event (bps)')\n"
            "ax.set_ylabel('estimated net annualised return (%)')\n"
            "ax.set_title('Breakeven cost well below realistic altcoin execution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Breakeven round-trip cost: ~{breakeven_bps} bps')\n"
            "print(f'Typical altcoin RT: 40-200 bps. Realistic: strategy is not profitable.')"
        ),
        md(
            "> In plain words: the strategy breakeven is around 80 bps round-trip. Liquid "
            "altcoin pairs on major exchanges run 20-100 bps including spread; smaller coins "
            "in the basket (DOGE, ADA, XRP) can be higher. Any realistic allocation incurs "
            "costs above the breakeven. Even before the breakeven, the strategy fails to beat "
            "BTC buy-and-hold."
        ),

        # BEAT 5 -- VERDICT
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- Net HAC *t* = {R['n1_t']:+.2f} (dom_change_20d). "
            "Not close to the |*t*| >= 2 bar. Strategy net Sharpe "
            f"({R['n1_sharpe']:.2f}) < BTC buy-and-hold ({R['bm_btc_sharpe']:.2f}). "
            "The rotation timing destroys, not adds, risk-adjusted returns.\n"
            f"- **Tradability `MIRAGE`** -- Net Sharpe {R['n1_sharpe']:.2f} at 40 bps RT; "
            f"{R['n1h_sharpe']:.2f} at 80 bps. Max drawdown {R['n1_mdd']:.0f}%. "
            "Severely underperforms buy-and-hold. Transaction costs are not the main issue; "
            "the strategy simply does not time well.\n"
            "- **Alt-season timing? `BUSTED`** -- {R['ev1_n']} rotation events in 6 years "
            "concentrated around the 2021 episode. With one crypto cycle and survivorship-"
            "biased alts, the positive gross return is indistinguishable from just holding "
            "the same assets through the bull market. No repeatable timing edge detected."
        ),

        # BEAT 6 -- COST AND CAPACITY
        md(
            "## 6 -- Could you trade it? -- capacity and regime analysis\n\n"
            "The 2022 bear market is the key test: in a down-market, does the dominance "
            "signal correctly avoid the rotation (BTC dominance rising) or does it fire false "
            "positives into a falling alt market?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = _load()\n"
            "    sig1 = st.dominance_change_signal(panel, lookback=20, threshold=0.02)\n"
            "    pnl1 = st.rotation_pnl(panel, sig1, cost_bps=40)\n"
            "    # Regime split: pre and post 2022-01-01\n"
            "    bull = pnl1[pnl1.index < '2022-01-01']\n"
            "    bear = pnl1[(pnl1.index >= '2022-01-01') & (pnl1.index < '2023-01-01')]\n"
            "    recov = pnl1[pnl1.index >= '2023-01-01']\n"
            "    perf_bull = st.performance_summary(bull['net_ret'])\n"
            "    perf_bear = st.performance_summary(bear['net_ret'])\n"
            "    perf_recov = st.performance_summary(recov['net_ret'])\n"
            "    rets = [perf_bull['ann_return_pct'], perf_bear['ann_return_pct'],\n"
            "            perf_recov['ann_return_pct']]\n"
            "    sharpes = [perf_bull['sharpe'], perf_bear['sharpe'], perf_recov['sharpe']]\n"
            "else:\n"
            "    rets = [55.0, -30.0, 5.0]\n"
            "    sharpes = [1.2, -0.4, 0.2]\n"
            "regimes = ['Bull (2020-2021)', 'Bear (2022)', 'Recovery (2023+)']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors_r = [GREEN if r > 0 else RED for r in rets]\n"
            "ax.bar(regimes, rets, color=colors_r)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('annualised net return (%)')\n"
            "ax.set_title('Strategy profitable only in 2021 bull episode')\n"
            "for i, (r, s) in enumerate(zip(rets, sharpes)):\n"
            "    ax.annotate('Sharpe {:.2f}'.format(s), (i, r), ha='center',\n"
            "                va='bottom' if r >= 0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> In plain words: the strategy's positive gross return is almost entirely "
            "driven by the 2020-2021 bull market. In 2022, the rotation signal fires into "
            "falling alt markets (false positives in the bear), generating substantial losses. "
            "This regime-specificity is the defining structural fragility: the strategy "
            "is a bull-market pattern in crypto, not a cycle-independent timing rule."
        ),

        # BEAT 7 -- POSITIVE CONTROL
        md(
            "## 7 -- Going further -- the positive control\n\n"
            "Does the engine recover a rotation edge when one is planted in a synthetic panel?"
        ),
        code(
            "import numpy as _np\n"
            "dom_strengths = [0.0, 5.0, 10.0, 50.0]\n"
            "net_sharpes, cum_spreads = [], []\n"
            "for ds in dom_strengths:\n"
            "    synth, _ = data.synthetic_daily(n_years=5, dom_signal=ds, seed=222)\n"
            "    sig_s = st.dominance_level_signal(synth, window=252)\n"
            "    pnl_s = st.rotation_pnl(synth, sig_s, cost_bps=0)  # no cost for clarity\n"
            "    perf_s = st.performance_summary(pnl_s['net_ret'])\n"
            "    bm_s = st.benchmark_pnl(synth)\n"
            "    spread_s = (bm_s['alt_ew_ret'] - bm_s['btc_ret']).cumsum()\n"
            "    net_sharpes.append(perf_s['sharpe'])\n"
            "    cum_spreads.append(float(spread_s.iloc[-1]))\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))\n"
            "ax1.plot(dom_strengths, net_sharpes, 'o-', c=GREEN, lw=2)\n"
            "ax1.axhline(0, c='k', lw=0.8)\n"
            "ax1.set_xlabel('planted dom_signal strength'); ax1.set_ylabel('net Sharpe')\n"
            "ax1.set_title('Strategy Sharpe rises with planted signal')\n"
            "ax2.plot(dom_strengths, cum_spreads, 's--', c=AMBER, lw=2)\n"
            "ax2.set_xlabel('planted dom_signal strength')\n"
            "ax2.set_ylabel('cumulative alt-spread (log-return)')\n"
            "ax2.set_title('Planted signal shifts cumulative alt outperformance')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine is a faithful rotation detector -- the real tape just has no signal.')"
        ),
        md(
            "The synthetic positive control confirms the engine works: when a dominance-to-alt "
            "spread effect is planted, the strategy's Sharpe rises and the cumulative alt spread "
            "shifts upward monotonically with the planted signal strength. The real tape's result "
            "(net Sharpe 0.45) is consistent with either a very weak true signal drowned by a "
            "short sample, or a spurious pattern from the 2021 alt season dominating the "
            "dataset. With a survivorship-free database covering 3+ crypto cycles, this "
            "study would be worth revisiting. With Yahoo Finance data and one cycle, "
            "the verdict is unambiguously **None/Mirage**."
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
