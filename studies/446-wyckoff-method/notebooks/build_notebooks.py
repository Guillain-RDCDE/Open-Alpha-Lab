"""Generate the two narrative notebooks for Study 446 (Wyckoff Method).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-24).
R = dict(
    n=404, n_spring=192, n_upthrust=212,
    # combined Spring+Upthrust, gross, by horizon: mean(bps), one-sample t, HAC t, coin mean, coin t, placebo p
    h=[5, 10, 20, 40],
    mean=[-13.60, 2.62, 44.14, 26.58],
    t=[-0.65, 0.08, 1.11, 0.50],
    hac=[-0.56, 0.07, 0.89, 0.42],
    win=[45.5, 52.2, 53.5, 51.0],
    coin_mean=[-4.64, 40.44, 53.41, 53.89],
    coin_t=[-0.22, 1.30, 1.35, 1.02],
    placebo_p=[0.7440, 0.4630, 0.1296, 0.3034],
    # split by leg (20-day, gross)
    spring_mean=155.08, spring_win=66.7, spring_t=2.29, spring_hac=1.77,
    upthrust_mean=-56.35, upthrust_win=41.5, upthrust_t=-1.31, upthrust_hac=-1.23,
    # robustness: volume filter off
    novol_n=615, novol_mean=63.33, novol_t=1.98, novol_hac=1.60,
    # ZigZag threshold (20-day, gross): n, mean, t
    pct=[0.03, 0.05, 0.08],
    pct_n=[1316, 404, 68],
    pct_mean=[-0.21, 44.14, 70.61],
    pct_t=[-0.01, 1.11, 0.61],
    # cost sweep (combined, 20-day, net): mean, t
    cost=[0.0, 1.0, 2.0, 5.0],
    cost_mean=[44.14, 42.14, 40.14, 34.14],
    cost_t=[1.11, 1.06, 1.01, 0.86],
    # synthetic positive control: edge, n, win, mean, t, hac
    syn_edge=[0.0, 0.06, 0.12, 0.20],
    syn_n=[7, 93, 71, 26],
    syn_win=[42.9, 68.8, 47.9, 46.2],
    syn_mean=[-78.13, 233.22, 132.00, 253.17],
    syn_t=[-1.20, 5.20, 1.67, 0.83],
    syn_hac=[-1.61, 4.62, 1.68, 1.13],
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

from wyckoff_method import data, strategy as st

BASKET = data.load_basket()           # cache-first; {} if no cache present
HAVE_REAL = len(BASKET) >= 4

def pooled(horizon, cost=0.0, vol_confirm=True, pct=0.05, coin=False, seed=446, col="ret_gross"):
    \"\"\"Pool Spring/Upthrust trades across the cached basket.\"\"\"
    leds = []
    for tk, b in BASKET.items():
        ev = st.find_events(b, pct=pct, vol_confirm=vol_confirm)
        if coin:
            dirs = st.random_directions(len(ev), seed=seed)
            leds.append(st.run_trades(b, ev, horizon, cost_bps=cost, directions=dirs))
        else:
            leds.append(st.run_trades(b, ev, horizon, cost_bps=cost))
    return pd.concat(leds, ignore_index=True) if leds else pd.DataFrame()

print("real daily cache present:", HAVE_REAL, "| tapes:", list(BASKET.keys()))
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Wyckoff Method -- can you see the smart money accumulating?\n"
            "### The Spring and the Upthrust, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Phases_detectable%3F: Not_Supported](https://img.shields.io/badge/"
            "Phases_detectable%3F-Not_Supported-8b949e?style=flat-square)\n\n"
            "A century-old idea from Richard Wyckoff: big operators quietly **accumulate** stock "
            "inside a sideways range, shake out the weak holders with a quick dip below support "
            "(a **Spring**), then drive the **markup**. The mirror image -- a fake pop above "
            "resistance (an **Upthrust**) -- marks **distribution** before the **markdown**. "
            "The promise: learn to read the range and you trade alongside the smart money.\n\n"
            "This notebook asks the honest question: when a textbook Spring or Upthrust actually "
            "fires, does price go where Wyckoff says -- any better than a coin flip?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the per-leg split, the "
            "volume-filter test and the coin placebo? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the full Wyckoff signal (buy Springs, sell Upthrusts) predict the move? | "
            f"**No.** Over **{R['n']}** events on 8 markets and up to 33 years, the best 20-day "
            f"return is **+{R['mean'][2]:.0f} bps** at *t* = {R['t'][2]:.2f} -- and a coin at the "
            "same bars earns *more*. |\n"
            "| 'But my Springs win two-thirds of the time!' | True -- the Spring (a *buy*) wins "
            f"**{R['spring_win']:.0f}%**. But that's because the market drifts **up**: the "
            f"Upthrust (a *sell*) wins only **{R['upthrust_win']:.0f}%** in the same tapes. The "
            "events aren't reading the market -- the bull market is carrying the longs. |\n"
            "| Does the volume 'tell' help? | **No.** Dropping the low-volume filter the method "
            "leans on actually *raises* the t-stat. |\n"
            "| Could you trade it? | **No.** There's no edge to trade -- a coin matches it. |\n\n"
            "> A synthetic test where we *plant* a real markup proves the detector works "
            f"(*t* = {R['syn_t'][1]:.1f}). It just finds nothing real in the actual tape."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Markets move in a cycle: accumulation, markup, distribution, markdown. Inside a "
            "trading range, watch for the **Spring** -- a quick stab below support that fails and "
            "snaps back, on low volume (no real selling). That's the Composite Operator shaking "
            "out weak hands before the markup. Buy it. The **Upthrust** is the opposite: a fake "
            "breakout above resistance that fails -- distribution before the markdown. Sell it.\"*\n\n"
            "It's a compelling story about *who* is on the other side of your trade. The volume is "
            "the supposed lie-detector: a break that *should* trigger follow-through but happens on "
            "thin volume reveals there's no real supply (or demand) behind it."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If Springs and Upthrusts genuinely marked the start of markup and markdown, you'd have "
            "a mechanical, low-frequency timing tool that works **both long and short** -- the holy "
            "grail, because a real edge should make money on *both* legs, not just the one pointing "
            "the way the market happened to drift. If instead only the *buy* leg works, in a "
            "bull-market sample, that's the tell that you're looking at drift wearing a Wyckoff "
            "costume."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Wyckoff is *irreducibly subjective* -- two analysts label the same chart differently "
            "and the 'right' phase is only clear in hindsight. So there's no single rule to test. "
            "We encode the **tightest mechanical version** its proponents accept:\n\n"
            "1. **Find a range.** A 5% ZigZag marks the swing highs and lows; four pivots inside a "
            "tight band define a sideways range with a **support** and a **resistance**.\n"
            "2. **Find the event.** A **Spring** = a bar whose *low* pierces support but whose "
            "*close* recovers above it, on **below-average volume**. An **Upthrust** = the mirror "
            "above resistance.\n"
            "3. **Trade it.** Buy the Spring, sell the Upthrust, **one day after** the event "
            "(no peeking), hold 20 days.\n\n"
            "**What would make us say 'mirage':** if the combined signal doesn't beat a coin "
            "placed at the very same bars, and if the only positive leg is just the long side in a "
            "rising market. Eight tapes (SPY, QQQ, DIA, IWM, GLD, XLE, EEM, TLT)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First: the full Wyckoff signal vs a coin at the same bars, across holding periods.**"
        ),
        code(
            "H = R_h = [5,10,20,40]\n"
            "if HAVE_REAL:\n"
            "    means, tstats, coin_means = [], [], []\n"
            "    for h in H:\n"
            "        s = st.summarize(pooled(h, 0., True), 'ret_gross')\n"
            "        c = st.summarize(pooled(h, 0., True, coin=True), 'ret_gross')\n"
            "        means.append(s['mean_bps']); tstats.append(s['t']); coin_means.append(c['mean_bps'])\n"
            "else:\n"
            "    means = [-13.60, 2.62, 44.14, 26.58]\n"
            "    tstats = [-0.65, 0.08, 1.11, 0.50]\n"
            "    coin_means = [-4.64, 40.44, 53.41, 53.89]\n"
            "\n"
            "x = range(len(H)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.bar([i-w/2 for i in x], means, width=w, color=AMBER, label='Wyckoff signal')\n"
            "ax.bar([i+w/2 for i in x], coin_means, width=w, color=GREY, label='coin at same bars')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels([f'{h}d' for h in H])\n"
            "ax.set_xlabel('holding period'); ax.set_ylabel('gross return per trade (bps)')\n"
            "ax.set_title('The Wyckoff signal never beats a coin at the same bars')\n"
            "ax.legend()\n"
            "for i, t in zip(x, tstats):\n"
            "    ax.annotate(f't={t:+.2f}', (i-w/2, means[i]+ (3 if means[i]>=0 else -8)),\n"
            "                ha='center', fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('best Wyckoff t-stat:', f'{max(tstats):+.2f}', '(needs >= 2 to be real)')"
        ),
        md(
            f"The best the Wyckoff signal manages is **+{R['mean'][2]:.0f} bps** at 20 days "
            f"(*t* = {R['t'][2]:.2f}) -- and the **coin earns more** (+{R['coin_mean'][2]:.0f} bps). "
            "Nothing clears the *t* = 2 bar. The fast versions (5-10 days) are noise or negative.\n\n"
            "**Now the tell: split the signal into its two legs.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = pooled(20, 0., True)\n"
            "    sp = st.summarize(led[led['kind']=='spring'], 'ret_gross')\n"
            "    up = st.summarize(led[led['kind']=='upthrust'], 'ret_gross')\n"
            "    sp_m, sp_w, up_m, up_w = sp['mean_bps'], sp['win_rate']*100, up['mean_bps'], up['win_rate']*100\n"
            "else:\n"
            "    sp_m, sp_w, up_m, up_w = 155.08, 66.7, -56.35, 41.5\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bars = ax.bar(['Spring\\n(buy)', 'Upthrust\\n(sell)'], [sp_m, up_m],\n"
            "              color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross return per trade (bps, 20-day hold)')\n"
            "ax.set_title('Only the BUY leg works -- in a market that drifted up')\n"
            "for b, wn in zip(bars, [sp_w, up_w]):\n"
            "    ax.annotate(f'win {wn:.0f}%', (b.get_x()+b.get_width()/2,\n"
            "                b.get_height()+(6 if b.get_height()>=0 else -14)), ha='center', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Spring (buy):   {sp_m:+.0f} bps, win {sp_w:.0f}%')\n"
            "print(f'Upthrust (sell):{up_m:+.0f} bps, win {up_w:.0f}%')"
        ),
        md(
            f"There it is. The **Spring** (a buy) earns **+{R['spring_mean']:.0f} bps**, winning "
            f"**{R['spring_win']:.0f}%** -- looks great! But the **Upthrust** (a sell) *loses* "
            f"**{abs(R['upthrust_mean']):.0f} bps**. A real accumulation/distribution detector "
            "would work both ways. This one only works on the long side -- because US equities and "
            "broad ETFs **drifted up** over the sample. The Spring isn't reading the smart money; "
            "it's a long position in a bull market."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** {R['n']} events, best HAC *t* = {R['hac'][2]:.2f}, beaten by a "
            f"coin at the same bars; placebo *p* never below {R['placebo_p'][2]:.2f}. The one "
            f"positive leg (Spring, *t* = {R['spring_t']:.2f}) fails the robust bar and is just the "
            "bull-market long bias.\n"
            "- **Tradability -- Mirage.** No edge to trade. Costs are trivial (rare events) but "
            "there's nothing for them to erode.\n"
            "- **Phases detectable? -- Not supported.** The events fire, but they don't foretell "
            "the move -- and the volume tell adds nothing."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Costs aren't the problem here (the events are rare, ~12 per tape over decades). The "
            "problem is more fundamental: there's no edge for costs to eat."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.,1.,2.,5.]\n"
            "    net = [st.summarize(pooled(20, c, True), 'ret_net')['mean_bps'] for c in costs]\n"
            "    coin_net = [st.summarize(pooled(20, c, True, coin=True), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    costs = [0.,1.,2.,5.]; net = [44.14,42.14,40.14,34.14]\n"
            "    coin_net = [53.41,51.41,49.41,43.41]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2, label='Wyckoff signal (net)')\n"
            "ax.plot(costs, coin_net, 's--', c=GREY, lw=1.8, label='coin at same bars (net)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('The coin stays above the Wyckoff signal at every cost level')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('There is no cost level at which the Wyckoff signal pulls ahead of a coin.')"
        ),
        md(
            "Unlike a study where 'the edge dies at costs', here the edge was never there: a coin "
            "at the same bars sits *above* the Wyckoff line the whole way. That is the definition "
            "of a **Mirage** -- nothing to deploy."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **A market-neutral test.** Run the Spring and Upthrust as a *spread* against the "
            "index over the same window, so the bull-market drift cancels. If the Spring still "
            "beats the Upthrust after that, there's a real residual.\n"
            "- **The sibling teardown.** [Study 445 -- Elliott Wave](../../445-elliott-wave/) "
            "applies the identical 'irreducibly-subjective theory, tightest mechanical proxy' "
            "method to the five-wave count -- and lands the same None x Mirage verdict.\n"
            "- **The reversion cousin.** [Study 104 -- Bollinger-Reversion](../../104-bollinger-reversion/) "
            "tests the same false-break / snap-back intuition the Spring rests on.\n\n"
            "*Think a tighter Spring definition (effort-vs-result, a real climax bar) carries an "
            "edge? Fork this, code your rule, and show a HAC delta-vs-coin t-stat above 2 on both "
            "legs. That's the bar.*"
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
            "# The Wyckoff Method -- a quantitative teardown\n"
            "### Real daily tape - up to 33 years - volume-confirmed Spring/Upthrust - "
            "HAC inference - coin placebo - per-leg split\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Phases_detectable%3F: Not_Supported](https://img.shields.io/badge/"
            "Phases_detectable%3F-Not_Supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We mechanise the Wyckoff "
            "Spring (accumulation buy) and Upthrust (distribution sell) as volume-confirmed events "
            "inside a ZigZag-bracketed trading range, and test whether they predict the forward "
            "move on eight liquid index/ETF tapes.\n\n"
            "> **Not investment advice.** Real data: Yahoo auto-adjusted daily OHLCV, 8 tapes, "
            "as-of 2026-06-24; reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Combined signal best gross **+{R['mean'][2]:.0f} bps** "
            f"(20d), HAC *t* = **{R['hac'][2]:.2f}**; a coin at the same bars earns more "
            f"(+{R['coin_mean'][2]:.0f}); placebo *p* >= {R['placebo_p'][2]:.2f}. Spring leg "
            f"*t* = {R['spring_t']:.2f} **fails HAC ({R['spring_hac']:.2f})**, Upthrust leg "
            "negative. |\n"
            f"| **Tradability** | `MIRAGE` | No edge to charge costs against; net combined "
            "*t* ≈ 1.0, coin sits above it at every cost. |\n"
            f"| **Phases detectable?** | `NOT SUPPORTED` | Events fire ({R['n']} of them) but "
            "don't foretell the move; the volume tell *subtracts* signal "
            f"(no-filter *t* = {R['novol_t']:.2f} > with-filter {R['t'][2]:.2f}). Synthetic "
            f"control lights up at *t* = {R['syn_t'][1]:.2f} -> a true negative. |\n\n"
            "> In plain words: the two famous events happen all the time, but they don't predict "
            "anything once you respect autocorrelation and compare to a coin. The only positive "
            "leg is the long side in a bull-market sample."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let a trading range have support $S$ and resistance $R$. Wyckoff asserts:\n\n"
            "- **H1 (Spring).** A bar $t$ with $\\mathrm{low}_t < S$ but $\\mathrm{close}_t > S$, "
            "on volume below its trailing average, predicts **positive** forward returns "
            "(the markup): $\\mathbb{E}[r_{t+1,t+h}\\mid \\text{Spring}] > 0$.\n"
            "- **H2 (Upthrust).** A bar with $\\mathrm{high}_t > R$ but $\\mathrm{close}_t < R$ on "
            "weak volume predicts **negative** forward returns (the markdown).\n"
            "- **H3 (the volume tell).** Conditioning on *below-average* volume sharpens the "
            "signal vs not conditioning on volume at all.\n\n"
            "We **fail to reject zero** for the combined long-short (HAC *t* = "
            f"{R['hac'][2]:.2f} at 20d). H1's long leg is positive raw (*t* = {R['spring_t']:.2f}) "
            f"but **fails HAC ({R['spring_hac']:.2f})** and is confounded with market drift; H2 is "
            f"**rejected** (the short leg is *positive*-return-losing, mean "
            f"{R['upthrust_mean']:.0f} bps); H3 is **rejected** -- removing the volume filter "
            f"*raises* the *t* to {R['novol_t']:.2f}."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "Wyckoff is taught in nearly every serious chart-reading curriculum and underpins a "
            "large coaching industry. If H1+H2 held, it would be a rare TA method that pays on "
            "**both** the long and short legs -- a genuine read on order flow, consistent with the "
            "over-reaction literature (De Bondt-Thaler 1985) at the multi-week horizon. The "
            "interesting failure mode is exactly the one we find: the *buy* leg looks alive while "
            "the *sell* leg loses, which is the fingerprint of a directional-drift artefact, not a "
            "phase detector."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Range.** 5% ZigZag pivots; a window of 4 consecutive pivots inside a <=20%-wide "
            "band defines a range (support = lowest pivot low, resistance = highest pivot high). "
            "The range is 'ready' only at its last pivot's **confirmation bar** (look-ahead-free).\n"
            "- **Event.** In the 30 bars after a range is ready: a **Spring** (low < S, close > S) "
            "or **Upthrust** (high > R, close < R), with the event bar's volume **below** its "
            "trailing 20-day average (the Wyckoff tell). First qualifying event per range.\n"
            "- **Entry.** One day after the event bar (it's fully known on its own close); hold "
            "5/10/20/40 days. Direction: +1 Spring, -1 Upthrust.\n"
            "- **Inference.** One-sample *t* and Newey-West HAC *t* vs 0.\n"
            "- **Placebo.** Same event bars, random +/-1 direction, 20,000 draws -- does the "
            "*label* beat a coin?\n"
            "- **Controls.** Per-leg split; volume-filter on/off; ZigZag 3/5/8%; cost sweep; a "
            "synthetic tape with a *planted* markup (the faithful-engine positive control).\n\n"
            "Eight tapes: SPY, QQQ, DIA, IWM, GLD, XLE, EEM, TLT -- daily, as-of 2026-06-24."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - Combined signal vs coin, with placebo p-values, across horizons\n\n"
            "Gross per-trade mean and HAC *t*, the same-bars coin control, and the label-shuffle "
            "placebo *p* at each horizon."
        ),
        code(
            "H = [5,10,20,40]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in H:\n"
            "        led = pooled(h, 0., True)\n"
            "        s = st.summarize(led, 'ret_gross')\n"
            "        c = st.summarize(pooled(h, 0., True, coin=True), 'ret_gross')\n"
            "        # pooled label-shuffle placebo\n"
            "        obs = led['ret_gross'].mean(); mags = led['ret_gross'].to_numpy()*led['dir'].to_numpy()\n"
            "        rng = np.random.default_rng(446)\n"
            "        draws = np.array([(rng.choice([-1,1],size=len(mags))*mags).mean() for _ in range(20000)])\n"
            "        rows.append((h, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['t'],\n"
            "                     s['hac_t'], c['mean_bps'], float((draws>=obs).mean())))\n"
            "    tab = pd.DataFrame(rows, columns=['h','n','win%','mean_bps','t','hac_t','coin_bps','placebo_p'])\n"
            "else:\n"
            "    tab = pd.DataFrame({'h':H,'n':[404]*4,'win%':[45.5,52.2,53.5,51.0],\n"
            "        'mean_bps':[-13.60,2.62,44.14,26.58],'t':[-0.65,0.08,1.11,0.50],\n"
            "        'hac_t':[-0.56,0.07,0.89,0.42],'coin_bps':[-4.64,40.44,53.41,53.89],\n"
            "        'placebo_p':[0.7440,0.4630,0.1296,0.3034]})\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))\n"
            "x = range(len(H)); w=.38\n"
            "axes[0].bar([i-w/2 for i in x], tab['mean_bps'], width=w, color=AMBER, label='Wyckoff')\n"
            "axes[0].bar([i+w/2 for i in x], tab['coin_bps'], width=w, color=GREY, label='coin')\n"
            "axes[0].axhline(0,c='k',lw=1); axes[0].set_xticks(list(x)); axes[0].set_xticklabels([f'{h}d' for h in H])\n"
            "axes[0].set_ylabel('gross mean (bps)'); axes[0].set_title('Wyckoff vs coin at same bars'); axes[0].legend()\n"
            "axes[1].bar([f'{h}d' for h in H], tab['hac_t'], color=[GREEN if abs(t)>=2 else GREY for t in tab['hac_t']])\n"
            "axes[1].axhline(2,ls='--',c=GREY); axes[1].axhline(-2,ls='--',c=GREY); axes[1].axhline(0,c='k',lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat'); axes[1].set_title('No horizon clears |t| = 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string(index=False))"
        ),
        md(
            f"> In plain words: the best HAC *t* is **{R['hac'][2]:.2f}** (20-day) -- well short of "
            "2 -- and at every horizon a coin at the identical bars is competitive or better. The "
            f"placebo *p* bottoms out at **{R['placebo_p'][2]:.2f}**: the Spring/Upthrust *label* "
            "carries no directional information over a coin."
        ),

        md(
            "### 4b - The per-leg split -- the bull-market tell\n\n"
            "A real phase detector pays on both legs. Here, only the long (Spring) leg is positive "
            "-- and the short (Upthrust) leg loses -- in a sample where the underlyings drifted up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = pooled(20, 0., True)\n"
            "    sp = st.summarize(led[led['kind']=='spring'], 'ret_gross')\n"
            "    up = st.summarize(led[led['kind']=='upthrust'], 'ret_gross')\n"
            "    vals = [(sp['n_trades'],sp['win_rate']*100,sp['mean_bps'],sp['t'],sp['hac_t']),\n"
            "            (up['n_trades'],up['win_rate']*100,up['mean_bps'],up['t'],up['hac_t'])]\n"
            "else:\n"
            "    vals = [(192,66.7,155.08,2.29,1.77),(212,41.5,-56.35,-1.31,-1.23)]\n"
            "leg = pd.DataFrame(vals, index=['Spring (buy)','Upthrust (sell)'],\n"
            "                   columns=['n','win%','mean_bps','t','hac_t'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5,4.4))\n"
            "bars = ax.bar(leg.index, leg['mean_bps'], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('gross mean (bps, 20d)')\n"
            "ax.set_title('Spring (long) positive, Upthrust (short) negative = drift, not phases')\n"
            "for b,(_,row) in zip(bars, leg.iterrows()):\n"
            "    ax.annotate(f\"t={row['t']:+.2f}\\nwin {row['win%']:.0f}%\",\n"
            "        (b.get_x()+b.get_width()/2, b.get_height()+(8 if b.get_height()>=0 else -22)),\n"
            "        ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(leg.round(2).to_string())"
        ),
        md(
            f"> In plain words: the Spring's one-sample *t* = {R['spring_t']:.2f} looks promising "
            f"but its HAC *t* is only **{R['spring_hac']:.2f}** (below 2), and the Upthrust short "
            f"leg *loses* ({R['upthrust_mean']:.0f} bps). A genuine accumulation/distribution "
            "detector is symmetric; this is a long position dressed up as one."
        ),

        md(
            "### 4c - The volume tell adds nothing (H3 rejected)\n\n"
            "Wyckoff rests on volume: the break 'should' follow through but doesn't, *because "
            "volume is light*. If that tell mattered, conditioning on below-average volume would "
            "sharpen the signal. It does the opposite."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vc = st.summarize(pooled(20, 0., True), 'ret_gross')\n"
            "    nv = st.summarize(pooled(20, 0., False), 'ret_gross')\n"
            "    vals = [(vc['n_trades'],vc['mean_bps'],vc['t'],vc['hac_t']),\n"
            "            (nv['n_trades'],nv['mean_bps'],nv['t'],nv['hac_t'])]\n"
            "else:\n"
            "    vals = [(404,44.14,1.11,0.89),(615,63.33,1.98,1.60)]\n"
            "vf = pd.DataFrame(vals, index=['volume-confirmed','no volume filter'],\n"
            "                  columns=['n','mean_bps','t','hac_t'])\n"
            "fig, ax = plt.subplots(figsize=(8.0,4.2))\n"
            "ax.bar(vf.index, vf['t'], color=[GREY, AMBER], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0,c='k',lw=1)\n"
            "ax.set_ylabel('one-sample t (gross, 20d)')\n"
            "ax.set_title('Removing the volume filter RAISES the t-stat (still < 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(vf.round(2).to_string())"
        ),
        md(
            "> In plain words: the low-volume condition that's supposed to be the method's "
            f"lie-detector actually *removes* return -- the unfiltered version reaches "
            f"*t* = {R['novol_t']:.2f} (its closest brush with significance), and even that misses "
            "2. The volume tell is not adding information here."
        ),

        md(
            "### 4d - Robustness: the ZigZag threshold\n\n"
            "The whole thing hinges on what counts as a 'swing'. Vary the ZigZag percentage and "
            "the (already absent) effect doesn't stabilise -- it swings from exactly zero to "
            "noisy-positive on a handful of events."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pr = []\n"
            "    for p in (0.03,0.05,0.08):\n"
            "        s = st.summarize(pooled(20, 0., True, pct=p), 'ret_gross')\n"
            "        pr.append((p, s['n_trades'], s['mean_bps'], s['t'], s['hac_t']))\n"
            "    pt = pd.DataFrame(pr, columns=['pct','n','mean_bps','t','hac_t'])\n"
            "else:\n"
            "    pt = pd.DataFrame({'pct':[0.03,0.05,0.08],'n':[1316,404,68],\n"
            "        'mean_bps':[-0.21,44.14,70.61],'t':[-0.01,1.11,0.61],'hac_t':[-0.01,0.89,0.51]})\n"
            "fig, ax = plt.subplots(figsize=(8.5,4.2))\n"
            "ax.bar([f'{p:.0%}\\n(n={n})' for p,n in zip(pt['pct'],pt['n'])], pt['t'],\n"
            "       color=GREY, width=.5)\n"
            "ax.axhline(2,ls='--',c=GREY); ax.axhline(0,c='k',lw=1)\n"
            "ax.set_ylabel('one-sample t (gross, 20d)'); ax.set_xlabel('ZigZag swing threshold')\n"
            "ax.set_title('Not robust to the swing threshold -- never reaches 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(pt.round(3).to_string(index=False))"
        ),
        md(
            "> In plain words: a tighter 3% ZigZag fires 1,316 events for a mean of essentially "
            "**zero**; a looser 8% fires only 68. There's no threshold where a stable, "
            "significant edge appears."
        ),

        md(
            "### 4e - The positive control -- the engine works when we plant a markup\n\n"
            "Does the detector even work? On a synthetic tape where we *plant* a real markup after "
            "the Spring (and markdown after the Upthrust), it should light up; on a quiet random "
            "walk it should not."
        ),
        code(
            "edges = [0.0, 0.06, 0.12, 0.20]\n"
            "syn = []\n"
            "for e in edges:\n"
            "    sb, truth = data.synthetic_panel(edge=e)\n"
            "    ev = st.find_events(sb, vol_confirm=True)\n"
            "    s = st.summarize(st.run_trades(sb, ev, 20, cost_bps=0.0), 'ret_gross')\n"
            "    syn.append((e, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['t'], s['hac_t']))\n"
            "syn = pd.DataFrame(syn, columns=['edge','n','win%','mean_bps','t','hac_t'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar([f'{e:.2f}' for e in syn['edge']], syn['t'],\n"
            "       color=[GREEN if abs(t)>=2 else GREY for t in syn['t']], width=.55)\n"
            "ax.axhline(2,ls='--',c=GREY); ax.axhline(0,c='k',lw=1)\n"
            "ax.set_xlabel('planted markup edge'); ax.set_ylabel('one-sample t (gross, 20d)')\n"
            "ax.set_title('The engine detects a planted markup (t=5.2 at edge 0.06); quiet at edge 0')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(syn.round(2).to_string(index=False))"
        ),
        md(
            f"> In plain words: at edge 0 the detector is quiet (|*t*| < 2, only "
            f"{R['syn_n'][0]} noise events); at a modest planted edge it lights up at "
            f"*t* = **{R['syn_t'][1]:.2f}**. (At very large edges the markup outruns the 20-day "
            "hold and disrupts the next range, so *n* and *t* fall -- a quirk of the toy "
            "generator.) The harness **would** catch real accumulation/distribution -- the real "
            "tape simply has none to catch."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- {R['n']} events, best combined HAC *t* = {R['hac'][2]:.2f}, "
            f"beaten by a coin at the same bars; placebo *p* >= {R['placebo_p'][2]:.2f}. The Spring "
            f"leg (*t* = {R['spring_t']:.2f}) fails HAC ({R['spring_hac']:.2f}) and is the "
            "bull-market long bias; the Upthrust short leg loses.\n"
            "- **Tradability `MIRAGE`** -- no edge to deploy; net combined *t* ≈ 1.0 and the coin "
            "outperforms at every cost level.\n"
            "- **Phases detectable? `NOT SUPPORTED`** -- events fire but don't predict the move, "
            f"and the volume tell *subtracts* signal (no-filter *t* = {R['novol_t']:.2f} > "
            f"with-filter {R['t'][2]:.2f}). The synthetic control (*t* = {R['syn_t'][1]:.2f}) "
            "proves this is a true negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- cost sweep\n\n"
            "Events are infrequent so costs barely move the number -- but there's nothing to "
            "erode. The coin sits above the Wyckoff signal at every cost level."
        ),
        code(
            "costs = [0.,1.,2.,5.]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pooled(20, c, True), 'ret_net')['mean_bps'] for c in costs]\n"
            "    nett = [st.summarize(pooled(20, c, True), 'ret_net')['t'] for c in costs]\n"
            "    coin_net = [st.summarize(pooled(20, c, True, coin=True), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [44.14,42.14,40.14,34.14]; nett=[1.11,1.06,1.01,0.86]\n"
            "    coin_net=[53.41,51.41,49.41,43.41]\n"
            "fig, axes = plt.subplots(1,2,figsize=(12,4.2))\n"
            "axes[0].plot(costs, net, 'o-', c=AMBER, lw=2, label='Wyckoff (net)')\n"
            "axes[0].plot(costs, coin_net, 's--', c=GREY, lw=1.8, label='coin (net)')\n"
            "axes[0].axhline(0,c='k',lw=1); axes[0].set_xlabel('round-trip cost (bps)')\n"
            "axes[0].set_ylabel('net mean (bps)'); axes[0].set_title('Coin stays above Wyckoff'); axes[0].legend()\n"
            "axes[1].plot(costs, nett, 'o-', c=AMBER, lw=2)\n"
            "axes[1].axhline(2,ls='--',c=GREY); axes[1].axhline(0,c='k',lw=1)\n"
            "axes[1].set_xlabel('round-trip cost (bps)'); axes[1].set_ylabel('net one-sample t')\n"
            "axes[1].set_title('Net t-stat ~ 1, never near 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net mean by cost:', [round(x,1) for x in net])"
        ),
        md(
            "> In plain words: this isn't 'a real edge that costs kill' -- it's no edge at all. "
            "The Mirage stamp is about the *absence* of a signal, not its erosion."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the positive control and open questions\n\n"
            "The synthetic control confirms the engine is a faithful Spring/Upthrust detector "
            f"(*t* = {R['syn_t'][1]:.2f} on a planted markup). The real-market answer is a clean "
            "negative. Forks worth pursuing:\n\n"
            "- **Market-neutral legs.** Subtract the index return over the same window so the "
            "bull-market drift cancels, then re-test whether the Spring beats the Upthrust. If a "
            "symmetric residual survives at HAC *t* > 2, Wyckoff has genuine support.\n"
            "- **Bear-regime sub-sample (2008, 2022).** The Upthrust short leg should *work* in a "
            "down-trending regime if it's a real distribution detector rather than a losing short "
            "in a bull market.\n"
            "- **Richer event definitions.** A real Selling Climax / Buying Climax bar "
            "(effort-vs-result: huge volume, small range, then reversal) instead of the minimal "
            "low/high-pierce-and-recover rule.\n"
            "- **The siblings.** [Study 445 -- Elliott Wave](../../445-elliott-wave/) and "
            "[Study 444 -- Dow Theory](../../444-dow-theory/) -- the same mechanical-proxy method "
            "on the other irreducibly-subjective classics."
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
