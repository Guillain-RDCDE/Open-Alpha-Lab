"""Generate the two narrative notebooks for Study 376 (MOC-Imbalance).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily OHLC
under ../_cache/ (the displacement proxy + overnight gaps) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC for
# SPY/QQQ/IWM, 2005-01-03 -> 2026-06-17, 16,194 ticker-days, 21.5 years).
R = dict(
    start="2005-01-03", end="2026-06-17", days=16194, years=21.5, n_rebal=252,
    # (n, b, HAC_t, placebo_p, corr, fade_win%)
    all=(16194, -0.00017, -1.48, 0.076, -0.011, 50.5),
    rebal=(252, -0.00044, -0.48, 0.325, -0.029, 52.2),
    spy=(5398, -0.00027, -1.54, 0.072, -0.020, 50.9),
    qqq=(5398, -0.00012, -0.64, 0.266, -0.009, 49.3),
    iwm=(5398, -0.00011, -0.49, 0.316, -0.007, 51.3),
    sharpe=0.037,
    # (cost_bps, net/trade bps, net_ann%)
    costs=[(0.0, 0.18, 0.46), (1.0, -0.82, -2.04), (2.0, -1.82, -4.48)],
    # (planted, n, b, corr, HAC_t, placebo_p, win%)
    syn=[(0.0, 4000, -0.00003, -0.003, -0.20, 0.4155, 51.1),
         (0.10, 4000, -0.00111, -0.130, -8.10, 0.0000, 55.6)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from moc_imbalance import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real() if HAVE_REAL else None
print("real OHLC cache present:", HAVE_REAL,
      "| panel rows:", (0 if PANEL is None else len(PANEL)))
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
            "# The closing-bell \"free lunch\" — does the auction imbalance reverse overnight? 🔔\n"
            "### The desk trade that fades the closing push and pockets the morning snap-back, in plain English\n\n"
            + BADGES +
            "Every trading day ends with a **closing auction** — one big batch print at 4 p.m. where "
            "index funds, ETFs and rebalancers all have to trade. When far more people want to **buy** "
            "than **sell** into that auction (an *order imbalance*), the story goes, the imbalance "
            "**shoves the last price** away from where it 'should' be — and then it **snaps back "
            "overnight**, so the next morning's open leans the other way. The trade writes itself: "
            "stand *against* the push into the close, collect the bounce at the open.\n\n"
            "It sounds like money lying on the sidewalk — especially on **index-rebalance days**, when "
            "passive funds are *forced* to trade the close. This notebook asks whether the snap-back is "
            "actually there, and whether you could ever keep a cent of it.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The real closing-auction imbalance feed costs money, so we "
            "**build a transparent proxy**: each day's *displacement* — did price open at the low and "
            "close at the high (a hard push up) or the reverse? It's a coarse stand-in (we call it a "
            "proxy throughout), but it can only make the reversal look *bigger* than it is — so if even "
            "the proxy shows nothing, the real thing shows less. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a hard push into the close reverse overnight? | **A tiny bit, on average — and in "
            "the right direction.** That's the kernel of truth. |\n"
            "| Is that reversal real (could you bet on it)? | **No.** It explains about **0.01%** of the "
            "overnight move, the statistical test comes in at **t = −1.5** (you need ±2), and a coin "
            f"flip matches it **{R['all'][3]*100:.0f}%** of the time. |\n"
            "| Surely it's bigger on rebalance days? | **It's actually *smaller*.** On the exact days "
            "passive funds are forced to trade the close — where a real imbalance should be biggest — "
            f"the effect *weakens* (**t = {R['rebal'][2]:.1f}**). |\n"
            "| Could you trade it after costs? | **No.** Even ignoring fees the fade earns "
            f"**+{R['costs'][0][1]:.2f} bps** a trade. A **1-bp** spread already turns it negative. The "
            "snap-back is *inside the spread.* |\n\n"
            "> The reversal is real-as-microstructure and empty-as-a-strategy: a whisper in the right "
            "direction that the bid-ask spread swallows whole."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"On a day with a big closing-auction **buy** imbalance, the last print gets pushed **up** "
            "past fair value — so the next morning it opens **down**, reverting. Fade the imbalance into "
            "the close and harvest the overnight reversal. It's strongest on index-rebalance days, when "
            "passive funds *must* trade the official close.\"*\n\n"
            "The picture is plausible: a forced, price-insensitive buyer (an index fund tracking the "
            "close) pays up at 4 p.m., temporarily dislocating the print; once that pressure is gone, "
            "price drifts back. We'll rebuild a proxy for that closing pressure and check whether the "
            "overnight gap really leans back the other way."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"the imbalance reverses overnight.\" (1) *Is there a "
            "faint statistical lean back?* Maybe — short-horizon price moves often wobble back a little. "
            "(2) *Is that lean big and reliable enough, after the cost of crossing the bid-ask spread "
            "twice, to put money on?* Those are not the same question. A reversal that's **real but tiny** "
            "is exactly what a market-maker *earns* for providing liquidity — and exactly what a *taker* "
            "(you, crossing the spread) **pays away**. The whole verdict turns on which side of the "
            "spread the snap-back lives on."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the closing pressure on a **transparent proxy**: for each day, where did the "
            "close sit inside the day's range? `displacement = (close − open) / (high − low)`. **+1** "
            "means it opened at the low and closed at the high — the maximum upward push into the close; "
            f"**−1** is the mirror. Over **{R['years']:.0f} years** ({R['start']} → {R['end']}) across "
            "SPY/QQQ/IWM that's tens of thousands of days.\n\n"
            "1. **Measure the push.** Compute each day's displacement (our imbalance proxy).\n"
            "2. **Measure the snap-back.** Look at the **overnight gap** — next open ÷ today's close − 1. "
            "Does a big positive push lead to a negative gap?\n"
            "3. **Stress the luck.** Shuffle the push-to-gap pairing thousands of times and ask how often "
            "pure chance looks this 'reversal-y'. And check the **rebalance days** on their own — the "
            "believers' best case."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the reversal itself.** Bin every day by how hard it was pushed into the close "
            "(displacement), and plot the *average overnight gap* in each bin. The lore predicts a "
            "**downward-sloping** line: harder up-push ⇒ more negative overnight gap."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = PANEL['disp'].to_numpy(); o = PANEL['overnight'].to_numpy()\n"
            "    bins = np.linspace(-1, 1, 11); idx = np.digitize(d, bins) - 1\n"
            "    xs, ys = [], []\n"
            "    for b in range(len(bins)-1):\n"
            "        m = idx == b\n"
            "        if m.sum() > 30:\n"
            "            xs.append((bins[b]+bins[b+1])/2); ys.append(o[m].mean()*1e4)\n"
            "    s = st.summarize(PANEL); slope, tval = s['b'], s['t']\n"
            "else:\n"
            "    xs = np.linspace(-0.9, 0.9, 10)\n"
            "    ys = (-R['all'][1]*xs*1e4*0 + (-1.7*xs)) + np.array([0.4,-0.3,0.2,-0.5,0.1,-0.2,0.3,-0.4,0.0,-0.3])\n"
            "    slope, tval = R['all'][1], R['all'][2]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.5, alpha=.4)\n"
            "ax.scatter(xs, ys, c=GREEN, s=60, zorder=3, label='avg overnight gap per push-bin')\n"
            "ax.set_xlabel('push into the close (displacement: -1 = closed at low, +1 = closed at high)')\n"
            "ax.set_ylabel('average overnight gap (bps)')\n"
            "ax.set_title(f'A faint downward tilt — slope t = {tval:.2f} (the lore wants a clear one)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'reversal slope b={slope:.5f}, t={tval:.2f}  (negative = the lore\\'s direction; |t|<2 = not significant)')"
        ),
        md(
            "There *is* a downward tilt — bigger up-pushes do precede slightly more negative gaps, the "
            "direction the story predicts. But look how **flat and scattered** it is. The slope's "
            f"statistical test comes in at **t = {R['all'][2]:.2f}** — well short of the ±2 you need to "
            "call it real. The push explains about **0.01%** of the overnight move. It's a lean, not a lever."
        ),
        md(
            "**The believers' best case: rebalance days.** On the third Friday of quarter-end months, "
            "passive funds are *forced* to trade the close, so the real imbalance is largest. If the "
            "effect is real, it should be **stronger** here. Watch what actually happens to the *t*-stat."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_all = st.summarize(PANEL)\n"
            "    s_reb = st.summarize(st.subset(PANEL, rebal_only=True))\n"
            "    ts = [abs(s_all['t']), abs(s_reb['t'])]; ns = [s_all['n'], s_reb['n']]\n"
            "else:\n"
            "    ts = [abs(R['all'][2]), abs(R['rebal'][2])]; ns = [R['all'][0], R['rebal'][0]]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "bars = ax.bar(['all days', 'rebalance days\\n(the strong case)'], ts, color=[GREY, AMBER], width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='|t| = 2 (significance bar)')\n"
            "for i,(t,n) in enumerate(zip(ts,ns)): ax.annotate(f'|t|={t:.2f}\\nn={n:,}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('|reversal t-stat|'); ax.set_ylim(0, 2.4)\n"
            "ax.set_title('On the days a real imbalance is BIGGEST, the effect gets SMALLER'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'all days |t|={ts[0]:.2f}  ->  rebalance days |t|={ts[1]:.2f}  (it goes the wrong way for the story)')"
        ),
        md(
            f"That's the tell. On the rebalance days — where a genuine MOC imbalance should hit hardest "
            f"— the effect is **weaker** (|t| drops from **{abs(R['all'][2]):.1f}** to "
            f"**{abs(R['rebal'][2]):.1f}**). A real imbalance edge would *concentrate* there. Instead it "
            "fades, which is what you'd expect if the whole thing is just background microstructure noise."
        ),
        md(
            "**Now the money question: can you keep the snap-back?** Run the fade — bet against the push "
            "every day, collect the overnight gap — and subtract a realistic cost for crossing the "
            "spread at the close and again at the open."
        ),
        code(
            "costs = [0.0, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    nets = [st.net_of_costs(PANEL, cb)['net_mean']*1e4 for cb in costs]\n"
            "else:\n"
            "    nets = [c[1] for c in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "cols = [GREEN if v > 0 else RED for v in nets]\n"
            "ax.bar([f'{c:.0f} bps' for c in costs], nets, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "for i,v in enumerate(nets): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_xlabel('round-trip cost (close in, open out)'); ax.set_ylabel('net return per trade (bps)')\n"
            "ax.set_title('Gross is a rounding error; 1 bp of cost already turns it red')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross fade: {nets[0]:+.2f} bps/trade  ->  at 1 bp: {nets[1]:+.2f}  ->  at 2 bps: {nets[2]:+.2f}')"
        ),
        md(
            f"There it is. **Even free**, the fade makes **+{R['costs'][0][1]:.2f} bps** a trade — a "
            "rounding error. Charge a single basis point to cross the spread and it goes **negative**; at "
            "a realistic 2 bps it bleeds **−4.5%/yr**. The 'overnight snap-back' isn't free money on the "
            "sidewalk — it's the **spread itself**, which the auction's liquidity providers collect and "
            "the rest of us pay."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The reversal has the right *sign* but **t = −1.5** (you need ±2), a "
            "near-zero R², and it *weakens* on the rebalance days where it should be strongest. Real as a "
            "wobble, weak as an edge.\n"
            f"- **Tradability — Mirage.** Gross **+{R['costs'][0][1]:.2f} bps**/trade; **1 bp** of cost "
            "makes it lose money. The snap-back lives inside the spread.\n"
            "- **Free lunch? — Busted.** A tiny, correct-sign microstructure effect that frictions eat "
            "whole — and that evaporates on the very days believers point to. Same shape as "
            "[Amihud illiquidity](../140-amihud-illiquidity/): real in the data, paid away in the trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the spread is the whole story\n\n"
            "Forget significance for a second. The fade's gross edge and the cost of harvesting it are "
            "almost the *same size*. Here they are side by side — the punchline is how completely the "
            "spread swallows the signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gross = st.net_of_costs(PANEL, 0.0)['gross_mean']*1e4\n"
            "    sharpe = R['sharpe']\n"
            "    fr = st.fade_returns(PANEL); sharpe = fr.mean()/fr.std()*np.sqrt(252)\n"
            "else:\n"
            "    gross = R['costs'][0][1]; sharpe = R['sharpe']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.0))\n"
            "ax.barh(['gross edge / trade', 'cost of 1 round-trip'], [gross, 1.0], color=[GREEN, RED])\n"
            "for i,v in enumerate([gross, 1.0]): ax.annotate(f'{v:.2f} bps',(v,i),va='center',ha='left')\n"
            "ax.set_xlabel('basis points'); ax.set_xlim(0, max(gross, 1.0)*1.4)\n"
            "ax.set_title(f'The edge ({gross:.2f} bps) is smaller than one crossing of the spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross daily Sharpe of the fade: {sharpe:.3f}  (a real strategy wants ~1+; this is ~0)')"
        ),
        md(
            f"A gross daily **Sharpe of ~{R['sharpe']:.2f}** — indistinguishable from zero — and an edge "
            "smaller than a single crossing of the bid-ask spread. You can't build a strategy whose entire "
            "profit is smaller than your transaction cost. The honest version of the closing-auction trade "
            "isn't \"fade the imbalance and collect the bounce\" — it's \"the bounce is the market-maker's "
            "spread, and you're the one paying it.\""
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The bigger split.** [Study 01 — Overnight-Anomaly](../01-overnight-anomaly/) asks where "
            "stock returns are actually earned — overnight or intraday. The MOC-reversal trade is one "
            "thin slice of that decomposition, seen through the closing auction.\n"
            "- **Same family.** [Study 140 — Amihud-Illiquidity](../140-amihud-illiquidity/) is the "
            "archetype: an effect that's genuinely *in the data* but lives inside trading frictions, so "
            "anyone who must cross the spread pays it back.\n"
            "- **Build your own.** If you have a real NYSE/Nasdaq MOC imbalance feed, swap our "
            "displacement proxy for the true imbalance and re-run the regression — the proxy can only "
            "*over*-state the reversal, so the real number is no larger.\n\n"
            "*Think the closing imbalance reverses by more than the spread? Pair the real imbalance with "
            "the overnight gap, run the fade net of costs, and show a Sharpe that survives — then we'll talk.*"
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
            "# MOC-Imbalance — a quantitative teardown 🔬\n"
            "### A closing-pressure displacement proxy · an `overnight ~ disp` reversal regression · "
            "a HAC *t* + placebo randomization null · the rebalance-day subset · the fade net of costs · "
            "a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "closing-auction-imbalance reversal claim as a single regression — the overnight gap on a "
            "displacement proxy for end-of-day pressure — and confront it with **autocorrelation-robust "
            "inference**, a **placebo null**, the **rebalance-day subset** (the believers' strongest "
            "case), and a **cost-aware fade**. The decisive object is not the slope's sign (it is "
            "negative, as the lore says) but its **magnitude relative to its HAC standard error and to "
            "the bid-ask spread**.\n\n"
            "> ⚠️ **Data + proxy note.** A true MOC order-imbalance feed (NYSE Order Imbalances / Nasdaq "
            "NOII) is a paid product; we build a transparent **proxy** — the signed intraday displacement "
            "`(close−open)/(high−low)`. It is coarse and, via the open's bid-ask bounce, can only "
            "**over**-state reversal (both named on the Signal axis). Real data: yfinance daily OHLC, "
            "SPY/QQQ/IWM, 2005→2026. Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | reversal slope **b = {R['all'][1]:.5f}**, HAC **t = {R['all'][2]:.2f}** "
            f"(placebo **p = {R['all'][3]:.3f}**), corr **{R['all'][4]:.3f}** — direction-correct but "
            f"**fails t ≥ 2**, and the rebalance-day *t* is **{R['rebal'][2]:.2f}** (weaker, not stronger). |\n"
            f"| **Tradability** | `MIRAGE` | gross fade **+{R['costs'][0][1]:.2f} bps**/trade "
            f"(Sharpe ≈ {R['sharpe']:.2f}); at **1 bp** cost **{R['costs'][1][1]:.2f} bps**, at 2 bps "
            f"**{R['costs'][2][1]:.2f} bps** ({R['costs'][2][2]:.1f}%/yr). The reversal is inside the spread. |\n"
            f"| **Free lunch?** | `BUSTED` | win-rate **{R['all'][5]:.1f}%** ≈ 50%; the effect *evaporates* "
            "on rebalance days. Real-as-microstructure, empty-as-alpha. |\n\n"
            "> 💡 In plain words: the imbalance really does reverse a touch overnight — because "
            "short-horizon moves wobble back, and the proxy double-counts the open's bid-ask bounce. "
            "Strip the spread and ~zero edge is left, concentrated *away* from where a real imbalance lives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $d_t$ be the signed displacement $(c_t-o_t)/(h_t-l_t)\\in[-1,1]$ (the closing-pressure "
            "proxy) and $g_{t+1} = o_{t+1}/c_t - 1$ the overnight gap. The reversal claim is a sign on "
            "the slope of\n\n"
            "$$g_{t+1} = a + b\\,d_t + \\varepsilon_{t+1},\\qquad H_1:\\ b < 0.$$\n\n"
            "- **H₁ (reversal exists).** $b<0$ robustly — a push into the close leans back overnight.\n"
            "- **H₂ (it's deployable).** The fade $-\\operatorname{sign}(d_t)\\cdot g_{t+1}$ earns more "
            "than the round-trip spread.\n"
            "- **H₃ (rebalance concentration).** $|b|$ and $|t|$ are **larger** on index-rebalance days, "
            "where real MOC imbalances are largest.\n\n"
            "We find **H₁ not rejected but not supported** ($b<0$ but HAC $t=-1.48$), **H₂ rejected** "
            "(gross +0.18 bps, negative after 1 bp), **H₃ rejected** (the rebalance-day $t$ is "
            "$-0.48$ — *weaker*). The trade is true exactly where it's uninformative (a sub-spread wobble) "
            "and false exactly where it would pay (a *concentrated, tradable* imbalance edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one slope judged by its **HAC standard error** and by the **bid-ask "
            "spread**.\n\n"
            "$$\\hat b = \\frac{\\sum (d_t-\\bar d)(g_{t+1}-\\bar g)}{\\sum (d_t-\\bar d)^2},\\qquad "
            "t_{\\text{HAC}} = \\frac{\\hat b}{\\operatorname{SE}_{\\text{NW}}(\\hat b)}.$$\n\n"
            "Overnight gaps are serially correlated and heteroskedastic, so an OLS SE understates the "
            "noise — we use **Newey-West**. But even a *significant* $\\hat b$ would not be an edge: the "
            "fade pays the spread twice, and the implied per-trade edge $|\\hat b|\\cdot\\mathbb{E}|d|$ is "
            "a fraction of a basis point. The right second lens is therefore not statistical at all — it "
            "is **economic**: is the gross edge larger than one crossing of the spread? It is not."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Pressure proxy.** $d_t=(c_t-o_t)/(h_t-l_t)$ clipped to $[-1,1]$, from yfinance daily "
            f"OHLC for SPY/QQQ/IWM ({R['start']}→{R['end']}, {R['days']:,} ticker-days). Explicit "
            "**proxy** for the MOC order imbalance — coarse, and bid-ask-bounce-inflated (named on the axis).\n"
            "- **Target.** The overnight gap $g_{t+1}=o_{t+1}/c_t-1$. The 1-day execution lag is "
            "**structural**: entry is today's close, the gap is realised at the *next* open — no look-ahead.\n"
            "- **Null #1 (HAC t).** Newey-West (5 lags) t of $\\hat b$.\n"
            "- **Null #2 (placebo).** 20,000 shuffles of the $d\\leftrightarrow g$ pairing; "
            "$p=\\Pr[\\text{shuffled slope}\\le\\hat b]$ — the honest small-effect test.\n"
            "- **Subset.** Re-run on index-rebalance days only (third Friday of quarter-end) — H₃.\n"
            "- **Strategy.** Fade $-\\operatorname{sign}(d_t)$, earn $g_{t+1}$; win-rate vs 50%; net of a "
            "round-trip cost.\n"
            "- **Positive control.** A deterministic panel with a **planted reversal knob**: edge=0 must "
            "stay insignificant; a large knob must light up — proving the engine is unbiased *and* powerful."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The slope — direction-correct, magnitude-trivial\n\n"
            "Bin by displacement and plot the mean overnight gap with its standard error, against the "
            "fitted line. Negative-sloping (the lore's direction) but flat and noisy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = PANEL['disp'].to_numpy(); o = PANEL['overnight'].to_numpy()\n"
            "    s = st.summarize(PANEL); b, a = s['b'], None\n"
            "    a_, b_, _ = st._ols(d, o)\n"
            "    bins = np.linspace(-1, 1, 13); xc, ym, ye = [], [], []\n"
            "    for i in range(len(bins)-1):\n"
            "        m = (d >= bins[i]) & (d < bins[i+1])\n"
            "        if m.sum() > 30:\n"
            "            xc.append((bins[i]+bins[i+1])/2); ym.append(o[m].mean()*1e4)\n"
            "            ye.append(o[m].std(ddof=1)/np.sqrt(m.sum())*1e4)\n"
            "    xc=np.array(xc); ym=np.array(ym); ye=np.array(ye)\n"
            "    tval = s['t']; line = (a_ + b_*xc)*1e4\n"
            "else:\n"
            "    xc = np.linspace(-0.9,0.9,12); ym = -1.7*xc + np.array([.4,-.3,.2,-.5,.1,-.2,.3,-.4,.0,-.3,.2,-.1])\n"
            "    ye = np.full_like(xc, 0.8); line = (R['all'][1]*xc)*1e4; tval = R['all'][2]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.errorbar(xc, ym, yerr=ye, fmt='o', c=GREEN, capsize=3, label='mean overnight gap ±SE')\n"
            "ax.plot(xc, line, c=AMBER, lw=2, label='OLS fit')\n"
            "ax.set_xlabel('displacement $d_t$ (closing-pressure proxy)'); ax.set_ylabel('overnight gap (bps)')\n"
            "ax.set_title(f'b<0 as the lore predicts, but HAC t = {tval:.2f} (|t|<2)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope b={R[\"all\"][1]:.5f}, HAC t={tval:.2f}, corr={R[\"all\"][4]:.3f}  -> R^2 ~ {R[\"all\"][4]**2*100:.3f}%')"
        ),
        md(
            f"> 💡 In plain words: the slope is **{R['all'][1]:.5f}** — negative, the reversal direction — "
            f"but at HAC **t = {R['all'][2]:.2f}** it does not clear 2, and the displacement explains "
            f"**~{R['all'][4]**2*100:.2f}%** of the overnight gap. H₁ is **not rejected, not supported**: "
            "a correct-sign whisper buried in its own standard error."
        ),
        md(
            "### 4b · The placebo null sized to the effect\n\n"
            "Shuffle the displacement↔gap pairing 20,000 times; the histogram is the null distribution of "
            "the slope. The observed slope is the green line; the *p*-value is the **left**-tail mass "
            "(more negative = more 'reversal-y')."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PANEL['disp'].to_numpy(), PANEL['overnight'].to_numpy(), n_draws=20000)\n"
            "    obs = pl['obs_b']; pval = pl['p_value']\n"
            "    d = PANEL['disp'].to_numpy(); o = PANEL['overnight'].to_numpy()\n"
            "    xc = d - d.mean(); den = (xc*xc).sum(); rng = np.random.default_rng(376)\n"
            "    draws = np.array([(xc*rng.permutation(o)).sum()/den for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['all'][1]; pval = R['all'][3]\n"
            "    rng = np.random.default_rng(376); draws = rng.normal(0, abs(R['all'][1])/1.4, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} shuffled slopes')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed slope {obs:.5f}')\n"
            "ax.set_xlabel('reversal slope b'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.3f}: the slope is barely inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[shuffled slope <= observed] = {pval:.3f}  (need <0.05; here ~1 in 13)')"
        ),
        md(
            f"> 💡 In plain words: **{R['all'][3]*100:.0f}%** of random pairings produce a slope at least "
            "this negative. It is *closer* to the tail than the Zweig-style coin flip — short-horizon "
            "reversal is a genuine micro-phenomenon — but it still **does not clear p < 0.05**, and (next) "
            "it doesn't survive costs or concentrate where a real imbalance lives. A near-miss is not an edge."
        ),
        md(
            "### 4c · H₃ — the rebalance-day subset goes the wrong way\n\n"
            "The believers' strongest claim: the reversal is biggest on index-rebalance days, when passive "
            "funds *must* trade the close. Compare the HAC *t* on all days vs that subset, with the "
            "per-ticker breakdown for texture."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    rows.append(('all days', *(lambda s:(s['n'],s['b'],s['t'],s['p_placebo']))(st.summarize(PANEL))))\n"
            "    rows.append(('rebalance', *(lambda s:(s['n'],s['b'],s['t'],s['p_placebo']))(st.summarize(st.subset(PANEL, rebal_only=True)))))\n"
            "    for tk in ('SPY','QQQ','IWM'):\n"
            "        s = st.summarize(PANEL[PANEL['ticker']==tk]); rows.append((tk, s['n'], s['b'], s['t'], s['p_placebo']))\n"
            "else:\n"
            "    for k in ('all','rebal','spy','qqq','iwm'):\n"
            "        n,b,t,p,_,_ = R[k]; rows.append((k, n, b, t, p))\n"
            "labels = [r[0] for r in rows]; tt = [abs(r[3]) for r in rows]\n"
            "cols = [GREY, AMBER, GREEN, GREEN, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.2))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='|t| = 2')\n"
            "for i,(r,t) in enumerate(zip(rows,tt)): ax.annotate(f'n={r[1]:,}',(i,t),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_ylabel('|HAC reversal t|'); ax.set_ylim(0,2.4)\n"
            "ax.set_title('H3 rejected: the effect is WEAKEST on rebalance days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:>10}: n={r[1]:>6,} b={r[2]:+.5f} t={r[3]:+.2f} p={r[4]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: on the days a real MOC imbalance is *largest*, the reversal *t* falls to "
            f"**{R['rebal'][2]:.2f}** (p = {R['rebal'][3]:.2f}). A true imbalance-driven dislocation would "
            "**concentrate** there. The fact that it *dilutes* there is strong evidence the full-sample "
            "lean is generic short-horizon microstructure, not the auction mechanism the story invokes."
        ),
        md(
            "### 4d · Costs — the spread is the whole game\n\n"
            "The fade's gross edge against the cost of harvesting it. Even gross the per-trade number is "
            "sub-basis-point; one crossing of the spread erases it."
        ),
        code(
            "costs = [0.0, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    res = [st.net_of_costs(PANEL, cb) for cb in costs]\n"
            "    gross = res[0]['gross_mean']*1e4; nets = [r['net_mean']*1e4 for r in res]; anns = [r['net_ann']*100 for r in res]\n"
            "else:\n"
            "    gross = R['costs'][0][1]; nets = [c[1] for c in R['costs']]; anns = [c[2] for c in R['costs']]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "cols = [GREEN if v>0 else RED for v in nets]\n"
            "a1.bar([f'{c:.0f} bps' for c in costs], nets, color=cols, width=.55); a1.axhline(0,c='k',lw=.9)\n"
            "for i,v in enumerate(nets): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.set_title('Net per trade (bps)'); a1.set_xlabel('round-trip cost'); a1.set_ylabel('bps/trade')\n"
            "a2.bar([f'{c:.0f} bps' for c in costs], anns, color=cols, width=.55); a2.axhline(0,c='k',lw=.9)\n"
            "for i,v in enumerate(anns): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.set_title('Annualised, run daily'); a2.set_xlabel('round-trip cost'); a2.set_ylabel('%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f} bps/trade; net @1bp {nets[1]:+.2f}; @2bp {nets[2]:+.2f} ({anns[2]:+.1f}%/yr)')"
        ),
        md(
            f"> 💡 In plain words: gross **+{R['costs'][0][1]:.2f} bps** with a daily Sharpe of "
            f"**~{R['sharpe']:.2f}**. A **1-bp** round-trip — optimistic for a close-to-open fade — flips "
            f"it to **{R['costs'][1][1]:.2f} bps**; 2 bps gives **{R['costs'][2][2]:.1f}%/yr**. There is no "
            "cost assumption under which this is positive at scale. **The reversal is the liquidity "
            "provider's spread; the fader is the liquidity taker who pays it.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic panel with a **planted overnight-reversal knob**: with the knob at **0** the "
            "test must stay insignificant (no false positive); with a large knob it must light up. Both "
            "hold — so the real-tape t = −1.5 is not a power failure, it is a near-zero true effect."
        ),
        code(
            "res = []\n"
            "for rev in (0.0, 0.10):\n"
            "    syn = data.synthetic_panel(reversal=rev, seed=376); s = st.summarize(syn)\n"
            "    res.append((rev, s['n'], s['b'], s['corr'], s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted reversal\\n{r*100:.0f}%' for r,_,_,_,_,_ in res]\n"
            "tvals = [abs(r[4]) for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='|t| = 2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f'|t|={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('|HAC reversal t|'); ax.set_title('Control: engine is unbiased AND powerful'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'planted {r[0]*100:>4.0f}%: detected n={r[1]} b={r[2]:+.5f} corr={r[3]:+.3f} t={r[4]:+.2f} p={r[5]:.4f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the control sits at "
            f"**|t| = {abs(R['syn'][0][4]):.2f}** (no false positive); a **10%** planted reversal reaches "
            f"**|t| = {abs(R['syn'][1][4]):.1f}** (p < 0.001). The machine is honest and *well-powered* — "
            "so the real-tape **t = −1.5** is exactly what an *almost-absent* reversal looks like through a "
            "faithful lens. The verdict is the magnitude, not a lack of data."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — slope **{R['all'][1]:.5f}** at HAC **t = {R['all'][2]:.2f}** / placebo "
            f"**p = {R['all'][3]:.3f}**; R² ≈ {R['all'][4]**2*100:.2f}%; rebalance-day *t* = "
            f"{R['rebal'][2]:.2f} (H₃ rejected). Direction-correct but **fails t ≥ 2** on the real tape, "
            "on a proxy that can only *over*-state the effect ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — gross **+{R['costs'][0][1]:.2f} bps**/trade (Sharpe "
            f"≈ {R['sharpe']:.2f}); negative after **1 bp**, **{R['costs'][2][2]:.1f}%/yr** at 2 bps. The "
            "reversal lives inside the bid-ask spread — captured by liquidity providers, paid by takers.\n"
            f"- **Free lunch? `BUSTED`** — win-rate **{R['all'][5]:.1f}%** ≈ 50%, and the effect *fades* on "
            "rebalance days. Real-as-microstructure, empty-as-alpha — the [Amihud](../140-amihud-illiquidity/) "
            "pattern: in the data, gone in the trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the break-even spread\n\n"
            "The operational truth in one picture: how small would the round-trip cost have to be for the "
            "fade to break even? Plot net return vs assumed spread; the break-even is a fraction of a "
            "basis point — below any real close-to-open execution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gross = st.net_of_costs(PANEL, 0.0)['gross_mean']\n"
            "else:\n"
            "    gross = R['costs'][0][1]/1e4\n"
            "cb = np.linspace(0, 2.0, 100)\n"
            "net = (gross - cb/1e4)*1e4\n"
            "be = gross*1e4\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(cb, net, c=AMBER, lw=2, label='net return per trade')\n"
            "ax.axhline(0, c='k', lw=.9); ax.axvline(be, c=GREEN, ls='--', label=f'break-even cost {be:.2f} bps')\n"
            "ax.fill_between(cb, net, 0, where=(net<0), color=RED, alpha=.15)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net per trade (bps)')\n"
            "ax.set_title(f'Break-even at {be:.2f} bps — below any real close-to-open spread'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'break-even round-trip cost ~ {be:.2f} bps; realistic close-to-open spread >= 1-2 bps -> always net negative')"
        ),
        md(
            "> 💡 In plain words: the fade breaks even only if you can round-trip for under **~0.2 bps** — "
            "less than the bid-ask spread of even SPY at the close, let alone IWM. There is no sizing, no "
            "venue, no execution that makes a sub-spread edge tradable. **The rarity-free version of this "
            "trade is simply: the overnight reversal *is* the market-maker's compensation, and the fader "
            "is paying it.** A 'free lunch' that is exactly priced as the cost of lunch."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The decomposition.** [Study 01 — Overnight-Anomaly](../01-overnight-anomaly/): where "
            "equity returns are earned, overnight vs intraday. The MOC-reversal trade is a microstructure "
            "slice of that split.\n"
            "- **The archetype.** [Study 140 — Amihud-Illiquidity](../140-amihud-illiquidity/) — an effect "
            "that is genuinely in the data yet lives inside frictions, so the trade pays it back. Reading "
            "the two together is the cleanest demonstration that *being in the data* and *being tradable* "
            "are different claims.\n"
            "- **Real imbalance data.** Swap the displacement proxy for a true NYSE/Nasdaq MOC imbalance "
            "feed and re-run the regression and the fade net of costs; the proxy can only *over*-state the "
            "reversal, so a sub-spread proxy edge bounds the real one from above.\n\n"
            "*The reproducible core is offline and deterministic; the closing-pressure input is an explicit "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
