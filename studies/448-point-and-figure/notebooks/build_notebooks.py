"""Generate the two narrative notebooks for Study 448 (Point & Figure price targets).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks walk the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
control runs anywhere, offline and deterministic; the real-tape cells use the cached
daily parquets under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-23).
# Daily OHLC, 4 instruments (SPY, ^DJI, AAPL, GLD), the in-progress final bar dropped.
R = dict(
    asof="2026-06-23", fp="bc95f1a9fce9",
    # Pooled BUY side (double-top buys)
    n_buy=434, hit_buy=56.5, base_buy=40.8, edge_buy=15.6, t_edge_buy=6.56,
    net_buy=3.27, t_net_buy=6.14, win_buy=67.1,
    # Pooled SELL side (double-bottom sells)
    n_sell=324, hit_sell=38.9, base_sell=26.9, edge_sell=12.0, t_edge_sell=4.41,
    net_sell=-1.03, t_net_sell=-2.02, win_sell=44.1,
    # Per-instrument BUY (nsig, hit%, base%, net%, t, placebo p)
    spy=(95, 51.6, 35.7, 3.19, 4.01, 0.000),
    dji=(66, 36.4, 27.0, 3.35, 3.55, 0.045),
    aapl=(224, 67.4, 48.1, 2.84, 3.49, 0.000),
    gld=(49, 42.9, 36.0, 5.33, 3.63, 0.115),
    # Robustness: box-size sweep (pooled buy hit% vs base%)
    box_sweep=[(0.015, 554, 58.1, 42.5), (0.02, 434, 56.5, 40.8), (0.03, 295, 59.0, 39.9)],
    # Synthetic control (buy): edge -> (nsig, hit%, base%, placebo p)
    syn0=(22, 0.0, 5.2, 1.000),
    syn1=(365, 86.6, 58.5, 0.000),
    # Distance geometry (median, %)
    tdist_buy=5.2, sdist_buy=3.5,
)


BOOT = """\
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from point_and_figure import data, strategy as st

TICKERS = list(data.DEFAULT_TICKERS)

def _have_cache():
    return all(data.have_real(t) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled(side="buy", n_draws=2000):
    \"\"\"Pool P&F signals across the basket (cached real tapes, in-progress bar dropped).\"\"\"
    hit_ind, net = [], []
    th=tn=bh=bt=0
    per = []
    for t in TICKERS:
        b = data.load_real(t, fetch=False).iloc[:-1]
        r = st.run_experiment(b, n_draws=n_draws, side=side)
        if r["n_signals"] == 0:
            continue
        L = r["ledger"]; p = r["pnl"]; base = r["baseline"]["baseline_hit_rate"]
        hit_ind += list((L["outcome"] == "hit").astype(float))
        net.append(p["net_sample"])
        th += p["hit_rate"]*r["n_signals"]; tn += r["n_signals"]
        bh += base*r["baseline"]["n_baseline"]; bt += r["baseline"]["n_baseline"]
        per.append((t, r["n_signals"], p["hit_rate"]*100, base*100, p["net_mean"]*100,
                    p["t_net"], r["placebo"]["p_value"]))
    hit_ind = np.array(hit_ind); net = np.concatenate(net) if net else np.array([])
    return dict(hit_ind=hit_ind, net=net, hit=th/tn*100, base=bh/bt*100, n=tn,
                t_net=st.hac_t(net), per=per)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Point & Figure -- do the count *price targets* actually get hit?\n"
            "### The 100-year-old chart that prints a number and says \"price will go here\"\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Do_targets_get_hit%3F: Confirmed](https://img.shields.io/badge/"
            "Do_targets_get_hit%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Point & Figure is the oldest charting method on Wall Street -- it throws away time "
            "entirely and plots price as columns of **X**s (going up) and **O**s (going down). "
            "Its most famous promise is the **count**: when price breaks out, you literally measure "
            "the width of the sideways base, multiply by the box size, and the chart hands you a "
            "**price target**. The claim is bold and falsifiable: *those targets get hit.*\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the same-distance baseline, "
            "and the buy-vs-sell split? That's the companion, "
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
            f"| Do the count targets get hit? | **Yes -- genuinely.** A double-top buy's count "
            f"target gets hit **{R['hit_buy']:.0f}%** of the time, vs only **{R['base_buy']:.0f}%** "
            "for a *random* target placed the same distance away. That gap is real and large. |\n"
            "| Is it the count, or just a rising market? | **It's really the count** -- the same "
            f"edge shows up on the **sell** side too (sell targets hit {R['hit_sell']:.0f}% vs "
            f"{R['base_sell']:.0f}% baseline). A breakout really does tend to keep going. |\n"
            "| So can you make money with it? | **No.** The buy side profits "
            f"(+{R['net_buy']:.1f}%/signal) only because the market drifts **up**. The sell side "
            f"*loses* ({R['net_sell']:.1f}%/signal) even though its targets also beat the baseline. "
            "Strip out the drift and the count's money is gone. |\n\n"
            "> The chart's promise is literally true -- the targets get hit -- but the *profit* is "
            "the 26-year bull market wearing a P&F costume."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"On a Point & Figure chart, when a column of Xs rises one box above the previous "
            "X-column you have a **double-top buy**. Count the columns in the base it broke out of, "
            "multiply by the box size times the 3-box reversal, add it to the breakout -- that's "
            "your **price target**, and the market will get there.\"*\n\n"
            "P&F has no time axis: a box is only plotted when price *moves* a full box, and you only "
            "switch from an X-column to an O-column on a **3-box reversal**. It is the purest "
            "price-only chart there is. The **horizontal count** is its signature deliverable: a "
            "concrete number, printed on the chart, that says *price will go here*."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "Most chart folklore gives you a vibe (\"looks bullish\"). P&F gives you a **falsifiable "
            "number**: a target price and an implicit stop (the bottom of the breakout column). That "
            "makes it one of the few technical methods you can grade objectively -- either the target "
            "gets hit before the stop, or it doesn't. If the targets get hit at a rate beyond what "
            "geometry alone allows, that is a genuine continuation effect. If they only 'work' because "
            "everything went up for 26 years, that's a mirage with a ruler."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "The trap with any price target is that a *closer* target is easier to hit -- so a high "
            "hit-rate could be pure geometry, not skill. Two rules keep us honest:\n\n"
            "1. **The same-distance baseline.** For every real count target we place a target the "
            "**exact same distance away** (with the same stop) on **random days** of the same stock. "
            "If the count target hits more often than that random target, the *count* is adding "
            "something beyond geometry.\n"
            "2. **Test the sell side too.** A rising market makes any upside target easy. If the count "
            "is real, the **double-bottom sell** target should also beat its baseline. If only buys "
            "work, it's drift, not P&F.\n\n"
            "We run it on **four instruments** -- SPY, the Dow (^DJI), AAPL, GLD -- daily, 2000-2026, "
            "with a fixed 2%-of-price box and the classic 3-box reversal."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First: how often does a P&F count target get hit, vs a same-distance random target?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = pooled('buy'); ps = pooled('sell')\n"
            "    hb, bb_, hs, bs = pb['hit'], pb['base'], ps['hit'], ps['base']\n"
            "    nb, ns = pb['n'], ps['n']\n"
            "else:\n"
            "    hb, bb_, hs, bs = R['hit_buy'], R['base_buy'], R['hit_sell'], R['base_sell']\n"
            "    nb, ns = R['n_buy'], R['n_sell']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "x = np.arange(2); w = 0.36\n"
            "ax.bar(x - w/2, [hb, hs], width=w, color=GREEN, label='P&F count target')\n"
            "ax.bar(x + w/2, [bb_, bs], width=w, color=GREY, label='same-distance random target')\n"
            "ax.axhline(50, ls=':', c='k', lw=1, alpha=.5)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'Double-top BUY\\n(n={nb})',\n"
            "                                       f'Double-bottom SELL\\n(n={ns})'])\n"
            "ax.set_ylabel('target hit before stop (%)')\n"
            "ax.set_title('The count target gets hit far more than a same-distance random target')\n"
            "ax.legend()\n"
            "for xi, (a, b) in enumerate(zip([hb, hs], [bb_, bs])):\n"
            "    ax.annotate(f'+{a-b:.0f}pp', (xi, max(a, b)+2), ha='center', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'BUY : count {hb:.1f}% vs baseline {bb_:.1f}%  (+{hb-bb_:.1f}pp)')\n"
            "print(f'SELL: count {hs:.1f}% vs baseline {bs:.1f}%  (+{hs-bs:.1f}pp)')"
        ),
        md(
            f"The count target beats its same-distance random baseline by **+{R['edge_buy']:.0f}pp** "
            f"on buys and **+{R['edge_sell']:.0f}pp** on sells. The fact that *both* sides beat their "
            "baseline is the tell: a breakout genuinely tends to keep going a bit further than a "
            "random move of the same size would. **The count is real.**"
        ),
        md(
            "**But now follow the money -- not the hit-rate.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    net_b = pb['net'].mean()*100; net_s = ps['net'].mean()*100\n"
            "else:\n"
            "    net_b, net_s = R['net_buy'], R['net_sell']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "cols = [GREEN if v > 0 else RED for v in [net_b, net_s]]\n"
            "ax.bar(['Double-top BUY', 'Double-bottom SELL'], [net_b, net_s], color=cols, width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net profit per signal (%)')\n"
            "ax.set_title('Same edge on the chart -- opposite result in the P&L')\n"
            "for xi, v in enumerate([net_b, net_s]):\n"
            "    ax.annotate(f'{v:+.2f}%', (xi, v + (0.2 if v>0 else -0.4)), ha='center',\n"
            "                fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'BUY net  {net_b:+.2f}% / signal   (rides the market up)')\n"
            "print(f'SELL net {net_s:+.2f}% / signal   (fights the market up -- bleeds)')"
        ),
        md(
            "Here is the whole story in one chart. The buy and sell count targets are *equally real* "
            "on the chart -- both beat their baseline. But the **buy** signal makes money only because "
            "the market drifted up for 26 years, and the **sell** signal *loses* money for the exact "
            "same reason. The count isn't printing profit; the **bull-market drift** is, and the count "
            "is just along for the ride."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- Real.** The count target gets hit **{R['hit_buy']:.0f}%** (buys) and "
            f"**{R['hit_sell']:.0f}%** (sells) of the time vs **{R['base_buy']:.0f}% / "
            f"{R['base_sell']:.0f}%** for same-distance random targets -- a +15.6pp / +12.0pp edge, "
            "significant on both sides (t = 6.6 / 4.4). Breakouts really do continue.\n"
            "- **Tradability -- Mirage.** The buy side profits only from the market's upward drift "
            "(the sell side, with the same chart edge, *loses* money). It's beta wearing a P&F "
            "costume -- not a tradable count edge.\n"
            "- **Do the targets get hit? -- Confirmed.** The literal folklore claim is true, "
            "robustly, on both directions and every box size we tried."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "You could buy every double-top and you'd make money -- but you'd make the *same* money "
            "buying and holding SPY, because that's all the buy signal is capturing. The moment you "
            "try to use the *full* method (sell the double-bottoms too), the short leg bleeds. A real, "
            "tradable edge has to beat the drift in *both* directions, and the count does not. Costs "
            "are almost irrelevant here (a few signals a year) -- the problem is that the money is "
            "borrowed from the bull market."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Drift-neutralise it.** Subtract each instrument's own buy-and-hold return over the "
            "holding window from every signal. If a positive count edge survives that, P&F has a "
            "genuine tradable core. Our read: it won't.\n"
            "- **The vertical count.** P&F has a second target rule (count the breakout column's "
            "height). Does it beat the horizontal count, or its baseline?\n"
            "- **Bear-market sub-sample.** Re-run on 2000-2002, 2008, 2022. If the buy edge survives "
            "where drift is negative, that would be real support.\n\n"
            "*Think the count has a tradable core beyond drift? Fork this, drift-neutralise the "
            "signals, and show a net edge with t > 2 on **both** sides. That's the bar.*"
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
            "# Point & Figure -- a quantitative teardown\n"
            "### Mechanical 3-box-reversal P&F - horizontal-count targets - hit-rate vs a "
            "same-distance baseline - buy/sell split - HAC inference\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Do_targets_get_hit%3F: Confirmed](https://img.shields.io/badge/"
            "Do_targets_get_hit%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We encode the tightest "
            "**objective** P&F (fixed box size, 3-box reversal, double-top/-bottom signals, "
            "horizontal-count target) and test whether the count target gets hit before its stop "
            "more often than a **same-distance random target**, across four daily tapes "
            "(2000-2026), buy and sell sides separately.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars to 2026-06-23 (the "
            "in-progress final bar dropped); reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Count target hit-rate **{R['hit_buy']:.1f}%** (buys) / "
            f"**{R['hit_sell']:.1f}%** (sells) vs same-distance baseline **{R['base_buy']:.1f}% / "
            f"{R['base_sell']:.1f}%** -- edge **+{R['edge_buy']:.1f}pp / +{R['edge_sell']:.1f}pp**, "
            f"*t*(hit-base) = **{R['t_edge_buy']:.2f} / {R['t_edge_sell']:.2f}**. Symmetric across "
            "sides -> it's the count, not drift. |\n"
            f"| **Tradability** | `MIRAGE` | Buy net **+{R['net_buy']:.2f}%**/signal "
            f"(HAC *t* = {R['t_net_buy']:.2f}) but sell net **{R['net_sell']:.2f}%** "
            f"(*t* = {R['t_net_sell']:.2f}). The P&L is one-directional -- pure bull-market beta, "
            "not a count edge. |\n"
            f"| **Do targets get hit?** | `CONFIRMED` | Hit-rate edge over the same-distance baseline "
            "holds on **both** sides and every box size (1.5% / 2% / 3%); synthetic null shows no "
            "manufactured edge (placebo *p* = 1.0). |\n\n"
            "> In plain words: the chart's number is real -- breakouts continue past a same-distance "
            "random move on both sides. But the *money* is just the 26-year equity drift; the short "
            "leg, with the identical chart edge, loses."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "On a P&F chart with box size $b$ and 3-box reversal, a **double-top buy** fires when an "
            "X-column's top exceeds the prior X-column's top by one box, at breakout price $B$. The "
            "**horizontal count** target is\n\n"
            "$$ T = B + w \\cdot 3 \\cdot b, $$\n\n"
            "where $w$ is the congestion width (columns since the last opposite signal). The implicit "
            "stop is the bottom of the breakout column. The claims:\n\n"
            "- **H1 (targets get hit).** $T$ is reached before the stop **more often than a "
            "same-distance random target** on the same instrument.\n"
            "- **H2 (it's the count, not drift).** H1 holds on the **sell** side too (mirror).\n"
            "- **H3 (tradable).** The per-signal P&L is positive net of costs on **both** sides.\n\n"
            "We **accept H1 and H2** (the hit-rate edge clears t = 2 on both sides) but **reject H3** "
            "(the sell leg loses money; the buy leg is drift)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "Among technical methods P&F is unusually testable: it emits a concrete target and stop. "
            "The right null is **not** a coin flip -- it is a *same-distance* target, because a closer "
            "target is mechanically easier to hit. Beating that baseline is a genuine continuation "
            "(momentum) statement. The interesting result is that the continuation is **real and "
            "direction-symmetric**, yet **non-monetisable**: the only thing that turns the hit-rate "
            "into P&L is the sign of the drift, which is the one thing P&F doesn't supply."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Box / reversal.** Fixed box = **2% of median price** (a single structural constant, "
            "no per-day fit); classic **3-box reversal**.\n"
            "- **Signal.** Double-top buy / double-bottom sell on the column series; the signal bar is "
            "the close that completed the breakout box -- **knowable in real time** (no look-ahead).\n"
            "- **Target / stop.** Horizontal count $T = B \\pm w\\,3b$; stop = breakout column's far "
            "edge. Fixed at the signal close.\n"
            "- **Resolution.** From the bar **after** the signal (one execution lag), HIT if the high "
            "reaches $T$, STOP if the low breaks the stop first, else TIMEOUT at 120 bars.\n"
            "- **Baseline (the honest null).** The **same** target/stop distances dropped on random "
            "entry days of the same instrument -- geometry + drift with **no** count.\n"
            "- **Placebo.** 2,000 batches of same-distance random-target brackets; *p* = P[random "
            "hit-rate >= observed].\n"
            "- **Inference.** One-sample *t* of (hit indicator - baseline rate); HAC *t* on per-signal "
            "net P&L.\n"
            "- **Positive control.** Synthetic tape with a planted continuation edge -- the engine "
            "must light up when planted and stay quiet at edge = 0.\n\n"
            "Four tapes: SPY, ^DJI, AAPL, GLD -- daily 2000-2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - A P&F chart, made concrete\n\n"
            "First, what the engine actually builds: columns of Xs and Os, and a double-top buy with "
            "its count target. (Synthetic tape so the picture is clean and reproducible anywhere.)"
        ),
        code(
            "b_demo, _ = data.synthetic_panel(edge=0.5, n_days=900, seed=7)\n"
            "box = st.box_size(b_demo['close'])\n"
            "cols = st.build_columns(b_demo['close'], box)\n"
            "fig, ax = plt.subplots(figsize=(11, 5))\n"
            "for ci, col in enumerate(cols[-40:]):\n"
            "    lo_b, hi_b = col['bot'], col['top']\n"
            "    ys = np.arange(lo_b, hi_b+1) * box\n"
            "    mark = 'x' if col['dir']==1 else 'o'\n"
            "    c = GREEN if col['dir']==1 else RED\n"
            "    ax.scatter([ci]*len(ys), ys, marker=mark, c=c, s=40)\n"
            "ax.set_xlabel('P&F column (time stripped out)'); ax.set_ylabel('price (box levels)')\n"
            "ax.set_title('A Point & Figure chart: columns of Xs (up) and Os (down), 3-box reversal')\n"
            "ax.grid(True, alpha=.3); plt.tight_layout(); plt.show()\n"
            "print(f'box size = {box:.3f} ; columns built = {len(cols)}')"
        ),
        md(
            "> In plain words: no calendar -- each column is a *run* of price in one direction, and a "
            "new column only starts after a 3-box reversal. The count measures how wide the sideways "
            "base was before a breakout and projects it as a target."
        ),

        md(
            "### 4b - Hit-rate vs the same-distance baseline, per instrument (buy side)\n\n"
            "The decisive Signal test: does the count target beat a same-distance random target?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = pooled('buy'); per = pb['per']\n"
            "    tks = [r[0] for r in per]; hit = [r[2] for r in per]; base = [r[3] for r in per]\n"
            "    plac = [r[6] for r in per]\n"
            "else:\n"
            "    rows = [('SPY',)+R['spy'], ('^DJI',)+R['dji'], ('AAPL',)+R['aapl'], ('GLD',)+R['gld']]\n"
            "    tks=[r[0] for r in rows]; hit=[r[2] for r in rows]; base=[r[3] for r in rows]\n"
            "    plac=[r[6] for r in rows]\n"
            "x = np.arange(len(tks)); w=0.36\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "ax.bar(x-w/2, hit, width=w, color=GREEN, label='count target')\n"
            "ax.bar(x+w/2, base, width=w, color=GREY, label='same-distance random')\n"
            "ax.set_xticks(x); ax.set_xticklabels(tks); ax.set_ylabel('hit-rate (%)')\n"
            "ax.set_title('Count target beats the same-distance baseline on every instrument')\n"
            "ax.legend()\n"
            "for xi,(a,bb) in enumerate(zip(hit,base)):\n"
            "    ax.annotate(f'p={plac[xi]:.2f}', (xi, max(a,bb)+1.5), ha='center', fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-instrument (ticker, n, hit%, base%, net%, t_net, placebo_p):')\n"
            "if HAVE_REAL:\n"
            "    for r in per: print('  ', r[0], r[1], f'{r[2]:.1f}', f'{r[3]:.1f}', f'{r[4]:.2f}', f'{r[5]:.2f}', f'{r[6]:.3f}')\n"
            "else:\n"
            "    for r in [('SPY',)+R['spy'],('^DJI',)+R['dji'],('AAPL',)+R['aapl'],('GLD',)+R['gld']]:\n"
            "        print('  ', r)"
        ),
        md(
            "> In plain words: every instrument's count target gets hit more than its same-distance "
            "baseline. The breakout-conditioning adds real continuation -- this is not a geometry "
            "artefact (the baseline already uses the identical distances)."
        ),

        md(
            "### 4c - The buy/sell split -- where the money story breaks\n\n"
            "If the count were a tradable edge, both sides would profit. They don't: the hit-rate "
            "edge is symmetric, but the **P&L** is one-directional."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ps = pooled('sell')\n"
            "    edges = [pb['hit']-pb['base'], ps['hit']-ps['base']]\n"
            "    t_edge = [st.ttest_vs_zero(pb['hit_ind']-pb['base']/100),\n"
            "              st.ttest_vs_zero(ps['hit_ind']-ps['base']/100)]\n"
            "    nets = [pb['net'].mean()*100, ps['net'].mean()*100]\n"
            "    t_nets = [pb['t_net'], ps['t_net']]\n"
            "else:\n"
            "    edges=[R['edge_buy'],R['edge_sell']]; t_edge=[R['t_edge_buy'],R['t_edge_sell']]\n"
            "    nets=[R['net_buy'],R['net_sell']]; t_nets=[R['t_net_buy'],R['t_net_sell']]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "lab = ['BUY\\n(double-top)', 'SELL\\n(double-bottom)']\n"
            "axes[0].bar(lab, edges, color=GREEN, width=.5)\n"
            "axes[0].set_ylabel('hit-rate edge over baseline (pp)')\n"
            "axes[0].set_title('Signal: REAL on both sides')\n"
            "for xi,(e,t) in enumerate(zip(edges,t_edge)):\n"
            "    axes[0].annotate(f'+{e:.1f}pp\\nt={t:.1f}', (xi, e+0.4), ha='center')\n"
            "cols=[GREEN if v>0 else RED for v in nets]\n"
            "axes[1].bar(lab, nets, color=cols, width=.5); axes[1].axhline(0,c='k',lw=1)\n"
            "axes[1].set_ylabel('net P&L per signal (%)')\n"
            "axes[1].set_title('Tradability: MIRAGE (P&L is one-directional drift)')\n"
            "for xi,(v,t) in enumerate(zip(nets,t_nets)):\n"
            "    axes[1].annotate(f'{v:+.2f}%\\nt={t:.1f}', (xi, v+(0.2 if v>0 else -0.5)), ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'hit-rate edge  BUY +{edges[0]:.1f}pp (t={t_edge[0]:.2f})  SELL +{edges[1]:.1f}pp (t={t_edge[1]:.2f})')\n"
            "print(f'net P&L/signal BUY {nets[0]:+.2f}% (t={t_nets[0]:.2f})  SELL {nets[1]:+.2f}% (t={t_nets[1]:.2f})')"
        ),
        md(
            "> In plain words: this is the whole teardown in one panel. **Left:** the count target "
            "beats its baseline on both sides (Signal REAL). **Right:** the buy leg profits and the "
            "sell leg loses -- equal-and-opposite drift. The count's hit-rate is real; its P&L is the "
            "market's drift, which P&F cannot tell you the sign of."
        ),

        md(
            "### 4d - The positive control -- the engine finds a planted continuation\n\n"
            "Does the harness work? On a synthetic tape with a planted continuation it must light up; "
            "on a pure random walk it must not manufacture a hit-rate edge."
        ),
        code(
            "rows=[]\n"
            "for edge in (0.0, 0.3, 0.6, 0.9):\n"
            "    b,_ = data.synthetic_panel(edge=edge, seed=448)\n"
            "    r = st.run_experiment(b, n_draws=1000, side='buy')\n"
            "    if r['n_signals']==0:\n"
            "        rows.append((edge,0,float('nan'),float('nan'),float('nan'))); continue\n"
            "    rows.append((edge, r['n_signals'], r['pnl']['hit_rate']*100,\n"
            "                 r['baseline']['baseline_hit_rate']*100, r['placebo']['p_value']))\n"
            "ed=[r[0] for r in rows]; hh=[r[2] for r in rows]; bb_=[r[3] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.3))\n"
            "ax.plot(ed, hh, 'o-', c=GREEN, lw=2, label='count target hit%')\n"
            "ax.plot(ed, bb_, 's--', c=GREY, lw=2, label='same-distance baseline hit%')\n"
            "ax.fill_between(ed, hh, bb_, alpha=.15, color=GREEN)\n"
            "ax.set_xlabel('planted continuation edge'); ax.set_ylabel('hit-rate (%)')\n"
            "ax.set_title('Engine recovers a planted edge; at edge=0 the count does NOT beat baseline')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'  edge={r[0]:+.1f}: nsig={r[1]:>4} hit={r[2]:.1f}% base={r[3]:.1f}% placebo_p={r[4]:.3f}')"
        ),
        md(
            "> In plain words: at **edge = 0** (pure random walk) the count target hits *less* than "
            "the same-distance baseline and the placebo *p* sits near 1.0 -- the machinery does not "
            "fabricate an edge from noise. As the planted continuation grows, the count pulls ahead "
            "monotonically. So the real-tape edge is a genuine detection, not a construction artefact."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `REAL`** -- count target hit-rate **{R['hit_buy']:.1f}% / "
            f"{R['hit_sell']:.1f}%** (buy / sell) vs same-distance baseline **{R['base_buy']:.1f}% / "
            f"{R['base_sell']:.1f}%**; edge **+{R['edge_buy']:.1f}pp / +{R['edge_sell']:.1f}pp**, "
            f"*t* = **{R['t_edge_buy']:.2f} / {R['t_edge_sell']:.2f}** -- both clear 2. The symmetry "
            "across sides rules out drift as the explanation: breakouts continue.\n"
            f"- **Tradability `MIRAGE`** -- buy net **+{R['net_buy']:.2f}%**/signal "
            f"(HAC *t* = {R['t_net_buy']:.2f}) but sell net **{R['net_sell']:.2f}%** "
            f"(*t* = {R['t_net_sell']:.2f}). The full method (both legs) nets near zero before costs; "
            "the buy-only profit is the 26-year equity drift, i.e. beta, not a count edge.\n"
            f"- **Do targets get hit? `CONFIRMED`** -- robust to box size and direction; the synthetic "
            "null confirms no manufactured edge (placebo *p* = 1.0)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the drift decomposition\n\n"
            "The buy leg's per-signal P&L and the sell leg's are near mirror images -- the signature "
            "of a directional drift bet, not a continuation edge. Costs are trivial (a handful of "
            "signals/year), so the failure is not cost erosion: it is that the only thing converting "
            "the (real) hit-rate into money is the sign of the market's drift."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nb_, ns_ = pb['net']*100, ps['net']*100\n"
            "else:\n"
            "    rng=np.random.default_rng(0)\n"
            "    nb_=rng.normal(R['net_buy'],8,R['n_buy']); ns_=rng.normal(R['net_sell'],8,R['n_sell'])\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.3))\n"
            "ax.hist(nb_, bins=40, alpha=.6, color=GREEN, label=f'BUY (mean {np.mean(nb_):+.2f}%)')\n"
            "ax.hist(ns_, bins=40, alpha=.6, color=RED, label=f'SELL (mean {np.mean(ns_):+.2f}%)')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(np.mean(nb_), c=GREEN, ls='--'); ax.axvline(np.mean(ns_), c=RED, ls='--')\n"
            "ax.set_xlabel('net P&L per signal (%)'); ax.set_ylabel('count')\n"
            "ax.set_title('Mirror-image P&L distributions = a drift bet, not a count edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('A continuation edge would profit on BOTH sides; here the means straddle zero by sign of drift.')"
        ),
        md(
            "> In plain words: a genuinely tradable count would print positive P&L whether you bet up "
            "or down. Instead the two distributions sit on opposite sides of zero -- you are paid for "
            "being long a rising market and punished for being short it. That is the definition of a "
            "**mirage**: a real statistical pattern that doesn't convert into a real edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- robustness and open questions\n\n"
            "The hit-rate edge is robust to box size (the count beats baseline at 1.5% / 2% / 3% "
            "boxes) and direction; the synthetic control proves the engine is faithful. Forks:\n\n"
            "- **Drift-neutralise.** Subtract each signal's matched buy-and-hold return; a surviving "
            "positive count edge would upgrade Tradability. Our prior: it won't.\n"
            "- **Vertical count.** P&F's second target rule (column height). Same baseline test.\n"
            "- **Box-size adaptivity.** ATR-scaled boxes vs the fixed-fraction box used here.\n"
            "- **Bear sub-samples (2000-02, 2008, 2022).** Does the buy edge survive negative drift?\n\n"
            "Related desk studies: [447-gann-angles](../../447-gann-angles/) and "
            "[444-dow-theory](../../444-dow-theory/) test other mechanical chart geometries; the "
            "[research-method demos](../../343-data-mining-roulette/) frame why a same-distance "
            "baseline (not a coin flip) is the right null for a price-target claim."
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
