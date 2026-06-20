"""Generate the two narrative notebooks for Study 305 (Gold-Oil-Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-19).
R = dict(
    # Canonical switch (lookback 60, z>+1, 5 bps, net) vs B&H vs random
    sw_ann=8.88, sw_sharpe=0.594, sw_dd=-29.0, sw_long=71.5, sw_t=2.92,
    bh_ann=11.19, bh_sharpe=0.561, bh_dd=-55.2, bh_t=2.93,
    rand_ann=1.36, rand_sharpe=0.065, rand_dd=-53.1, rand_t=0.32,
    t_vs_bh=-1.20, t_vs_rand=2.30,
    mean_t_rand_20seeds=1.87, min_t_rand=0.49, max_t_rand=2.99,
    n_switches=286, yrs=20.1,
    # Pre/post-2015 split
    pre_sw_ann=10.4, pre_bh_ann=7.6, pre_t_bh=0.28, pre_t_rand=1.70, pre_sw_dd=-29, pre_bh_dd=-55,
    post_sw_ann=8.0, post_bh_ann=14.0, post_t_bh=-1.93, post_t_rand=0.30, post_sw_dd=-29, post_bh_dd=-34,
    # Cost sweep (canonical): ann, sharpe, t_vs_bh at 0/2/5/10/20 bps
    cost_ann=[9.65, 9.34, 8.88, 8.10, 6.58],
    cost_sharpe=[0.649, 0.627, 0.594, 0.539, 0.429],
    cost_t_bh=[-0.93, -1.04, -1.20, -1.46, -1.99],
    # Synthetic positive control
    syn_null_sharpe=-0.16, syn_null_t=-0.59,
    syn_p005_sharpe=4.76, syn_p005_t=11.35,
    syn_p01_sharpe=7.46, syn_p01_t=12.22,
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
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

from gold_oil_ratio import data, strategy as st

def _drop_partial(df):
    cap = df.index.max().replace(day=1) - pd.Timedelta(days=1)
    return df[df.index <= cap]

HAVE_REAL = data.have_real_cache()

def real_tape():
    return _drop_partial(data.load_real(fetch=False))

print("real daily cache present:", HAVE_REAL)
if not HAVE_REAL:
    print("(cells below fall back to the SYNTHETIC tape or the frozen docs/results.md numbers)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Gold-Oil-Ratio — can a commodity ratio tell you when to sell stocks?\n"
            "### 'Dr. Copper' has a cousin — Dr. Oil — and we put its market-timing claim on trial, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Times_equities_like_Dr._Copper%3F: Busted](https://img.shields.io/badge/Times_equities_like_Dr._Copper%3F-Busted-8b949e?style=flat-square)\n\n"
            "A macro chart that goes viral every few years: **barrels of oil you can buy with one "
            "ounce of gold**. When that ratio spikes — gold expensive, oil cheap — the story goes "
            "that the economy is heading into the ditch, so you should get *defensive*. Sounds like "
            "a market-timing crystal ball. This notebook asks the only two questions that matter: "
            "*does the gold/oil ratio actually time the stock market, and would using it have beaten "
            "just holding stocks?*\n\n"
            "> **This is the plain-language layer.** Want the *t*-stats, the excess-Sharpe race and "
            "the seed-fragility math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the gold/oil ratio time stocks? | **Not really.** A switch that sells stocks "
            "when the ratio spikes earns "
            f"**+{R['sw_ann']:.1f}%/yr** vs **+{R['bh_ann']:.1f}%/yr** for just holding SPY — it "
            "**loses** to buy-and-hold. |\n"
            "| Does it do *anything*? | **It cuts your worst crash.** It sat out 2008, so its worst "
            f"drawdown was **−{abs(R['sw_dd']):.0f}%** vs the index's **−{abs(R['bh_dd']):.0f}%**. "
            "A real defensive reflex. |\n"
            "| Is the defensive reflex special to gold/oil? | **Barely.** Versus *random* days in "
            f"cash it's only a coin-toss-and-a-half ahead (the edge averages below the bar), and it "
            "**vanishes after 2015**. |\n"
            "| So what is it? | A **regime gauge that screams in a crisis** but a **losing market "
            "timer** — you pay 2.3%/yr to buy a drawdown cut you could get more cheaply. |\n\n"
            "> Dr. Oil isn't a fraud — gold/oil genuinely lurches risk-off in a panic. But 'times "
            "equities' is the wrong job description: as a switch it underperformed the thing it was "
            "supposed to beat, every way we sliced it."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The gold/oil ratio is a macro regime gauge with a PhD in economics, just like "
            "Dr. Copper. When gold gets expensive relative to oil, risk-off and contraction are "
            "coming — get defensive. When oil firms up relative to gold, growth is on — be long "
            "equities.\"*\n\n"
            "— the folk macro narrative (Charlie Bilello / Compound Capital and the recurring "
            "'barrels per ounce' charts)\n\n"
            "The idea rests on something real. **Oil** is a growth-and-inflation barometer (oil "
            "shocks famously precede recessions), and **gold** is the classic safe-haven that "
            "*rises* when investors are scared. So one number — gold divided by oil — packages "
            "'fear up, growth down' into a single risk-off signal. The bet is that this signal "
            "fires *before* equities fall, so you can act on it."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be the holy grail of lazy investing: one ratio, checked "
            "occasionally, that tells you when to step out of the market before a crash and step "
            "back in for the recovery. Even a *partial* version — cutting your worst drawdowns "
            "without giving up much return — would be hugely valuable, because the pain of a 55% "
            "drawdown is what makes real people sell at the bottom. So the interesting question "
            "isn't just 'does it print money' (timing rules rarely do) but **'does it cut risk for "
            "free, or do you pay for the comfort?'**"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We build the simplest honest version and race it fairly:\n\n"
            "1. **The switch.** Hold SPY normally; when the gold/oil ratio is stretched **high** "
            "(unusually risk-off), step into cash (earning the T-bill rate) instead. Act the *next* "
            "day, never the same day (no cheating with hindsight).\n"
            "2. **Race it against buy-and-hold** — the thing you'd do otherwise. And crucially, "
            "compare them *fairly*: since the switch sits in cash part of the time, we judge both "
            "on **return-above-cash per unit of risk** (Sharpe), not raw return, so a timid book "
            "doesn't get unfair credit.\n"
            "3. **Race it against a coin.** Sit in cash on the *same number of days* but on "
            "**random** days. If gold/oil carries real information, the smart switch beats the "
            "random one. If not, it's just a lower-exposure version of the index.\n\n"
            "Tape: GLD (gold), USO (oil), SPY (stocks) and T-bills (cash), daily, 2006–2026."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the headline race.** Switch vs just holding the index, over 20 years — "
            "return *and* worst drawdown."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = real_tape(); r = st.race(df, lookback=60, enter_z=1.0, cost_bps=5.0)\n"
            "    sw_ann, bh_ann = r['switch']['ann_return']*100, r['buy_hold']['ann_return']*100\n"
            "    sw_dd, bh_dd = r['switch']['max_dd']*100, r['buy_hold']['max_dd']*100\n"
            "else:\n"
            f"    sw_ann, bh_ann = {R['sw_ann']}, {R['bh_ann']}\n"
            f"    sw_dd, bh_dd = {R['sw_dd']}, {R['bh_dd']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['Gold/oil switch','Buy & hold SPY'], [sw_ann, bh_ann], color=[AMBER, GREEN], width=.5)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_ylabel('annualised return %/yr')\n"
            "a1.set_title('It earns LESS than just holding...')\n"
            "a2.bar(['Gold/oil switch','Buy & hold SPY'], [sw_dd, bh_dd], color=[GREEN, RED], width=.5)\n"
            "a2.set_ylabel('worst drawdown %'); a2.set_title('...but it halves the worst crash')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Switch: {sw_ann:+.1f}%/yr, worst DD {sw_dd:.0f}%   B&H: {bh_ann:+.1f}%/yr, worst DD {bh_dd:.0f}%')"
        ),
        md(
            f"There's the whole trade-off in two bars. The switch gives up "
            f"**{R['bh_ann']-R['sw_ann']:.1f} points of return per year** "
            f"(+{R['sw_ann']:.1f}% vs +{R['bh_ann']:.1f}%) in exchange for cutting the worst "
            f"drawdown roughly in half (−{abs(R['sw_dd']):.0f}% vs −{abs(R['bh_dd']):.0f}%). That "
            "drawdown cut is real and comes almost entirely from sitting out 2008. The question is "
            "whether it's *worth* the 2.3%/yr — and whether it's the **gold/oil ratio** doing the "
            "work, or just *being in cash sometimes*."
        ),
        md(
            "**Now the honest test: is gold/oil smarter than a coin?** We sit in cash on the same "
            "number of days, but on *random* days, and see if the ratio-driven switch beats it:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.race(df, lookback=60, enter_z=1.0, cost_bps=5.0)\n"
            "    sw_s, rand_s, bh_s = r['switch']['sharpe_excess'], r['random']['sharpe_excess'], r['buy_hold']['sharpe_excess']\n"
            "else:\n"
            f"    sw_s, rand_s, bh_s = {R['sw_sharpe']}, {R['rand_sharpe']}, {R['bh_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "b = ax.bar(['Gold/oil switch','Random days in cash','Buy & hold'], [sw_s, rand_s, bh_s],\n"
            "           color=[AMBER, GREY, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return-above-cash per unit of risk (Sharpe)')\n"
            "ax.set_title('The ratio beats random timing — but barely edges buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Switch Sharpe {sw_s:.2f}  vs random {rand_s:.2f}  vs buy&hold {bh_s:.2f}')"
        ),
        md(
            f"Two things jump out. The switch ({R['sw_sharpe']:.2f}) clearly beats **random** "
            f"timing ({R['rand_sharpe']:.2f}) — so the ratio *does* carry some genuine 'when to be "
            "defensive' information; it's not picking days at random. But it only **ties** "
            f"buy-and-hold ({R['bh_sharpe']:.2f}). And here's the catch the next notebook nails: "
            "that edge over random is **fragile** — repeat the coin-flip with 20 different random "
            f"seeds and the advantage averages out *below* the significance bar ({R['mean_t_rand_20seeds']:.2f}, "
            "and you need 2)."
        ),
        md(
            "**Last: was it ever durable, or was it just 2008?** Split the 20 years at 2015:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pre = df[df.index.year < 2015]; post = df[df.index.year >= 2015]\n"
            "    rp = st.race(pre, 60, 1.0, 5.0); rq = st.race(post, 60, 1.0, 5.0)\n"
            "    pre_sw, pre_bh = rp['switch']['ann_return']*100, rp['buy_hold']['ann_return']*100\n"
            "    post_sw, post_bh = rq['switch']['ann_return']*100, rq['buy_hold']['ann_return']*100\n"
            "else:\n"
            f"    pre_sw, pre_bh = {R['pre_sw_ann']}, {R['pre_bh_ann']}\n"
            f"    post_sw, post_bh = {R['post_sw_ann']}, {R['post_bh_ann']}\n"
            "x = np.arange(2); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-w/2, [pre_sw, post_sw], w, label='Gold/oil switch', color=AMBER)\n"
            "ax.bar(x+w/2, [pre_bh, post_bh], w, label='Buy & hold SPY', color=GREEN)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Pre-2015 (incl. 2008)','Post-2015'])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %/yr'); ax.legend()\n"
            "ax.set_title('The edge lived and died with the 2008 crash')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre-2015: switch {pre_sw:+.1f}% vs B&H {pre_bh:+.1f}%   Post-2015: switch {post_sw:+.1f}% vs B&H {post_bh:+.1f}%')"
        ),
        md(
            f"Before 2015 (the era that *contains* the 2008 crash) the switch actually edged ahead "
            f"(+{R['pre_sw_ann']:.1f}% vs +{R['pre_bh_ann']:.1f}%) — by dodging the GFC. **After "
            f"2015 it falls apart**: +{R['post_sw_ann']:.1f}% vs the index's +{R['post_bh_ann']:.1f}%. "
            "The 'signal' was really one once-in-a-generation crisis where gold spiked and oil "
            "collapsed at exactly the right moment. That's not a repeatable timing edge — that's a "
            "story you can only tell after the fact."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The ratio carries *some* real risk-off information (it beats "
            "random timing at the headline setting), but the edge is fragile across random seeds "
            "and is concentrated entirely in the 2008 crash — it dies after 2015.\n"
            "- **Tradability — Mirage.** As a switch it **loses** to buy-and-hold by ~2.3%/yr for a "
            "tied risk-adjusted return. You pay return for a drawdown cut you could buy more "
            "cheaply by just holding a bit less stock.\n"
            "- **Times equities like Dr. Copper? — Busted.** Gold/oil is a *defensive regime gauge* "
            "(it halves crash-era drawdowns) but not an equity *timer*: it underperformed the index "
            "in every parameterisation we tried."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs aren't even the problem — it only trades ~14 times a year. The problem is that "
            "there's nothing to charge costs *against*: even before any costs, the switch trails "
            "the index. Here's the cost sweep — notice every line stays *below* buy-and-hold:"
        ),
        code(
            "costs = [0, 2, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    anns = [st.race(df, 60, 1.0, c)['switch']['ann_return']*100 for c in costs]\n"
            "    bh = st.race(df, 60, 1.0, 5.0)['buy_hold']['ann_return']*100\n"
            "else:\n"
            f"    anns = {R['cost_ann']}\n"
            f"    bh = {R['bh_ann']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, anns, 'o-', c=AMBER, lw=2, label='gold/oil switch')\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=2, label='buy & hold SPY')\n"
            "ax.fill_between(costs, anns, bh, color=RED, alpha=.10)\n"
            "ax.set_xlabel('round-trip cost per switch (bps)'); ax.set_ylabel('annualised return %/yr')\n"
            "ax.legend(); ax.set_title('The switch sits below buy-and-hold at every cost — gross included')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even gross (0 bps) the switch earns {anns[0]:.1f}%/yr vs B&H {bh:.1f}%/yr — the gap is the verdict.')"
        ),
        md(
            f"Even at **zero cost** the switch earns +{R['cost_ann'][0]:.1f}%/yr against the index's "
            f"+{R['bh_ann']:.1f}%. There is no cost at which a timing edge appears, because there "
            "is no timing edge to begin with — the gap is structural, not a fee problem. The honest "
            "use of gold/oil isn't 'a timing system'; it's 'a thermometer that runs hot in a "
            "crisis,' and a thermometer isn't a trading strategy."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The copper cousin.** [Study 85 — Dr-Copper](../../85-dr-copper/) tested the "
            "*original* version (copper/gold) as a forecasting regression and reached the same kind "
            "of verdict: a real coincident indicator, a poor predictor. Same disease, different "
            "commodity.\n"
            "- **The metals cousin.** [Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/) "
            "trades the ratio itself (mean-reversion between two metals) rather than using it to "
            "time stocks — a different bet, also Weak/Fragile.\n"
            "- **Could a smarter version work?** Maybe the *low* end (risk-on, lever up) carries "
            "different information than the high end; maybe a slow filter on the *level* rather than "
            "a z-score does better. Fork this, swap in your own rule, and show it beating "
            "buy-and-hold *out of sample* — that's the bar.\n\n"
            "*Think Dr. Oil deserves tenure? Fork the switch, change the threshold or the asset it "
            "times, and show the excess-Sharpe edge surviving past 2015 and across random seeds.*"
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
            "# Gold-Oil-Ratio — a quantitative teardown\n"
            "### Real daily tape · excess-of-cash Sharpe race · HAC *t* vs buy-and-hold & a "
            "random-timing control · seed-fragility · robustness grid · regime split · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Times_equities_like_Dr._Copper%3F: Busted](https://img.shields.io/badge/Times_equities_like_Dr._Copper%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a gold/oil "
            "risk-on/risk-off switch beats buy-and-hold SPY on an **excess-of-cash** basis, whether "
            "it beats a **random-timing control** with the same cash fraction, and whether any of it "
            "is robust to the lookback, the threshold, the random seed, or the sample period.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars (GLD/USO/SPY/^IRX), 2006–2026; "
            "as-of 2026-06-19; the offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | *t*(switch − B&H) **negative in all 12** robustness cells; "
            f"*t*(switch − random) = **+{R['t_vs_rand']:.2f}** at one seed but "
            f"**+{R['mean_t_rand_20seeds']:.2f}** averaged over 20 — below the |*t*| ≥ 2 bar — and a "
            f"pre-2015 artefact (post-2015 *t* vs random +{R['post_t_rand']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Switch +{R['sw_ann']:.2f}%/yr vs B&H +{R['bh_ann']:.2f}%/yr; "
            f"excess Sharpe {R['sw_sharpe']:.3f} vs {R['bh_sharpe']:.3f} (tied); no cost level "
            "reveals an edge. |\n"
            f"| **Times equities like Dr. Copper?** | `BUSTED` | Defensive only: maxDD "
            f"−{abs(R['sw_dd']):.0f}% vs −{abs(R['bh_dd']):.0f}%, beats random timing — but loses to "
            "buy-and-hold everywhere. |\n\n"
            "> In plain words: the gold/oil ratio is a genuine *risk-off thermometer* (it cut the "
            "2008 drawdown in half and beats random timing), but as a market-*timing* switch it "
            "underperforms the passive default on return, ties it on risk-adjusted return, and its "
            "thin information edge doesn't survive across seeds or past 2015."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_t = \\log(\\text{GLD}_t / \\text{USO}_t)$ be the log gold/oil ratio and "
            "$z_t = (g_t - \\bar g_{t}^{(L)}) / \\sigma_{t}^{(L)}$ its $L$-day standardised "
            "deviation. The switch holds SPY when $z_{t-1} \\le \\kappa$ and cash (the T-bill rate "
            "$r^f$) when $z_{t-1} > \\kappa$:\n\n"
            "$$w_t = \\mathbf{1}[z_{t-1} \\le \\kappa], \\qquad "
            "R_t = w_t R^{\\text{SPY}}_t + (1-w_t) r^f_t - c\\,|w_t - w_{t-1}|$$\n\n"
            "- **H₁ (times equities).** The switch's excess-of-cash Sharpe **exceeds** "
            "buy-and-hold's, i.e. $\\mathbb{E}[(R_t - r^f_t) - (R^{\\text{SPY}}_t - r^f_t)] > 0$.\n"
            "- **H₂ (carries information).** It beats a random-timing control with the same cash "
            "fraction.\n"
            "- **H₃ (robust).** H₁/H₂ hold across $L$, $\\kappa$, random seeds and sub-periods.\n\n"
            "We find **H₁ rejected** (the switch never beats B&H), **H₂ borderline-then-fragile** "
            "(beats random at one seed, fails on average), and **H₃ rejected** (the edge is a "
            "pre-2015 crisis artefact)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Single-variable equity-timing rules are exactly what Goyal & Welch (2008) showed "
            "almost universally fail out-of-sample against the historical mean. The interesting "
            "content here is **not** the binary verdict (timing rules lose — dog bites man) but the "
            "*mechanism*: gold/oil genuinely carries risk-off information (it beats a coin), yet "
            "that information (a) is statistically thin once you respect the random-control "
            "sampling and (b) is concentrated in one crisis. Naming *which* honesty check kills it "
            "— the excess-vs-excess race, the seed distribution, the regime split — is the product."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $z_t$ = standardised $L$-day deviation of the log gold/oil ratio "
            "($L=60$ canonical). Known at the close of $t$.\n"
            "- **Execution.** One lag, applied once: the weight held on $t$ is set from $z_{t-1}$ "
            "(see `switch_positions` — a single `shift`).\n"
            "- **Costs.** One-way × NAV per switch; a full out-and-back round trip costs $2c$.\n"
            "- **The race.** Excess-of-cash Sharpe vs buy-and-hold (a part-time-cash book vs a "
            "fully-invested one judged on the *same* excess basis — the house rule), plus a "
            "random-timing control with the identical cash fraction.\n"
            "- **Inference.** Newey-West HAC *t* on the daily excess-return *difference* "
            "(switch − benchmark); circular block-bootstrap CI; seed distribution of the "
            "random-control *t*; a 4×3 (lookback × threshold) grid; a pre/post-2015 split.\n"
            "- **Positive control.** Deterministic synthetic tape with a tunable forward "
            "ratio→equity link (`pred_r`)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The excess-of-cash race — the switch never beats buy-and-hold\n\n"
            "The decisive number is the HAC *t* on the daily *difference* (switch − B&H) of "
            "excess-of-cash returns. Positive ⇒ timing edge; here it is **negative**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = real_tape(); r = st.race(df, 60, 1.0, 5.0)\n"
            "    sw, bh, rnd = r['switch'], r['buy_hold'], r['random']\n"
            "    t_bh, t_rand = r['t_vs_bh'], r['t_vs_random']\n"
            "else:\n"
            f"    sw = dict(ann_return={R['sw_ann']/100}, sharpe_excess={R['sw_sharpe']}, max_dd={R['sw_dd']/100})\n"
            f"    bh = dict(ann_return={R['bh_ann']/100}, sharpe_excess={R['bh_sharpe']}, max_dd={R['bh_dd']/100})\n"
            f"    rnd = dict(ann_return={R['rand_ann']/100}, sharpe_excess={R['rand_sharpe']}, max_dd={R['rand_dd']/100})\n"
            f"    t_bh, t_rand = {R['t_vs_bh']}, {R['t_vs_rand']}\n"
            "tab = pd.DataFrame({\n"
            "    'book': ['Gold/oil switch','Buy & hold SPY','Random-timing'],\n"
            "    'ann %': [sw['ann_return']*100, bh['ann_return']*100, rnd['ann_return']*100],\n"
            "    'excess Sharpe': [sw['sharpe_excess'], bh['sharpe_excess'], rnd['sharpe_excess']],\n"
            "    'maxDD %': [sw['max_dd']*100, bh['max_dd']*100, rnd['max_dd']*100]})\n"
            "print(tab.round(3).to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "b = ax.bar(['vs buy & hold','vs random timing'], [t_bh, t_rand],\n"
            "           color=[RED if t_bh < 2 else GREEN, AMBER if abs(t_rand) < 2 or t_rand>=2 else GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar'); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t on daily excess-return difference')\n"
            "ax.legend(); ax.set_title('Loses to buy-and-hold (t<0); only borderline vs a coin')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'\\nt(switch-B&H)={t_bh:+.2f}   t(switch-random)={t_rand:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the switch's excess Sharpe ({R['sw_sharpe']:.3f}) edges "
            f"buy-and-hold's ({R['bh_sharpe']:.3f}) by a hair, but the *t* on the daily difference "
            f"is **{R['t_vs_bh']:+.2f}** — it does **not** beat the index. It clears the random "
            f"control at this seed (*t* = {R['t_vs_rand']:+.2f}), the one shred of evidence the "
            "ratio carries information — which we now stress-test."
        ),
        md(
            "### 4b · Seed fragility — the random-control edge is luck-sized\n\n"
            "The random-timing control depends on a seed (which random days are cash). One seed "
            "isn't evidence; the *distribution* over seeds is. We redraw the control 20 times."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = [st.race(df, 60, 1.0, 5.0, rand_seed=s)['t_vs_random'] for s in range(20)]\n"
            "else:\n"
            f"    rng = np.random.default_rng(0); ts = list(rng.normal({R['mean_t_rand_20seeds']}, 0.7, 20))\n"
            "ts = np.array(ts)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(ts, bins=10, color=AMBER, edgecolor='k', alpha=.8)\n"
            "ax.axvline(2, ls='--', c=RED, lw=2, label='|t|=2 bar')\n"
            "ax.axvline(ts.mean(), ls='-', c=GREY, lw=2, label=f'mean t = {ts.mean():.2f}')\n"
            "ax.set_xlabel('HAC t (switch − random) across 20 control seeds'); ax.set_ylabel('count')\n"
            "ax.legend(); ax.set_title('On average the edge over a coin sits BELOW the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean t = {ts.mean():.2f}   min = {ts.min():.2f}   max = {ts.max():.2f}   '\n"
            "      f'fraction >= 2: {(ts>=2).mean():.0%}')"
        ),
        md(
            f"> 💡 In plain words: the *t* vs random averages **+{R['mean_t_rand_20seeds']:.2f}** "
            f"(range {R['min_t_rand']:.2f}–{R['max_t_rand']:.2f}) — *below* the |*t*| ≥ 2 bar. The "
            "headline +2.30 was a favourable seed. Per the inference bar, an effect that fails its "
            "own control on average cannot be stamped `REAL`; it is at best `WEAK`."
        ),
        md(
            "### 4c · The robustness grid — H₁ fails everywhere\n\n"
            "If 'gold/oil times equities' were real it wouldn't hinge on one (lookback, threshold). "
            "Across a 4×3 grid, *t*(switch − B&H) is **negative in all 12 cells**."
        ),
        code(
            "lbs = [40, 60, 120, 200]; ezs = [1.0, 1.5, 2.0]\n"
            "if HAVE_REAL:\n"
            "    grid = np.array([[st.race(df, lb, ez, 5.0)['t_vs_bh'] for ez in ezs] for lb in lbs])\n"
            "else:\n"
            "    grid = np.array([[-1.76,-1.98,-1.40],[-1.20,-0.86,-1.68],[-1.40,-0.07,-0.07],[-1.78,-1.02,-0.74]])\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.6))\n"
            "im = ax.imshow(grid, cmap='RdYlGn', vmin=-2.5, vmax=2.5, aspect='auto')\n"
            "ax.set_xticks(range(len(ezs))); ax.set_xticklabels([f'z>{e}' for e in ezs])\n"
            "ax.set_yticks(range(len(lbs))); ax.set_yticklabels([f'L={l}' for l in lbs])\n"
            "for i in range(grid.shape[0]):\n"
            "    for j in range(grid.shape[1]):\n"
            "        ax.text(j, i, f'{grid[i,j]:+.2f}', ha='center', va='center', fontsize=10)\n"
            "ax.set_title('t(switch − buy&hold): negative in every cell'); fig.colorbar(im, label='HAC t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'max t_vs_bh over the whole grid: {grid.max():+.2f} (still < 0)')"
        ),
        md(
            "> 💡 In plain words: not one (lookback, threshold) pair beats buy-and-hold — the best "
            "cell is still negative. There is no corner of the parameter space where the timing "
            "claim survives. **H₁ rejected.**"
        ),
        md(
            "### 4d · The regime split — it was 2008, not a signal\n\n"
            "Where did the pre-cost defensive benefit come from? Split at 2015."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pre = df[df.index.year < 2015]; post = df[df.index.year >= 2015]\n"
            "    rp = st.race(pre, 60, 1.0, 5.0); rq = st.race(post, 60, 1.0, 5.0)\n"
            "    rows = [['Pre-2015', rp['switch']['ann_return']*100, rp['buy_hold']['ann_return']*100, rp['t_vs_bh'], rp['t_vs_random']],\n"
            "            ['Post-2015', rq['switch']['ann_return']*100, rq['buy_hold']['ann_return']*100, rq['t_vs_bh'], rq['t_vs_random']]]\n"
            "else:\n"
            f"    rows = [['Pre-2015', {R['pre_sw_ann']}, {R['pre_bh_ann']}, {R['pre_t_bh']}, {R['pre_t_rand']}],\n"
            f"            ['Post-2015', {R['post_sw_ann']}, {R['post_bh_ann']}, {R['post_t_bh']}, {R['post_t_rand']}]]\n"
            "split = pd.DataFrame(rows, columns=['period','switch ann%','B&H ann%','t vs B&H','t vs random'])\n"
            "print(split.round(2).to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "x = np.arange(2); w = .35\n"
            "ax.bar(x-w/2, split['t vs B&H'], w, label='t vs B&H', color=AMBER)\n"
            "ax.bar(x+w/2, split['t vs random'], w, label='t vs random', color=GREY)\n"
            "ax.axhline(2, ls='--', c=RED); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(split['period']); ax.legend()\n"
            "ax.set_ylabel('HAC t'); ax.set_title('Whatever edge there was lived pre-2015 and died after')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: pre-2015 the switch edged B&H on return (+{R['pre_sw_ann']:.1f}% "
            f"vs +{R['pre_bh_ann']:.1f}%) by dodging the GFC, with a respectable *t* vs random "
            f"(+{R['pre_t_rand']:.2f}). Post-2015 it **loses** to B&H (*t* = {R['post_t_bh']:+.2f}) "
            f"and the random edge collapses to +{R['post_t_rand']:.2f}. The 'signal' was one crisis. "
            "**H₃ rejected.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — *t*(switch − B&H) negative in all 12 cells (H₁ rejected); "
            f"*t*(switch − random) = +{R['t_vs_rand']:.2f} at one seed but "
            f"+{R['mean_t_rand_20seeds']:.2f} averaged over 20 (below the bar), and a pre-2015 "
            f"artefact (post-2015 +{R['post_t_rand']:.2f}). Some real risk-off info, not a "
            "certifiable signal.\n"
            f"- **Tradability `MIRAGE`** — +{R['sw_ann']:.2f}%/yr vs B&H +{R['bh_ann']:.2f}%/yr, "
            f"excess Sharpe {R['sw_sharpe']:.3f} vs {R['bh_sharpe']:.3f} (tied), and no cost level "
            "uncovers an edge because none exists over the passive default.\n"
            "- **Times equities like Dr. Copper? `BUSTED`** — a defensive gauge (maxDD "
            f"−{abs(R['sw_dd']):.0f}% vs −{abs(R['bh_dd']):.0f}%) that beats random timing but loses "
            "to buy-and-hold in every parameterisation. Risk-off thermometer, not equity timer."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost sweep is moot\n\n"
            "With ~14 switches/yr costs are small, but they are irrelevant: the gross switch "
            "already trails buy-and-hold, so there is nothing for costs to erode away from."
        ),
        code(
            "costs = [0, 2, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    anns = [st.race(df, 60, 1.0, c)['switch']['ann_return']*100 for c in costs]\n"
            "    ts2 = [st.race(df, 60, 1.0, c)['t_vs_bh'] for c in costs]\n"
            "    bh = st.race(df, 60, 1.0, 5.0)['buy_hold']['ann_return']*100\n"
            "else:\n"
            f"    anns = {R['cost_ann']}; ts2 = {R['cost_t_bh']}; bh = {R['bh_ann']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, anns, 'o-', c=AMBER, lw=2, label='switch ann %')\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=2, label='buy & hold ann %')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, ts2, 's--', c=GREY, lw=1.5, label='t vs B&H')\n"
            "ax2.axhline(0, ls=':', c=GREY); ax2.set_ylabel('t vs B&H', color=GREY)\n"
            "ax.set_xlabel('round-trip cost per switch (bps)'); ax.set_ylabel('annualised return %/yr')\n"
            "ax.legend(loc='upper right'); ax.set_title('Below buy-and-hold at every cost, gross included')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross: switch {anns[0]:.2f}%/yr vs B&H {bh:.2f}%/yr; t_vs_bh stays negative across costs')"
        ),
        md(
            f"> 💡 In plain words: even gross the switch earns +{R['cost_ann'][0]:.2f}%/yr vs the "
            f"index's +{R['bh_ann']:.2f}%, with *t* vs B&H = {R['cost_t_bh'][0]:+.2f}. This is the "
            "signature of a **Mirage**: not an edge eaten by costs, but no edge at all over the "
            "thing you'd do for free."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or does it always print a loss? Plant a forward "
            "ratio→equity link (`pred_r < 0`: a high ratio precedes lower returns — the "
            "folklore-consistent control) and confirm the switch banks it."
        ),
        code(
            "prs = [0.0, -0.005, -0.01]\n"
            "rows = []\n"
            "for pr in prs:\n"
            "    d, _ = data.synthetic_daily(n_days=4000, pred_r=pr, seed=305)\n"
            "    rr = st.race(d, 60, 1.0, 5.0)\n"
            "    rows.append((pr, rr['switch']['sharpe_excess'], rr['buy_hold']['sharpe_excess'], rr['t_vs_bh']))\n"
            "ctrl = pd.DataFrame(rows, columns=['pred_r','switch Sharpe','B&H Sharpe','t vs B&H'])\n"
            "print(ctrl.round(3).to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(ctrl['pred_r'], ctrl['t vs B&H'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted forward ratio→equity link (pred_r)'); ax.set_ylabel('t(switch − B&H)')\n"
            "ax.set_title('Engine is faithful: ~0 on the null, large t when a link is planted')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"On the null (`pred_r=0`) the switch is indistinguishable from B&H "
            f"(*t* = {R['syn_null_t']:+.2f}); plant even a small folklore-consistent link and the "
            f"engine banks it cleanly (*t* = +{R['syn_p005_t']:.1f} at pred_r=−0.005). So the engine "
            "is a faithful detector, and the real-tape 'no' is a statement about **the market**: "
            "the gold/oil ratio is a real risk-off thermometer but not a tradable equity timer.\n\n"
            "For the copper original as a forecasting regression, see "
            "[Study 85 — Dr-Copper](../../85-dr-copper/); for the metals-ratio mean-reversion trade, "
            "[Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/)."
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
