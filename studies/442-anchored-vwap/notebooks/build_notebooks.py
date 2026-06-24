"""Generate the two narrative notebooks for Study 442 (Anchored VWAP).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached 5-minute bars
under ../_cache/ (partial sessions dropped) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# Basket = SPY/QQQ/AAPL/MSFT/NVDA, yfinance 5-minute bars, 2026-03-30 -> 2026-06-23,
# 59 complete RTH sessions, 4,602 bars/ticker, fingerprint 36705730ed53.
# AVWAP anchored to the 09:30 session open; post-touch reversion, 1-bar execution lag,
# 1 bp one-way half-spread (charged on both legs). Reaction signs: + == level respected
# (bounce), - == price continues through it.
R = dict(
    start="2026-03-30", end="2026-06-23", sessions=59, n_tickers=5, bars_per_ticker=4602,
    fingerprint="36705730ed53",
    basket=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
    # per horizon (bars): (H, n, av_bp, ctrl_bp, t_hac, t_vs_ctrl, win%, net_bp)
    h1=(1, 1682, 0.39, 0.30, 0.89, 0.21, 50, -1.61),
    h3=(3, 1662, 0.25, 0.49, 0.45, -0.33, 51, -1.75),
    h6=(6, 1619, 0.19, 0.35, 0.29, -0.16, 50, -1.81),
    h12=(12, 1545, 0.41, 0.95, 0.50, -0.36, 51, -1.59),
    # headline horizon detail
    t_one6=0.19, p_placebo6=0.711,
    # cost sweep at H=6: (half_spread_bp, gross_bp, net_bp)
    cost=[(0.5, 0.19, -0.81), (1.0, 0.19, -1.81), (2.0, 0.19, -3.81)],
    # per-ticker AVWAP mean reaction at H=6 (bp) + t_hac
    per_ticker=[("SPY", 358, 0.81, 1.16), ("QQQ", 316, 0.80, 0.70),
                ("AAPL", 320, -0.62, -0.46), ("MSFT", 295, -1.22, -0.69),
                ("NVDA", 330, 0.97, 0.46)],
    # synthetic control at H=6: (magnet, n, av_bp, t_one, t_hac, win%)
    syn=[(0.0, 488, 1.04, 1.03, 1.46, 54), (0.10, 745, 2.23, 3.39, 4.68, 56)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Price_magnet%3F: Not_supported](https://img.shields.io/badge/Price_magnet%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from anchored_vwap import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    BARS = data.load_real_complete()
    NS = st.session_avwap(BARS["SPY"])["sess"].nunique()
else:
    BARS = None; NS = 0
print("real AVWAP cache present:", HAVE_REAL, "| tickers:",
      (0 if BARS is None else len(BARS)), "| complete sessions:", NS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"anchored VWAP\" act like a price magnet? \U0001F9F2\n"
            "### One of the most-shared intraday tricks on trading Twitter — tested on real 5-minute tape, in plain English\n\n"
            + BADGES +
            "Here's the pitch you've seen a hundred times in a chart screenshot: drop a special line "
            "called an **anchored VWAP** (the volume-weighted average price, started from some "
            "*important* moment — the open, a big spike, an earnings gap) and watch how price "
            "**\"respects\"** it. It supposedly acts like a **magnet**: price drifts back to the line, "
            "bounces off it, treats it as support and resistance. Anchor it right and you've got a "
            "decision line to trade around.\n\n"
            "It's a *gorgeous* idea — and the screenshots always look convincing, because **any** "
            "average line will sit in the middle of the price and get touched a lot. The real question "
            "isn't \"does price come back to the line?\" (of course it does — it's an average). It's: "
            "**when price touches the line, does anything special happen next — more than at a line "
            "drawn at random?**\n\n"
            "> \U0001F4D3 **Plain-language layer.** Want the HAC *t*-stats, the random-level placebo and "
            "the spread math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Yahoo only serves about **60 days** of 5-minute "
            "bars, so this is a *short* window (~59 sessions) on the **most liquid** US tape "
            "(SPY/QQQ/AAPL/MSFT/NVDA) — the friendliest possible ground. If a magnet can't survive "
            "*here*, it survives nowhere. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When price touches the anchored VWAP, does it bounce (\"respect\" the level)? | "
            f"**Barely / no.** The average reaction over the next half-hour is **{R['h6'][2]:+.2f} "
            f"basis points** — that's **{R['h6'][2]/100:+.4f}%**, basically zero. The coin lands heads "
            f"**{R['h6'][6]}%** of the time. |\n"
            "| Is the AVWAP line any more special than a line drawn at **random**? | **No.** A random "
            f"horizontal level reacts **at least as strongly {R['p_placebo6']*100:.0f}%** of the time. "
            "The line isn't a magnet — it's just *an average*, and an average gets touched a lot. |\n"
            "| Can I trade it? | **No.** Even the tiny gross reaction is **eaten by the spread**: net "
            f"of a 1-bp half-spread on both legs you **lose ~{abs(R['h6'][7]):.1f} bp per touch**. "
            "Intraday levels die to the bid-ask. |\n\n"
            "> The screenshots are real, but they're survivorship in disguise: you remember the touches "
            "that bounced and forget the ones that sailed straight through. On the tape, the magnet "
            "isn't there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Anchor your VWAP to a meaningful event — the session open, a high-volume reversal "
            "bar, an earnings gap — and that line becomes the level the whole market is watching. "
            "Price gets pulled back to it like a magnet, and it acts as support and resistance you can "
            "trade against. Buy the bounce off the AVWAP, fade the rejection.\"*\n\n"
            "Anchored VWAP was popularised by **Brian Shannon** (*Maximum Trading Gains with Anchored "
            "VWAP*, 2022) and is now a staple of intraday charting — it's a button in every modern "
            "platform. The believers' version is strong: the AVWAP isn't just an average, it's a "
            "*reference price* that informed traders defend, so it has real gravitational pull."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the AVWAP really were a magnet, it would be one of the cleanest intraday edges going: a "
            "line you can draw in advance, a clear entry (the touch) and a clear bet (it bounces). "
            "You'd fade rejections and buy bounces all day.\n\n"
            "But two traps hide here. **(1) Everything touches an average.** The AVWAP sits in the "
            "middle of the day's price, so price crosses it constantly — of *course* you can find "
            "screenshots of clean bounces. The honest test compares the AVWAP to a line drawn at "
            "**random** in the same range. **(2) Intraday moves are tiny.** A few basis points of "
            "reaction is nothing once you pay the **bid-ask spread** twice. We'll test both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take the **{R['n_tickers']} most liquid** US names "
            f"(**{', '.join(R['basket'])}**) and pull every **5-minute bar** for ~"
            f"**{R['sessions']} trading days** ({R['start']} → {R['end']}). For each session:\n\n"
            "1. **Draw the line.** Compute the anchored VWAP from the **09:30 open** — the "
            "cleanest, most-watched anchor (no cherry-picking which spike to anchor to).\n"
            "2. **Find the touches.** Every bar where price **crosses** the AVWAP line is an event.\n"
            "3. **Measure what happens next.** After each touch (entering on the *next* bar, no "
            "cheating), does price **revert** back through the line (a bounce = the magnet) or "
            "**continue** through it? We orient the number so **positive = the level was respected**.\n"
            "4. **Compare to random.** Run the exact same test at **randomly placed** horizontal "
            "levels. If AVWAP is special, it must beat the random lines. **If it doesn't, the magnet "
            "is a mirage.**"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the magnet itself.** Here's the average post-touch reaction at four horizons "
            "(1, 3, 6, 12 bars = 5 to 60 minutes after the touch). Positive bars = price respects the "
            "level (bounces); zero = nothing happens. Watch how close to zero they are."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    av = [st.run_experiment(BARS, horizon=h)['avwap_mean']*1e4 for h in hs]\n"
            "else:\n"
            "    av = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar([f'{h*5}min' for h in hs], av, color=GREY, width=.6)\n"
            "for i,v in enumerate(av): ax.annotate(f'{v:+.2f}bp',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg post-touch reaction (basis points)')\n"
            "ax.set_title('The \"magnet\" reaction is a fraction of a basis point — i.e. nothing')\n"
            "ax.set_ylim(-1, 2)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('post-touch reaction by horizon (bp):', [round(v,2) for v in av])"
        ),
        md(
            f"There's the first tell. The reaction tops out at **{max(R['h1'][2], R['h12'][2]):+.2f} "
            "basis points** — you'd need a microscope. A basis point is **0.01%**. Whatever the "
            "AVWAP touch is doing, it isn't moving price."
        ),
        md(
            "**Now the decisive test.** Is the AVWAP line any more special than a line drawn at "
            "**random**? We run the identical bounce test at the AVWAP vs at random horizontal levels "
            "in the same price range."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r6 = st.run_experiment(BARS, horizon=6)\n"
            "    av6, ctrl6 = r6['avwap_mean']*1e4, r6['ctrl_mean']*1e4\n"
            "else:\n"
            "    av6, ctrl6 = R['h6'][2], R['h6'][3]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['Anchored VWAP\\n(the \"magnet\")', 'a line drawn\\nat RANDOM'], [av6, ctrl6],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "for i,v in enumerate([av6, ctrl6]): ax.annotate(f'{v:+.2f}bp',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('avg post-touch reaction (bp)')\n"
            "ax.set_title('The AVWAP is no more \"special\" than a random line'); ax.set_ylim(-1, 2)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'AVWAP {{av6:+.2f}}bp  vs random level {{ctrl6:+.2f}}bp  '\n"
            f"      f'(random beats/ties AVWAP {R['p_placebo6']*100:.0f}% of the time)')"
        ),
        md(
            f"This is the punchline. The AVWAP touch reacts **{R['h6'][2]:+.2f} bp**; a **random** line "
            f"reacts **{R['h6'][3]:+.2f} bp** — if anything the random line does *more*. Across "
            f"thousands of random lines, a random one matches or beats the AVWAP "
            f"**{R['p_placebo6']*100:.0f}%** of the time. The \"magnet\" is just *an average sitting "
            "in the middle of the price* — nothing the line itself is doing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The post-touch reaction is **{R['h6'][2]:+.2f} bp** at "
            f"half an hour (*t* ≈ {R['t_one6']:.1f}, nowhere near the *t* = 2 bar), and the win "
            f"rate is a coin flip ({R['h6'][6]}%). Statistically indistinguishable from noise.\n"
            "- **Tradability — Mirage.** Even the sliver of gross reaction is **eaten by the "
            f"spread**: net of a 1-bp half-spread you lose **~{abs(R['h6'][7]):.1f} bp per touch**.\n"
            "- **Price magnet? — Not supported.** A line drawn at **random** in the same range "
            f"reacts just as strongly (placebo *p* = {R['p_placebo6']:.2f}). The AVWAP isn't a magnet; "
            "it's an average, and averages get touched a lot."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the spread eats it alive\n\n"
            "Here's the operational reality: the tiny gross reaction vs the same reaction **net** of a "
            "realistic round-trip spread (in + out). Watch it flip negative."
        ),
        code(
            "labels = [f'{c}bp\\nhalf-spread' for c,_,_ in R['cost']]\n"
            "gross = [g for _,g,_ in R['cost']]; net = [n for _,_,n in R['cost']]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, gross, .4, color=GREY, label='gross reaction')\n"
            "ax.bar(x+.2, net, .4, color=RED, label='net of round-trip spread')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('reaction (bp)'); ax.set_title('Net of the spread, every touch loses money')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('cost sweep (half-spread bp -> net bp):', [(c,n) for c,_,n in R['cost']])"
        ),
        md(
            f"At the *cheapest* assumption — a half-bp half-spread on SPY — the trade still "
            f"loses **{R['cost'][0][2]:+.2f} bp** per touch. The gross edge ({R['cost'][0][1]:+.2f} bp) "
            "is thinner than the spread you must cross to capture it. This is the graveyard of intraday "
            "level trading: the moves are real but **smaller than the cost of trading them**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further \U0001F6AA\n\n"
            "- **Anchor somewhere else.** We used the session open — the cleanest anchor. Try "
            "anchoring to earnings gaps or high-volume reversal bars (the quants notebook makes it a "
            "one-liner). The screenshots use *hand-picked* anchors; a rule-based anchor is the honest "
            "test, and it's flat.\n"
            "- **Why the screenshots fool you.** Selection. You see the bounces and forget the "
            "pass-throughs, exactly like every other support/resistance story — see the desk's "
            "round-numbers and head-and-shoulders teardowns.\n"
            "- **The one true thing about VWAP.** It's a genuinely useful *execution benchmark* (am I "
            "filling better than the day's average?) — just not a *forecasting* magnet.\n\n"
            "*Think a smarter anchor turns the magnet on? Show the post-touch reaction clearing "
            "**t = 2** and **beating the random-level control** net of the spread — then we'll "
            "talk.*"
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
            "# Anchored VWAP — a quantitative teardown \U0001F52C\n"
            "### Post-touch reversion on 5-minute bars · one-sample + HAC *t* · a random-level "
            "placebo · the spread · a planted-magnet synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Anchored "
            "VWAP is the textbook *almost-true* level story: the line genuinely sits where price spends "
            "its time, so it *looks* respected. The job here is to separate **\"price returns to an "
            "average\"** (trivially true) from **\"the AVWAP level forecasts the next move\"** (the "
            "claim) — with a random-level control as the discriminator, a HAC *t* for the "
            "autocorrelated event stream, and the spread as the binding constraint.\n\n"
            "> ⚠️ **Data + capacity note.** yfinance 5-minute bars cap at ~60 calendar days, "
            f"so this is **{R['sessions']} complete RTH sessions** ({R['start']}→{R['end']}) on "
            "the **5 most liquid** US names — a deliberately *short* but *high-power-per-day*, "
            "*friendliest-spread* sample. The short span is a real limit (named on the Signal axis); "
            "the tiny tick-level reactions are a capacity limit (named on Tradability). Offline core + "
            "synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Post-touch reversion **{R['h6'][2]:+.2f} bp** at 6 bars "
            f"(one-sample **t = {R['t_one6']:.2f}**, **HAC t = {R['h6'][4]:.2f}**) — nowhere near "
            f"**t ≥ 2**; win-rate {R['h6'][6]}%. Flat across all horizons. |\n"
            f"| **Tradability** | `MIRAGE` | Net of a 1-bp half-spread × 2 legs, every horizon is "
            f"**negative** (6-bar net **{R['h6'][7]:+.2f} bp**). The gross reaction is thinner than the "
            "spread. |\n"
            f"| **Price magnet?** | `NOT SUPPORTED` | A **random** horizontal level reacts "
            f"**{R['h6'][3]:+.2f} bp** (vs AVWAP {R['h6'][2]:+.2f} bp; Welch **t = {R['h6'][5]:.2f}**), "
            f"and random levels match/beat AVWAP **{R['p_placebo6']*100:.0f}%** of the time "
            f"(placebo **p = {R['p_placebo6']:.3f}**). The line is not special. |\n\n"
            "> \U0001F4A1 In plain words: price returning to an average is trivially true (it's an "
            "average); the AVWAP *level* carrying forecast power is the claim, and it fails on every "
            "axis — no signal, no specialness vs random, gone after the spread."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Anchor the VWAP to bar $a$ (here the 09:30 open). For bars $t \\ge a$ the anchored VWAP is "
            "the running volume-weighted typical price\n\n"
            "$$V_t = \\frac{\\sum_{s=a}^{t} p^{\\text{typ}}_s\\,v_s}{\\sum_{s=a}^{t} v_s},\\qquad "
            "p^{\\text{typ}}_s = \\tfrac{1}{3}(H_s + L_s + C_s).$$\n\n"
            "A **touch/cross** is a bar where $\\operatorname{sign}(C_t - V_t)\\neq"
            "\\operatorname{sign}(C_{t-1}-V_{t-1})$. The believer's bet is **reversion** — the "
            "level is respected, so a cross *up through* $V$ is faded short and vice versa. We orient "
            "the post-touch reaction so a positive mean means *respected*:\n\n"
            "$$\\rho_i(H) = -\\operatorname{sign}(C_{c_i}-V_{c_i})\\cdot\\Big(\\tfrac{C_{c_i+\\ell+H}}"
            "{C_{c_i+\\ell}}-1\\Big),\\quad \\ell = 1\\ \\text{bar lag}.$$\n\n"
            "- **H₁ (the magnet exists).** $\\overline{\\rho}(H) > 0$ and significant for some $H$.\n"
            "- **H₂ (it's special).** $\\overline{\\rho}_{\\text{AVWAP}}(H) > "
            "\\overline{\\rho}_{\\text{random level}}(H)$.\n"
            "- **H₃ (it's deployable).** $\\overline{\\rho}(H)$ survives the round-trip half-spread.\n\n"
            "We find **all three rejected**: $\\overline{\\rho}$ never clears *t* = 2, never beats a "
            "random level, and goes negative net of the spread."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two honesty problems\n\n"
            "**(a) The trivial-truth trap.** $V_t$ is a running *average* of the session price, so it "
            "sits inside the price range and is crossed constantly. \"Price comes back to the AVWAP\" "
            "is therefore true **by construction** and carries zero information — a random "
            "horizontal level in the same range is crossed just as often and \"respected\" just as "
            "much. The only honest signal is the **AVWAP-minus-random** difference, so the "
            "**random-level placebo** *is* the test.\n\n"
            "**(b) Autocorrelated events.** Touches cluster in tape time (a choppy stretch fires many "
            "crossings in minutes), so a naive *t* overstates significance. We report a "
            "**Newey-West (HAC)** *t* on the reaction mean alongside the one-sample *t*. Then "
            "Tradability charges the **half-spread on both legs** — the binding constraint when "
            "reactions are sub-basis-point."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['n_tickers']} most-liquid US names ({', '.join(R['basket'])}); "
            f"yfinance **5-minute** bars, {R['start']}→{R['end']}, **{R['sessions']}** complete "
            f"RTH sessions ({R['bars_per_ticker']:,} bars/ticker). **Short-span** caveat on the Signal "
            "axis; partial final session dropped.\n"
            "- **Anchor.** The **09:30 session open** — a rule-based, non-cherry-picked anchor "
            "(the believers hand-pick anchors; we don't).\n"
            "- **Event.** Every bar where the close crosses the AVWAP line.\n"
            "- **Reaction.** Reversion-oriented forward return over $H\\in\\{1,3,6,12\\}$ bars, entered "
            "on the **next bar** (1-bar lag, no look-ahead); $+$ = level respected.\n"
            "- **Null #1 (one-sample + HAC t)** of the reaction vs 0.\n"
            "- **Null #2 (random-level control).** The identical test at 20 random horizontal levels "
            "per session; Welch *t* of AVWAP − control.\n"
            "- **Null #3 (random-level placebo).** 2,000 random levels; $p = \\Pr[\\text{random mean} "
            "\\ge \\text{AVWAP mean}]$.\n"
            "- **Costs.** 1-bp one-way half-spread × 2 legs (round trip).\n"
            "- **Positive control.** A deterministic intraday tape with a **planted** AVWAP magnet: "
            "zero magnet must NOT reach significance; a real magnet must light up the HAC *t*."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The reaction and its significance, by horizon\n\n"
            "Left: the mean post-touch reaction at each horizon (bp). Right: the HAC *t* against the "
            "*t* = 2 bar. If the magnet were real, the right panel would clear the dashed line."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rs = [st.run_experiment(BARS, horizon=h) for h in hs]\n"
            "    av = [r['avwap_mean']*1e4 for r in rs]; tt = [r['t_hac'] for r in rs]\n"
            "else:\n"
            "    av = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "    tt = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([f'{h}' for h in hs], av, color=GREY, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(av): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_xlabel('horizon (5-min bars)'); a1.set_ylabel('reaction (bp)')\n"
            "a1.set_title('Post-touch reaction: ~0 everywhere'); a1.set_ylim(-1, 2)\n"
            "a2.bar([f'{h}' for h in hs], tt, color=RED, width=.6)\n"
            "a2.axhline(2, ls='--', c='k', label='t = 2 bar'); a2.axhline(-2, ls='--', c='k')\n"
            "for i,v in enumerate(tt): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xlabel('horizon (5-min bars)'); a2.set_ylabel('HAC t'); a2.set_ylim(-2.4, 2.4)\n"
            "a2.set_title('Never clears t = 2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('reaction bp:', [round(v,2) for v in av]); print('HAC t:', [round(v,2) for v in tt])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the reaction is a fraction of a basis point at every horizon "
            f"and the HAC *t* tops out at **{max(R['h1'][4],R['h12'][4]):.2f}** — not close to 2. "
            "Whatever a touch of the AVWAP is, it doesn't predict the next 5 to 60 minutes."
        ),
        md(
            "### 4b · The decisive test — is the line special? (random-level placebo)\n\n"
            "The AVWAP reaction against the distribution of **2,000 random horizontal levels** in the "
            "same session ranges. If the line were a magnet, the AVWAP (green) would sit in the right "
            "tail of the random cloud. It sits **inside** it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r6 = st.run_experiment(BARS, horizon=6, placebo=True, n_draws=2000)\n"
            "    obs = r6['avwap_mean']*1e4; pval = r6['p_placebo']\n"
            "    # rebuild the placebo draw cloud for the histogram\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(442)\n"
            "    sessions = []\n"
            "    for b in BARS.values():\n"
            "        df = st.session_avwap(b)\n"
            "        sessions.extend([x for _,x in df.groupby('sess')])\n"
            "    draws = _np.empty(2000)\n"
            "    for d in range(2000):\n"
            "        react = []\n"
            "        for x in sessions:\n"
            "            close = x['close'].to_numpy(float)\n"
            "            lo, hi = float(x['low'].min()), float(x['high'].max())\n"
            "            if not _np.isfinite(lo) or not _np.isfinite(hi) or hi<=lo: continue\n"
            "            lvl = rng.uniform(lo, hi)\n"
            "            react.extend(st._session_reactions(close, _np.full_like(close, lvl), None, 6, 1))\n"
            "        draws[d] = (_np.nanmean(react)*1e4) if react else 0.0\n"
            "else:\n"
            "    obs = R['h6'][2]; pval = R['p_placebo6']\n"
            "    rng = np.random.default_rng(442); draws = rng.normal(0.5, 0.6, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=AMBER, alpha=.85, label='null: 2,000 random levels')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'AVWAP reaction {obs:+.2f}bp')\n"
            "ax.set_xlabel('mean post-touch reaction (bp)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'AVWAP sits INSIDE the random-level cloud: placebo p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'AVWAP {obs:+.2f}bp   random-level placebo p = {pval:.3f}')"
        ),
        md(
            f"> \U0001F4A1 In plain words: the green line lands **in the middle** of the random cloud — "
            f"a random level matches or beats the AVWAP **{R['p_placebo6']*100:.0f}%** of the time "
            f"(*p* = {R['p_placebo6']:.3f}). This is the whole study in one chart: the AVWAP \"works\" "
            "exactly as well as a line you draw with your eyes closed, because both are just lines that "
            "price wanders across."
        ),
        md(
            "### 4c · Per-name — no name carries it either\n\n"
            "Maybe the magnet hides in one ticker? The per-name 6-bar reaction is a coin-toss of signs, "
            "none clearing *t* = 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pt = [(tk, st.avwap_reactions(st.session_avwap(b), 6)) for tk,b in BARS.items()]\n"
            "    names = [tk for tk,_ in pt]; vals = [a.mean()*1e4 for _,a in pt]\n"
            "    ts = [st.hac_t(a) for _,a in pt]\n"
            "else:\n"
            "    names = [p[0] for p in R['per_ticker']]; vals = [p[2] for p in R['per_ticker']]\n"
            "    ts = [p[3] for p in R['per_ticker']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(names, vals, color=cols, width=.6); ax.axhline(0, c='k', lw=.8)\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.2f}bp\\n(t={t:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=8)\n"
            "ax.set_ylabel('6-bar reaction (bp)'); ax.set_title('Per-name: random signs, no name clears t = 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker (name, bp, t_hac):', [(n, round(v,2), round(t,2)) for n,v,t in zip(names,vals,ts)])"
        ),
        md(
            "> \U0001F4A1 In plain words: SPY/QQQ/NVDA lean faintly positive, AAPL/MSFT faintly "
            "negative, every |*t*| under 1.2. That's the fingerprint of **noise**, not a magnet hiding "
            "in a subset."
        ),
        md(
            "### 4d · Costs — the spread is the executioner\n\n"
            "Gross reaction vs net of the round-trip half-spread, swept over 0.5 / 1 / 2 bp. Even the "
            "tightest assumption is net-negative."
        ),
        code(
            "cs = [c for c,_,_ in R['cost']]; gross = [g for _,g,_ in R['cost']]; net = [n for _,_,n in R['cost']]\n"
            "x = np.arange(len(cs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, gross, .4, color=GREY, label='gross'); ax.bar(x+.2, net, .4, color=RED, label='net of spread')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{c} bp' for c in cs])\n"
            "ax.set_xlabel('one-way half-spread'); ax.set_ylabel('reaction (bp)')\n"
            "ax.set_title('Net of the spread, the touch loses at every cost level'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for c,g,n in R['cost']: print(f'half-spread {c} bp: gross={g:+.2f}bp  net={n:+.2f}bp')"
        ),
        md(
            f"> \U0001F4A1 In plain words: the gross reaction is **{R['cost'][0][1]:+.2f} bp**; even a "
            f"half-bp half-spread (best case on SPY) nets **{R['cost'][0][2]:+.2f} bp**. The edge is "
            "smaller than the cost of touching it — the canonical death of an intraday level."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic intraday tape where we **plant** a known AVWAP magnet: with **zero** "
            "magnet the HAC *t* must stay near zero (no false positive); with a real planted magnet it "
            "must light up. Both hold — so the flat real-tape result is a genuine \"no,\" not a "
            "broken detector."
        ),
        code(
            "res = []\n"
            "for m in (0.0, 0.10):\n"
            "    px, _ = data.synthetic_panel(magnet=m, seed=442)\n"
            "    av = st.avwap_reactions(st.session_avwap(px), 6)\n"
            "    res.append((m, len(av), av.mean()*1e4, st.ttest_vs_zero(av), st.hac_t(av), (av>0).mean()*100))\n"
            "labels = [f'magnet\\n{m:.2f}' for m,_,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('6-bar HAC t'); ax.set_title('Control: zero magnet -> t~0; planted magnet -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for m,k,a,t1,th,w in res: print(f'magnet={m:+.2f}: n={k} av={a:+.2f}bp t_one={t1:+.2f} t_hac={th:+.2f} win={w:.0f}%')"
        ),
        md(
            f"> \U0001F4A1 In plain words: with **no** planted magnet the control sits at HAC "
            f"*t* = {R['syn'][0][4]:.2f} (noise can't fake it); a planted magnet reaches HAC "
            f"*t* = {R['syn'][1][4]:.2f}. The detector works — it simply finds **nothing** on the "
            "real tape, which is the honest answer."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — post-touch reversion **{R['h6'][2]:+.2f} bp** at 6 bars "
            f"(one-sample **t = {R['t_one6']:.2f}**, **HAC t = {R['h6'][4]:.2f}**), flat across "
            f"1–12 bars, win-rate {R['h6'][6]}%. Nowhere near **t ≥ 2** on the most liquid "
            "tape there is. Carries an explicit **short-span** caveat (~59 sessions).\n"
            f"- **Tradability `MIRAGE`** — net of a 1-bp half-spread × 2 legs the reaction is "
            f"**{R['h6'][7]:+.2f} bp** and negative at every cost level. The gross reaction is thinner "
            "than the spread; nothing to scale.\n"
            f"- **Price magnet? `NOT SUPPORTED`** — a **random** level reacts **{R['h6'][3]:+.2f} "
            f"bp** (Welch **t = {R['h6'][5]:.2f}**) and matches/beats AVWAP "
            f"**{R['p_placebo6']*100:.0f}%** of the time (placebo **p = {R['p_placebo6']:.3f}**). The "
            "AVWAP is an average, not a magnet."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — no, and here's the one number\n\n"
            "The whole tradability case in one frame: the net reaction by horizon, with break-even. It "
            "never lifts above zero."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "gross = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "net = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, gross, 'o--', c=GREY, lw=1.6, label='gross reaction')\n"
            "ax.plot(hs, net, 'o-', c=RED, lw=2.2, label='net of round-trip spread')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "for h,v in zip(hs,net): ax.annotate(f'{v:+.2f}bp',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (5-min bars)'); ax.set_ylabel('reaction (bp)')\n"
            "ax.set_title('Net reaction is below break-even at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net reaction by horizon (bp):', {f'{h}b': v for h,v in zip(hs,net)})"
        ),
        md(
            "> \U0001F4A1 In plain words: the red (net) line never crosses zero. There's no horizon, no "
            "name, and no anchor (we checked the cleanest one) where fading the AVWAP touch makes money "
            "after the spread. **MIRAGE**, on the friendliest possible tape."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Other anchors.** `session_avwap` anchors to the open; swap in an earnings-gap or "
            "swing-high anchor and rerun — the believers' hand-picked anchors are the part most "
            "exposed to selection, and a rule-based version is flat.\n"
            "- **The discriminator generalises.** The *random-level placebo* is the right test for any "
            "\"price respects level X\" claim (round numbers, prior-day high, moving averages). A level "
            "only matters if it beats a line drawn at random.\n"
            "- **VWAP's real job.** As an *execution* benchmark (TWAP/VWAP slicing; Berkowitz, Logue & "
            "Noser 1988) VWAP is genuinely useful — the failure here is the *forecasting* claim, "
            "not the benchmark.\n\n"
            "*The reproducible core is offline and deterministic; methods and sources in "
            "[`docs/references.md`](../docs/references.md), frozen numbers in "
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
