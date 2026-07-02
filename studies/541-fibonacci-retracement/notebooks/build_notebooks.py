"""Generate the two narrative notebooks for Study 541 (Fibonacci-Retracement).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached daily OHLC under
../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; 8 tapes;
# 5% ZigZag, 20-day hold; combined tape fp 11844738632b).
R = dict(
    fp_combined="11844738632b",
    n_tapes=8, asof="2026-06-30",
    fib_n=225, fib_bps=-15.0, fib_wr=0.587, fib_t=-0.27, fib_hac=-0.23, fib_net=-17.0,
    plc_n=197, plc_bps=-79.2, plc_wr=0.508, plc_t=-1.38,
    edge_bps=64.2, edge_t=0.80, coin_p=0.61,
    # robustness grid: (zigzag_pct, horizon, fib_n, fib_t, edge_t)
    grid=[
        (0.03, 10, 505, -0.68, 1.53), (0.03, 20, 505, -0.71, 0.84), (0.03, 40, 505, -1.01, 0.45),
        (0.05, 10, 225, -0.81, 0.89), (0.05, 20, 225, -0.27, 0.80), (0.05, 40, 225, 0.65, 0.18),
        (0.07, 10, 139, -0.53, 0.18), (0.07, 20, 139, -0.12, -0.55), (0.07, 40, 138, 0.47, -1.00),
    ],
    # per-tape (pct=0.05 h=20): (tk, fib_n, fib_bps, edge_bps)
    per_tape=[("SPY", 22, -24, -36), ("QQQ", 24, 442, 768), ("DIA", 21, -108, -215),
              ("IWM", 19, -228, -478), ("^GSPC", 62, -69, -15), ("^IXIC", 47, -107, -55),
              ("^DJI", 20, 273, 347), ("GLD", 10, -301, -21)],
    # synthetic positive control (swing panel, 25 seeds): (fib_pull, mean edge_t)
    ctrl=[(0.0, -0.01), (0.02, 4.58), (0.04, 9.17), (0.06, 13.76)],
    tape_null_edge_t=-0.02,
)


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

from fibonacci_retracement import data, strategy as st

ASOF = "2026-06-30"
PCT, HORIZON, TOL = 0.05, 20, 0.025

def load_tapes():
    \"\"\"Cache-first basket of daily OHLC tapes, trimmed to the as-of date (empty offline).\"\"\"
    bk = data.load_basket(fetch=False)
    return {tk: b.loc[:ASOF] for tk, b in bk.items()}

TAPES = load_tapes()
HAVE_REAL = len(TAPES) > 0
print("real tapes present:", HAVE_REAL, "" if not HAVE_REAL else f"({len(TAPES)} tapes)")

def pooled_arms(tapes, pct=PCT, horizon=HORIZON, tol=TOL):
    \"\"\"Pool the Fib-arm and placebo-arm gross reversal returns across all tapes.\"\"\"
    FA, PA = [], []
    for b in tapes.values():
        close = b["close"].to_numpy(float)
        piv = st.zigzag(close, pct); rd = st.swing_retracements(piv)
        lf = st.run_trades(close, st.bin_touches(rd, data.FIB_LEVELS, tol), horizon, 0.0)
        lp = st.run_trades(close, st.bin_touches(rd, data.PLACEBO_LEVELS, tol), horizon, 0.0)
        if len(lf): FA.append(lf["ret_gross"].to_numpy())
        if len(lp): PA.append(lp["ret_gross"].to_numpy())
    fa = np.concatenate(FA) if FA else np.array([])
    pa = np.concatenate(PA) if PA else np.array([])
    return fa, pa
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Fibonacci Retracements — does a bounce really stop at 61.8%? 🌀\n"
            "### Drawing the 'magic' levels on real charts, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Fib_levels_special%3F: Not_Supported](https://img.shields.io/badge/Fib_levels_special%3F-Not_Supported-8b949e?style=flat-square)\n\n"
            "Open any charting app and you'll find a **Fibonacci retracement** tool. You draw it from "
            "a swing low to a swing high, and it paints horizontal lines at **23.6%, 38.2%, 50% and "
            "61.8%** of the move. The folklore: after a rally, a pullback tends to *stop and turn* "
            "right at one of these lines — especially 38.2% and 61.8% (the 'golden ratio'). Traders "
            "buy the bounce there.\n\n"
            "It's a beautiful story rooted in a real number sequence. But is it *true* on real "
            "prices? We test it the honest way: mark the swings, see how deep each pullback goes, and "
            "ask whether pullbacks that land on a Fibonacci level bounce any harder than pullbacks "
            "that land on some **arbitrary** level like 44%.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "robustness grid? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do pullbacks bounce *at* the Fib levels? | **No.** Betting the trend resumes at a "
            f"Fibonacci-depth pullback earned **{R['fib_bps']:.0f} bps** over 33 years and 8 tapes "
            f"(*t* {R['fib_t']}). That's nothing. |\n"
            f"| Are the Fib levels *special* vs random levels? | **No.** Arbitrary non-Fibonacci "
            f"fractions did just as well — the Fib-minus-placebo edge is only *t* {R['edge_t']}, and "
            "it flips sign depending on the settings. |\n"
            f"| Is it just luck either way? | The bounce bet matches a **coin** flipped at the same "
            f"swings (*p* {R['coin_p']}). The level adds no information. |\n"
            "| Could you trade it? | **No** — flat-to-negative before costs, and the per-chart "
            "results are all over the place. |\n\n"
            "> The Fibonacci sequence is real and lovely. The claim that price *obeys* it at "
            "retracement levels is not — a bounce at 61.8% is no more likely than a bounce at 44%."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"After a swing, price retraces and reverses at a Fibonacci fraction of the move — "
            "23.6%, 38.2%, 50%, 61.8%.\"* — technical-analysis folklore (Elliott/Prechter; Murphy's "
            "*Technical Analysis of the Financial Markets*).\n\n"
            "The 38.2% and 61.8% levels come from ratios of the Fibonacci sequence (each number over "
            "the next, or the one after). 61.8% is the 'golden ratio'. The idea is that markets — "
            "being made of human crowds — pull back in these natural proportions and then resume the "
            "trend. It's the backbone of a lot of chart trading."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it's true, it's a free map of where to buy the dip. If it's not, it's a self-imposed "
            "pattern — the eye finding meaning in noise, helped by the fact that *some* level is "
            "always nearby. The desk has already tested the wave-count version "
            "([445 Elliott Wave](../../445-elliott-wave/)) and round *price* levels "
            "([93 Round Numbers](../../93-round-numbers/)); here we go straight at the bare "
            "**retracement level** — and, crucially, we race it against arbitrary levels."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Mark the swings.** A 'ZigZag' tool flags each swing high and low once price has "
            "reversed 5% — the same swing a chartist would eyeball.\n"
            "2. **Measure each pullback.** How far back did it retrace the prior swing? 40%? 62%?\n"
            "3. **Bet the bounce.** At the pullback low, bet the trend resumes. Hold 20 days.\n"
            "4. **The key trick — a placebo.** Compare pullbacks that landed on a **Fibonacci** depth "
            "against pullbacks that landed on an **arbitrary** depth (44%, 56%) in the *same* range. "
            "If Fibonacci is magic, the Fib bounces must be better. If not, they'll match.\n\n"
            "*Timing:* we only use the pullback once it's confirmed (price has turned), and enter the "
            "next day — no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Did betting the bounce at a Fibonacci level make money?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fa, pa = pooled_arms(TAPES)\n"
            "    fib_bps, fib_t = fa.mean()*1e4, st.ttest_vs_zero(fa)\n"
            "    plc_bps = pa.mean()*1e4\n"
            "    banner = f'REAL tapes ({len(TAPES)} indices/ETFs, 5% ZigZag, 20-day hold)'\n"
            "else:\n"
            f"    fib_bps, fib_t, plc_bps = {R['fib_bps']}, {R['fib_t']}, {R['plc_bps']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['Fibonacci\\nlevels', 'arbitrary\\nlevels (placebo)'], [fib_bps, plc_bps],\n"
            "       color=[GREY, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg reversal return (bps)')\n"
            "ax.set_title(f'Bouncing at a Fib level earned {fib_bps:+.0f} bps — basically nothing')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Fibonacci arm {fib_bps:+.0f} bps (t {fib_t:+.2f})  |  placebo arm {plc_bps:+.0f} bps')"
        ),
        md(
            f"There it is. Betting the bounce at a Fibonacci level earned **{R['fib_bps']:.0f} bps** "
            f"per trade (*t* {R['fib_t']}) — indistinguishable from zero. And the *arbitrary* levels "
            f"did about the same. The 'magic' 38.2% / 61.8% lines aren't special."
        ),
        md(
            "**Is Fibonacci at least *better* than a random level?** Look at the gap between the two "
            "arms across different chart settings (how sensitive the ZigZag is, how long you hold):"
        ),
        code(
            "grid = " + repr(R["grid"]) + "\n"
            "labs = [f'{int(p*100)}%/{h}d' for (p,h,_,_,_) in grid]\n"
            "edge = [e for (_,_,_,_,e) in grid]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if abs(x)>=2 else GREY for x in edge]\n"
            "ax.bar(range(len(edge)), edge, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=GREEN, lw=1)\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(range(len(edge))); ax.set_xticklabels(labs, rotation=40, ha='right')\n"
            "ax.set_ylabel('Fib-minus-placebo edge (t-stat)')\n"
            "ax.set_title('The Fib \\u2212 placebo edge: never near \\u00b12, and it flips sign')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge t-stats:', edge)"
        ),
        md(
            "The edge of Fibonacci *over* arbitrary levels wanders between about **+1.5 and −1.0** "
            "depending on the settings — never close to the ±2 you'd need to believe it, and it "
            "doesn't even keep the same sign. That's the fingerprint of **noise**, not a real effect."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The bounce bet at a Fib level earns nothing (*t* {R['fib_t']}) and "
            "isn't better than a random level.\n"
            "- **Tradability — Mirage.** Flat-to-negative before costs; per-chart results are pure "
            "scatter.\n"
            "- **Fib levels special? — Not supported.** Arbitrary fractions do just as well; a coin "
            "at the same swings matches the Fibonacci arm."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even ignoring the missing edge, the per-tape results swing from a big *loss* on "
            "small-caps to a big *gain* on the Nasdaq-100 with no rhyme or reason — that's what you "
            "get averaging noise. And you'd pay a spread every time you place the trade, turning a "
            "flat bet into a small loss."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The wave version.** [445 Elliott Wave](../../445-elliott-wave/) tests the full "
            "'buy wave 3' count that Fibonacci ratios are part of — also a null.\n"
            "- **Other 'levels'.** [93 Round Numbers](../../93-round-numbers/) and the pivot-point "
            "studies ([440](../../440-pivot-points/)) ask the same 'is this line special?' question.\n\n"
            "*Think Fibonacci works and we missed it? Fork this, add intraday or single-name tapes, "
            "and show the Fib arm beating the arbitrary-fraction placebo at t \\u2265 2.*"
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
            "# Fibonacci Retracement — a quantitative teardown 🔬\n"
            "### 5% ZigZag swings · realised-retracement binning · Fib-vs-placebo two-sample *t* · same-bars coin placebo · threshold\\u00d7horizon grid · costs · two synthetic controls\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Fib_levels_special%3F: Not_Supported](https://img.shields.io/badge/Fib_levels_special%3F-Not_Supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We mark swings with a ZigZag, "
            "measure each pullback's realised retracement, bet the trend resumes, and race the "
            "**Fibonacci** depths against a **placebo** of arbitrary fractions in the same band.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily OHLC for 8 broad "
            f"index/ETF tapes (as-of {R['asof']}, combined fp `{R['fp_combined']}`); the offline core "
            "and controls run on deterministic synthetic worlds. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Fib arm **{R['fib_bps']:.0f} bps** (one-sample *t* "
            f"**{R['fib_t']}**, HAC {R['fib_hac']}); Fib\\u2212placebo edge *t* **{R['edge_t']}**; "
            "sign flips across the grid; coin *p* "
            f"{R['coin_p']}. |\n"
            f"| **Tradability** | `MIRAGE` | Gross {R['fib_bps']:.0f} bps \\u2192 net "
            f"{R['fib_net']:.0f} bps (1 bp/side); per-tape edge \\u2212478..+768 bps (noise). |\n"
            "| **Fib levels special?** | `NOT SUPPORTED` | Arbitrary interleaved fractions match; the "
            f"engine catches a *planted* level-effect at edge *t* up to +13.8 (25 seeds). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on real "
            "index/ETF swings the Fibonacci retracement levels carry no reversal edge and are no "
            "better than arbitrary levels."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a swing run P0 \\u2192 P1 and pull back to P2. The realised retracement is "
            "$\\rho = (P_1-P_2)/(P_1-P_0)$. Let $r_+$ be the forward return of betting the trend "
            "resumes at P2.\n\n"
            "- **H\\u2081 (reversal at Fib).** $\\mathbb{E}[r_+ \\mid \\rho \\text{ near a Fib "
            "level}] > 0$ at *t* \\u2265 2.\n"
            "- **H\\u2082 (Fib is special).** $\\mathbb{E}[r_+ \\mid \\rho \\text{ near Fib}] - "
            "\\mathbb{E}[r_+ \\mid \\rho \\text{ near an arbitrary level}] > 0$ at *t* \\u2265 2.\n"
            "- **H\\u2083 (robustness).** The sign/size of H\\u2081/H\\u2082 is stable across ZigZag "
            "threshold, horizon and tolerance.\n"
            "- **H\\u2084 (beats a coin).** The Fib arm beats a same-bars random-direction coin.\n\n"
            "We find **H\\u2081 not rejected** in the null direction (Fib arm *t* "
            f"{R['fib_t']}), **H\\u2082 rejected** (edge *t* {R['edge_t']}), **H\\u2083 rejected** "
            "(sign flips), **H\\u2084 rejected** (coin *p* "
            f"{R['coin_p']})."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? \\u2014 the placebo is the whole point\n\n"
            "The trap in testing chart levels is that *some* level is always near any pullback, and "
            "markets drift up, so a 'bounce' bet looks alive. The honest control is a **placebo of "
            "non-Fibonacci fractions interleaved in the same depth band** (0.31 / 0.44 / 0.56 between "
            "the Fib depths). Matching the depth range strips out the drift-and-mean-reversion "
            "baseline, leaving only the question that matters: *are the Fibonacci fractions "
            "themselves special?* Same spirit as [93 Round Numbers](../../93-round-numbers/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know \\u2014 the protocol\n\n"
            "- **Swings.** 5% ZigZag pivots (look-ahead-free confirmation bar).\n"
            "- **Retracement.** $\\rho = (P_1-P_2)/(P_1-P_0)$ per pullback, kept only for genuine "
            "retracements $0<\\rho<1$ of an impulse-expansion leg (one $\\rho$ per pullback).\n"
            "- **Bet.** Trend resumes at P2; enter one bar after P2 is confirmed; hold 20 days.\n"
            "- **Arms.** Fib bin (near 0.236/0.382/0.5/0.618) vs placebo bin (near 0.31/0.44/0.56), "
            "$\\pm0.025$.\n"
            "- **Inference.** One-sample & HAC *t* per arm; Welch two-sample *t* for the edge; a "
            "same-bars **coin placebo**.\n"
            "- **Robustness.** ZigZag $\\in\\{3,5,7\\}\\%$ \\u00d7 horizon $\\in\\{10,20,40\\}$ "
            "\\u00d7 tolerance.\n"
            "- **Frictions.** 1 bp/side one-way, round trip.\n"
            "- **Controls.** A planted-level-effect swing panel (engine must catch it) and a genuine "
            "price-path null (engine must stay flat), each averaged over \\u2265 20 seeds.\n\n"
            "Timing: $\\rho$ is only known once P2 is confirmed (price has turned); we enter the next "
            "bar. No look-ahead. The same-bar-after-confirmation fill is the documented convention."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a \\u00b7 The head-to-head \\u2014 H\\u2081 & H\\u2082\n\n"
            "Pooled Fib-arm vs placebo-arm reversal returns, with the one-sample and two-sample *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fa, pa = pooled_arms(TAPES)\n"
            "    fib_bps, fib_t, fib_hac = fa.mean()*1e4, st.ttest_vs_zero(fa), st.hac_t(fa)\n"
            "    plc_bps, plc_t = pa.mean()*1e4, st.ttest_vs_zero(pa)\n"
            "    edge_bps, edge_t = (fa.mean()-pa.mean())*1e4, st.two_sample_t(fa, pa)\n"
            "    nfib, nplc = len(fa), len(pa)\n"
            "else:\n"
            f"    fib_bps, fib_t, fib_hac = {R['fib_bps']}, {R['fib_t']}, {R['fib_hac']}\n"
            f"    plc_bps, plc_t = {R['plc_bps']}, {R['plc_t']}\n"
            f"    edge_bps, edge_t = {R['edge_bps']}, {R['edge_t']}\n"
            f"    nfib, nplc = {R['fib_n']}, {R['plc_n']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['Fibonacci', 'placebo'], [fib_bps, plc_bps], color=[GREY, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('reversal return (bps)')\n"
            "ax.set_title(f'Fib {fib_bps:+.0f} bps (t {fib_t:+.2f}) vs placebo {plc_bps:+.0f} bps  |  edge t {edge_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Fib n={nfib} {fib_bps:+.0f} bps t {fib_t:+.2f} HAC {fib_hac:+.2f} | plc n={nplc} {plc_bps:+.0f} bps | edge {edge_bps:+.0f} bps t {edge_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **H\\u2081 not rejected in the null direction** \\u2014 the Fib arm "
            f"earns {R['fib_bps']:.0f} bps (*t* {R['fib_t']}). **H\\u2082 rejected** \\u2014 the "
            f"Fib\\u2212placebo edge is only *t* {R['edge_t']}. Fibonacci depths are not special."
        ),
        md(
            "### 4b \\u00b7 Beats a coin? \\u2014 H\\u2084\n\n"
            "Keep the same Fibonacci swing bars, randomise the direction with a fair coin many times: "
            "does the reversal bet beat the coin?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fa, _ = pooled_arms(TAPES)\n"
            "    obs = fa.mean(); mags = np.abs(fa)\n"
            "    rng = np.random.default_rng(541)\n"
            "    draws = np.array([(rng.choice([-1,1], size=len(mags))*mags).mean() for _ in range(5000)])\n"
            "    coin_p = float((draws >= obs).mean())\n"
            "else:\n"
            f"    obs = {R['fib_bps']}/1e4; draws = np.random.default_rng(541).normal(0, abs(obs)+1e-4, 5000); coin_p = {R['coin_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "ax.hist(draws*1e4, bins=40, color=GREY, alpha=.7)\n"
            "ax.axvline(obs*1e4, c=RED, lw=2, label=f'observed {obs*1e4:+.0f} bps')\n"
            "ax.set_xlabel('mean reversal return under a coin (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Fibonacci reversal vs a coin at the same swings \\u2014 p = {coin_p:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'coin placebo p = {coin_p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: **H\\u2084 rejected** \\u2014 the observed Fibonacci bounce sits in "
            f"the fat middle of the coin distribution (*p* {R['coin_p']}). The level adds no "
            "directional information over a coin placed at the identical swings."
        ),
        md(
            "### 4c \\u00b7 Robustness \\u2014 H\\u2083 (does the (non-)edge hold?)\n\n"
            "The Fib arm *t* and the Fib\\u2212placebo edge *t* across the ZigZag\\u00d7horizon grid."
        ),
        code(
            "grid = " + repr(R["grid"]) + "\n"
            "if HAVE_REAL:\n"
            "    grid = []\n"
            "    for pct in (0.03, 0.05, 0.07):\n"
            "        for h in (10, 20, 40):\n"
            "            fa, pa = pooled_arms(TAPES, pct=pct, horizon=h)\n"
            "            grid.append((pct, h, len(fa), st.ttest_vs_zero(fa), st.two_sample_t(fa, pa)))\n"
            "tab = pd.DataFrame(grid, columns=['zigzag','horizon','fib_n','fib_t','edge_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "x = range(len(tab)); labs = [f'{int(p*100)}%/{h}d' for p,h in zip(tab.zigzag, tab.horizon)]\n"
            "ax.plot(x, tab.fib_t, 'o-', c=RED, lw=2, label='Fib arm t')\n"
            "ax.plot(x, tab.edge_t, 's-', c=GREY, lw=2, label='Fib \\u2212 placebo edge t')\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(-2, ls='--', c=GREEN, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(labs, rotation=40, ha='right')\n"
            "ax.set_ylabel('t-stat'); ax.set_ylim(-3, 3)\n"
            "ax.set_title('Nothing clears \\u00b12; the edge even flips sign \\u2014 noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: **H\\u2083 rejected.** No cell of the grid clears |*t*| = 2, the Fib "
            "arm is flat-to-negative throughout, and the Fib\\u2212placebo edge wanders from +1.53 to "
            "\\u22121.00 \\u2014 the sign isn't even stable. That is exactly what a non-effect looks "
            "like."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** \\u2014 H\\u2081 null (Fib arm *t* {R['fib_t']}); H\\u2082 rejected "
            f"(edge *t* {R['edge_t']}); H\\u2083 rejected (sign flips); H\\u2084 rejected (coin *p* "
            f"{R['coin_p']}).\n"
            f"- **Tradability `MIRAGE`** \\u2014 gross {R['fib_bps']:.0f} bps \\u2192 net "
            f"{R['fib_net']:.0f} bps; per-tape edge is pure scatter.\n"
            "- **Fib levels special? `NOT SUPPORTED`** \\u2014 arbitrary fractions match; the engine "
            "would catch a real level-effect (next beat)."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? \\u2014 costs and the per-tape scatter\n\n"
            "There's no edge to charge costs against, and the per-tape results are noise."
        ),
        code(
            "per = " + repr(R["per_tape"]) + "\n"
            "if HAVE_REAL:\n"
            "    per = []\n"
            "    for tk, b in TAPES.items():\n"
            "        close = b['close'].to_numpy(float); piv = st.zigzag(close, PCT); rd = st.swing_retracements(piv)\n"
            "        lf = st.run_trades(close, st.bin_touches(rd, data.FIB_LEVELS, TOL), HORIZON, 0.0)\n"
            "        lp = st.run_trades(close, st.bin_touches(rd, data.PLACEBO_LEVELS, TOL), HORIZON, 0.0)\n"
            "        fb = lf['ret_gross'].mean()*1e4 if len(lf) else np.nan\n"
            "        eb = (lf['ret_gross'].mean()-lp['ret_gross'].mean())*1e4 if len(lf) and len(lp) else np.nan\n"
            "        per.append((tk, len(lf), fb, eb))\n"
            "tab = pd.DataFrame(per, columns=['tape','fib_n','fib_bps','edge_bps'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if e>0 else RED for e in tab.edge_bps]\n"
            "ax.bar(tab.tape, tab.edge_bps, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Fib \\u2212 placebo edge (bps)')\n"
            "ax.set_title('Per-tape edge: \\u2212478 to +768 bps with no pattern \\u2014 averaging noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(0).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the pooled +{R['edge_bps']:.0f} bps edge is a small average of large "
            "cancelling per-tape numbers (IWM \\u2212478, QQQ +768). Add a 1 bp/side spread and the "
            f"Fibonacci bet goes from {R['fib_bps']:.0f} bps to {R['fib_net']:.0f} bps net. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further \\u2014 the synthetic controls\n\n"
            "Is the engine a faithful detector, or would it miss a real level-effect? Two controls. "
            "**(a)** A clean swing panel where swings that retrace to a Fibonacci depth are *given* an "
            "extra forward return `fib_pull`; the binning + two-sample test must recover it, averaged "
            "over 25 seeds so no lucky seed can fake it. **(b)** The whole pipeline on a genuine "
            "random-swing price path with no planted effect must stay flat."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "if True:  # deterministic, offline, fast\n"
            "    pulls = [0.0, 0.02, 0.04, 0.06]\n"
            "    ctrl = [(p, st.synthetic_control(data, fib_pull=p, n_seeds=25)['mean_edge_t']) for p in pulls]\n"
            "xs = [c[0] for c in ctrl]; ts = [c[1] for c in ctrl]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(xs, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted level-effect (fib_pull)'); ax.set_ylabel('mean Fib\\u2212placebo edge t (25 seeds)')\n"
            "ax.set_title('The engine banks a planted Fib level-effect \\u2014 flat at the null, soaring as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for p, t in ctrl: print(f'fib_pull {p:+.2f} -> mean edge t {t:+.2f}')\n"
            "# (b) genuine price-path null: pipeline must find no false edge\n"
            "null_ed = []\n"
            "for s in range(541, 551):\n"
            "    bars, _ = data.synthetic_tape(n_swings=1200, fib_pull=0.0, seed=s)\n"
            "    r = st.run_experiment(bars, horizon=20, cost_bps=0.0, coin_draws=1, seed=s)\n"
            "    null_ed.append(r['edge_t'])\n"
            "print(f'price-path NULL mean edge t (10 seeds): {np.mean(null_ed):+.2f}')"
        ),
        md(
            "At the null both controls sit at \\u2248 0 (edge *t* \\u22120.01 on the swing panel, "
            f"{R['tape_null_edge_t']:+.2f} on the price path); a planted Fibonacci level-effect drives the "
            "edge *t* to +4.6 / +9.2 / +13.8 as it grows. **The detector is faithful** \\u2014 so the "
            "real-tape null is a statement about the tape, not a broken test. For the wave-count "
            "version of the Fibonacci story, see [445 Elliott Wave](../../445-elliott-wave/); for "
            "round *price* levels, [93 Round Numbers](../../93-round-numbers/)."
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
