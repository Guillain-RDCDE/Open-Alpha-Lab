"""Generate the two narrative notebooks for Study 406 (Harami pattern).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
control runs anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-18).
# Basket: 29 long-listed large-caps + SPY, yfinance daily 2005-01-03 -> 2026-06-18.
R = dict(
    n_names=30, years=21.5, fingerprint="bf1d6cb7ca54",
    # Pooled signed return (long bullish / short bearish), gross, by horizon
    n1=13078, n3=13073, n5=13071, n10=13054,
    mean1=0.020, mean3=0.048, mean5=0.099, mean10=0.142,
    base1=0.028, base3=0.141, base5=0.253, base10=0.530,
    win1=50.1, win3=51.1, win5=51.3, win10=51.5,
    thac1=1.65, thac3=2.12, thac5=3.39, thac10=3.37,
    pplac1=0.051, pplac3=0.019, pplac5=0.001, pplac10=0.000,
    net1=-0.081, net3=-0.055, net5=-0.006, net10=0.032,
    # Leg split — bullish (long) vs bearish (short)
    bull1=0.075, bull3=0.198, bull5=0.337, bull10=0.652,
    bull_t1=4.50, bull_t3=6.34, bull_t5=8.00, bull_t10=10.17,
    bull_win5=55.8, bull_win10=57.4,
    bear1=-0.039, bear3=-0.109, bear5=-0.151, bear10=-0.394,
    bear_t1=-2.15, bear_t3=-3.31, bear_t5=-3.38, bear_t10=-5.81,
    bear_win5=46.5, bear_win10=45.3,
    # Bullish leg's EXCESS over the unconditional long drift (Welch t)
    excess_t1=2.74, excess_t3=1.68, excess_t5=1.88, excess_t10=1.94,
    # Myth check (H=1): does a prior-trend / volume-contraction filter rescue it?
    myth_plain_t=1.65, myth_trend_t=1.17, myth_vol_t=1.31, myth_both_t=0.06,
    myth_plain_m=0.020, myth_trend_m=0.018, myth_vol_m=0.018, myth_both_m=0.001,
    # Synthetic positive control (H=1)
    syn_null_t=0.06, syn_null_p=0.479, syn_edge_t=37.73, syn_edge_p=0.000, syn_edge_win=62,
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

from harami_pattern import data, strategy as st

HORIZONS = st.HORIZONS
CACHE = os.path.abspath(os.path.join("..", "_cache"))
HAVE_REAL = data.have_real(cache_dir=CACHE)

def real_panel():
    return data.load_real(cache_dir=CACHE, allow_fetch=False)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Harami — does the 'inside bar' flip the trend?\n"
            "### The two-candle reversal that's really one leg of beta in a costume\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Reliable reversal%3F: Misattributed](https://img.shields.io/badge/Reliable_reversal%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "A *harami* (Japanese for 'pregnant') is two candles: a big one, then a small one whose "
            "body sits **inside** the big one's body. Candlestick lore reads it as the market pausing "
            "for breath before turning around — a **bullish harami** (big red day, small green inside) "
            "after a slide is supposed to mark the bottom; a **bearish harami** (big green, small red "
            "inside) after a rally marks the top. Does the inside bar actually flip the trend?\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the leg split, the placebo and "
            "the cost ladder? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the bullish harami go up? | **Yes — but mostly because the market does.** "
            f"Long after a bullish harami earns +{R['bull5']:.2f}% over 5 days (*t* = {R['bull_t5']:.1f}) "
            f"— against a +{R['base5']:.2f}% you'd get just being long *any* day. |\n"
            f"| Does the bearish harami go down? | **No — it keeps going up.** Shorting after a "
            f"bearish harami *loses* {abs(R['bear5']):.2f}% over 5 days (*t* = {R['bear_t5']:.1f}). "
            "The 'top' signal points the wrong way. |\n"
            f"| Is it a real reversal? | **No.** The only working leg is the long one, and most of its "
            "gain is the market's natural upward drift — not a trend *flip*. |\n"
            f"| Could you trade it? | **Barely.** Net of costs the two-sided rule is roughly **zero** "
            f"(net {R['net5']:+.3f}% at 5 days). The fast money is gone before you start. |\n\n"
            "> The harami looks like it 'works' in a backtest because half of it is just *being long "
            "in a 21-year bull market.* Strip that out and the reversal story collapses."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A bullish harami — a large down candle followed by a small up candle contained "
            "inside it — shows sellers losing momentum at the end of a downtrend: the bottom is in, "
            "buy. A bearish harami is the mirror at a top: sell. The small inside bar is the market "
            "hesitating before it reverses.\"*\n\n"
            "The harami is one of the canonical two-candle reversals in Steve Nison's *Japanese "
            "Candlestick Charting Techniques* — the book that brought candlesticks to the West. It is "
            "taught on essentially every charting platform. We test whether the inside bar actually "
            "predicts a reversal on modern, liquid US large-caps."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a two-candle shape reliably called tops and bottoms, it would be a free, mechanical "
            "timing signal anyone could read off a chart. Millions of retail traders act on harami "
            "signals daily. If the bullish leg is just beta in disguise and the bearish leg points the "
            "*wrong way*, then traders shorting the 'top' are systematically fading a market that keeps "
            "rising — paying spreads and borrow to be on the losing side."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "One honest event study:\n\n"
            "1. **Find every harami** by precise OHLC rules (small body fully inside the prior, larger "
            "body, opposite colours) across 30 liquid names over 21 years.\n"
            "2. **Enter the next open** (one day's lag — the signal is only known at the close) and "
            "measure the forward 1 / 3 / 5 / 10-day return, **long after bullish, short after bearish.**\n"
            "3. **Compare to doing nothing** — the unconditional forward return of every bar (the "
            "market's natural drift) — and ask whether the *bullish* leg beats just being long.\n"
            "4. **Split the legs.** A real reversal pattern should work *both* ways. If only the long "
            "leg works and the short leg fails, it isn't a reversal — it's beta.\n\n"
            "**What would make us say 'mirage':** the bearish leg pointing the wrong way, the bullish "
            "leg not clearly beating buy-and-hold, and the net edge dying at realistic costs."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First, the two legs side by side.** Long after bullish, short after bearish — does each "
            "leg make money in its predicted direction?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = real_panel()\n"
            "    bull, bear = [], []\n"
            "    for h in HORIZONS:\n"
            "        ev = st.collect_events(panel, h)\n"
            "        bull.append(ev[ev['dir']>0]['signed_ret'].mean()*100)\n"
            "        bear.append(ev[ev['dir']<0]['signed_ret'].mean()*100)\n"
            "else:\n"
            f"    bull = [{R['bull1']},{R['bull3']},{R['bull5']},{R['bull10']}]\n"
            f"    bear = [{R['bear1']},{R['bear3']},{R['bear5']},{R['bear10']}]\n"
            "x = np.arange(len(HORIZONS)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-w/2, bull, w, color=GREEN, label='bullish leg (go long)')\n"
            "ax.bar(x+w/2, bear, w, color=RED, label='bearish leg (go short)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HORIZONS])\n"
            "ax.set_ylabel('signed return per event (%)')\n"
            "ax.set_title('Harami: the long leg works, the short leg points the WRONG way')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Bearish (short) leg is negative at every horizon — it keeps rising.')"
        ),
        md(
            f"The bullish leg is solidly positive ({R['bull5']:+.2f}% at 5 days). The bearish leg is "
            f"**negative** at every horizon ({R['bear5']:+.2f}% at 5 days) — shorting the 'top' loses "
            "money because the stock keeps climbing. A genuine reversal pattern should work both ways. "
            "This one only works on the side that agrees with the market's long-run direction."
        ),
        md(
            "**So is the bullish leg a real signal — or just 'be long in a bull market'?** "
            "Compare the bullish-harami return to the unconditional return of *every* bar (the drift "
            "you'd capture by buying on any random day)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull_m, base_m = [], []\n"
            "    for h in HORIZONS:\n"
            "        ev = st.collect_events(panel, h)\n"
            "        bull_m.append(ev[ev['dir']>0]['signed_ret'].mean()*100)\n"
            "        base_m.append(st.unconditional_base(panel, h).mean()*100)\n"
            "else:\n"
            f"    bull_m = [{R['bull1']},{R['bull3']},{R['bull5']},{R['bull10']}]\n"
            f"    base_m = [{R['base1']},{R['base3']},{R['base5']},{R['base10']}]\n"
            "x = np.arange(len(HORIZONS)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-w/2, bull_m, w, color=GREEN, label='after a bullish harami')\n"
            "ax.bar(x+w/2, base_m, w, color=GREY, label='any random day (drift)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HORIZONS])\n"
            "ax.set_ylabel('long return (%)')\n"
            "ax.set_title('The bullish harami barely beats just being long')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Most of the bullish-harami gain is the market drift you get for free.')"
        ),
        md(
            f"The bullish harami ({R['bull10']:+.2f}% at 10 days) only modestly beats the unconditional "
            f"drift ({R['base10']:+.2f}%). The 'edge' over buy-and-hold is small — and as the quant "
            "notebook shows, that excess clears the *t* = 2 bar only at the 1-day horizon, fading to "
            "noise as you hold longer."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The bullish leg is genuinely positive (HAC *t* up to "
            f"{R['bull_t10']:.1f}), but the bearish leg points the **wrong way** ({R['bear_t10']:.1f}). "
            "It's real on one leg, busted on the other.\n"
            f"- **Tradability — Mirage.** Net of costs the two-sided rule is ~zero (net "
            f"{R['net5']:+.3f}% at 5 days); the round trip plus short borrow eats the thin edge.\n"
            f"- **Reliable reversal? — Misattributed.** The working leg's gain is mostly the market's "
            "long-run drift, not a trend *flip* — and the short side fails outright."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Each event is a fresh round trip: a spread in, a spread out, plus borrow on the short "
            "legs. Watch the gross edge melt into the costs:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    ev5 = st.collect_events(panel, 5)\n"
            "    gross5 = ev5['signed_ret'].mean()\n"
            "    net = [st.net_of_costs(gross5, 5, cost_bps=c)*100 for c in costs]\n"
            "else:\n"
            f"    gross5 = {R['mean5']/100}\n"
            f"    net = [st.net_of_costs(gross5, 5, cost_bps=c)*100 for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net per event (%, 5-day hold)')\n"
            "ax.set_title('A realistic 5 bps each way pushes the two-sided rule to zero')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'gross 5d = {{gross5*100:+.3f}}%  ->  net at 5bps = {{net[2]:+.3f}}%')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a real day-after reversal in a "
            "synthetic tape — the detector lights up cleanly (*t* ≈ 38). The engine works; the market "
            "just doesn't reverse after a harami.\n"
            "- **Long-only the bullish leg?** That captures beta with extra steps — a fork worth "
            "running against a simple buy-and-hold benchmark to see if anything survives.\n"
            "- **The engulfing cousin.** [Study 402](../../402-engulfing-pattern/) tests the "
            "*opposite* geometry (the second body swallows the first) — also busted as a reversal.\n\n"
            "*Think the inside bar works on a different asset, timeframe, or with a confirming filter "
            "we missed? Fork this, change it, and show a two-legged edge that clears HAC *t* = 2 over "
            "the unconditional drift. That is the bar.*"
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
            "# Harami — a quantitative teardown\n"
            "### Daily OHLCV · real-body inside-bar detector · signed event study · HAC inference · leg split\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Reliable reversal%3F: Misattributed](https://img.shields.io/badge/Reliable_reversal%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We detect every harami across "
            "a fixed 30-name basket, measure the signed forward return (long bullish / short bearish), "
            "and ask whether it clears HAC *t* = 2 — and crucially whether the **bullish leg beats the "
            "unconditional drift** rather than just riding it.\n\n"
            "> **Not investment advice.** Real data: yfinance daily bars, 2005-01-03 → 2026-06-18 "
            f"(as-of 2026-06-18, fingerprint `{R['fingerprint']}`); the offline core and the synthetic "
            "control run with no network. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Pooled signed return clears the bar at multi-day horizons "
            f"(HAC *t* = **{R['thac5']:.2f}** at 5d), but it is **one-legged**: the bullish leg is "
            f"strong (*t* = **{R['bull_t10']:.2f}** at 10d) while the bearish leg is **wrong-signed** "
            f"(*t* = **{R['bear_t10']:.2f}**). |\n"
            f"| **Tradability** | `MIRAGE` | Net of a 5 bps round trip + short borrow the two-sided "
            f"rule is ≈ 0 (net **{R['net5']:+.3f}%** at 5d); the thin gross edge does not survive. |\n"
            f"| **Reliable reversal?** | `MISATTRIBUTED` | The bullish leg's *excess over the "
            f"unconditional long drift* clears *t* = 2 only at 1 day ({R['excess_t1']:.2f}), fading to "
            f"{R['excess_t10']:.2f} by 10d — the gain is beta/drift, not a trend reversal. |\n\n"
            "> In plain words: the harami 'works' in a naive backtest because its long leg is long "
            "during a 21-year bull market. Net out the drift and net out costs, and the reversal "
            "story is gone — while the short leg fails outright."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "Let $(O_0,C_0)$ be the prior bar and $(O_1,C_1)$ the current bar, with real-body sizes "
            "$b_0=|C_0-O_0|$, $b_1=|C_1-O_1|$. A **harami** requires the current real body to sit fully "
            "inside the prior one and be strictly smaller, with opposite colours:\n\n"
            "$$ b_1 < b_0,\\quad \\min(O_1,C_1) \\ge \\min(O_0,C_0),\\quad \\max(O_1,C_1) \\le \\max(O_0,C_0) $$\n\n"
            "Direction $d=+1$ (bullish: $C_0<O_0,\\ C_1>O_1$) or $d=-1$ (bearish: $C_0>O_0,\\ C_1<O_1$). "
            "Confirmed at the close of $t$, we enter at the open of $t+1$ and hold $H$ bars. The signed "
            "event return is $d\\cdot r_{t+1\\to t+1+H}$. The hypotheses:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[d\\cdot r] > 0$ with HAC *t* ≥ 2.\n"
            "- **H₂ (two-legged).** *Both* legs work in their predicted direction (a true reversal).\n"
            "- **H₃ (alpha not beta).** The bullish leg beats the unconditional long drift.\n"
            "- **H₄ (tradable).** It survives a realistic round trip + short borrow.\n\n"
            "We find H₁ holds *pooled*, but H₂ fails (bearish leg wrong-signed), H₃ fails beyond 1 day, "
            "and H₄ fails — hence **Mixed × Mirage × Misattributed**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The harami is one of the most-taught two-candle reversals (Nison 1991). If H₁–H₄ held it "
            "would be a free, mechanical timing signal. The actual finding — a one-legged, beta-driven, "
            "cost-fragile effect with a *wrong-signed* short leg — matters because it is exactly the "
            "failure mode candlestick lore is most prone to: a pattern that survives a naive long-biased "
            "backtest gets blessed as a 'reversal', when the only thing reversing is the reader's P&L on "
            "the short side."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** Real-body harami (opposite colours, current body strictly smaller and "
            "fully inside the prior body). One signal per qualifying bar.\n"
            "- **Entry.** Next bar's *open* (one execution lag). **Exit.** Close of bar $t+H$, no stop.\n"
            "- **Sign.** Long after bullish, short after bearish — the folk-predicted direction.\n"
            "- **Benchmark.** The unconditional forward return of every bar (the market drift).\n"
            "- **Inference.** Newey–West HAC *t* on signed event returns vs 0; a coin-signed "
            "label-shuffle placebo; a Welch *t* of the bullish leg vs the unconditional drift.\n"
            "- **Costs.** One-way bps × NAV charged on the round trip; shorts pay 50 bps/yr borrow.\n"
            "- **Myth-check.** Does a prior trend (down for bullish, up for bearish) and/or a volume "
            "*contraction* on the inside day rescue it?\n"
            "- **Positive control.** Synthetic tape with a planted day-after reversal — the harness "
            "must recover it and must read ≈ 0 when the edge is 0.\n\n"
            "Basket: 29 long-listed large-caps + SPY; yfinance daily 2005→2026, **survivors** (named "
            "on the Signal axis)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Pooled signed return by horizon — and its HAC *t*\n\n"
            "If the pattern were a real reversal, the signed return should clear +2 cleanly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel = real_panel()\n"
            "    rows = [st.summarize(panel, h, n_draws=2000) for h in HORIZONS]\n"
            "    df = pd.DataFrame(rows)\n"
            "    means = (df['mean']*100).tolist(); ths = df['t_hac'].tolist()\n"
            "else:\n"
            f"    means = [{R['mean1']},{R['mean3']},{R['mean5']},{R['mean10']}]\n"
            f"    ths = [{R['thac1']},{R['thac3']},{R['thac5']},{R['thac10']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([f'{h}d' for h in HORIZONS], means, 'o-', c=GREEN, lw=2, label='signed mean (%)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot([f'{h}d' for h in HORIZONS], ths, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold horizon'); ax.set_ylabel('signed mean (%)', color=GREEN)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Pooled signed return clears t=2 from 3 days out — but read the leg split next')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, m, t in zip(HORIZONS, means, ths):\n"
            "    print(f'H={h:2d}  signed={m:+.3f}%  HAC t={t:+.2f}')"
        ),
        md(
            f"> In plain words: pooled, the signed return clears *t* = 2 at 3 days ({R['thac3']:.2f}) "
            f"and is strong by 5 days ({R['thac5']:.2f}). By the raw inference bar that's a hit — but "
            "'pooled' hides which leg is doing the work."
        ),
        md(
            "### 4b · The leg split — the reversal is one-legged\n\n"
            "A genuine reversal pattern must work in *both* directions. Split bullish (long) from "
            "bearish (short)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull_m, bull_t, bear_m, bear_t = [], [], [], []\n"
            "    for h in HORIZONS:\n"
            "        ev = st.collect_events(panel, h)\n"
            "        b = ev[ev['dir']>0]['signed_ret'].to_numpy()\n"
            "        s = ev[ev['dir']<0]['signed_ret'].to_numpy()\n"
            "        bull_m.append(b.mean()*100); bull_t.append(st.hac_t(b))\n"
            "        bear_m.append(s.mean()*100); bear_t.append(st.hac_t(s))\n"
            "else:\n"
            f"    bull_m=[{R['bull1']},{R['bull3']},{R['bull5']},{R['bull10']}]\n"
            f"    bull_t=[{R['bull_t1']},{R['bull_t3']},{R['bull_t5']},{R['bull_t10']}]\n"
            f"    bear_m=[{R['bear1']},{R['bear3']},{R['bear5']},{R['bear10']}]\n"
            f"    bear_t=[{R['bear_t1']},{R['bear_t3']},{R['bear_t5']},{R['bear_t10']}]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "x = np.arange(len(HORIZONS))\n"
            "ax[0].bar(x, bull_m, color=GREEN); ax[0].bar(x, bear_m, color=RED, alpha=.85)\n"
            "ax[0].axhline(0,c='k',lw=1); ax[0].set_xticks(x); ax[0].set_xticklabels([f'{h}d' for h in HORIZONS])\n"
            "ax[0].set_title('signed return per event (%)'); ax[0].set_ylabel('%')\n"
            "ax[0].legend(['bullish (long)','bearish (short)'])\n"
            "ax[1].plot([f'{h}d' for h in HORIZONS], bull_t, 'o-', c=GREEN, lw=2, label='bullish leg t')\n"
            "ax[1].plot([f'{h}d' for h in HORIZONS], bear_t, 's-', c=RED, lw=2, label='bearish leg t')\n"
            "ax[1].axhline(2, ls=':', c=GREY); ax[1].axhline(-2, ls=':', c=GREY); ax[1].axhline(0,c='k',lw=1)\n"
            "ax[1].set_title('HAC t by leg'); ax[1].set_ylabel('HAC t'); ax[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Bullish leg t (10d): {:+.2f}   |   Bearish leg t (10d): {:+.2f}'.format(bull_t[-1], bear_t[-1]))"
        ),
        md(
            f"> In plain words: the bullish leg is strongly positive (*t* = {R['bull_t10']:.2f} at 10d). "
            f"The bearish leg is **significantly wrong-signed** (*t* = {R['bear_t10']:.2f}) — shorting "
            "the 'top' loses money. The pooled significance was the long leg carrying a dead short leg."
        ),
        md(
            "### 4c · Is the bullish leg alpha or beta? — excess over the drift\n\n"
            "The bullish leg is long. So is buying on any random day in a bull market. Welch *t* of the "
            "bullish-harami return **minus the unconditional long drift** is the honest alpha test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex_t = []\n"
            "    for h in HORIZONS:\n"
            "        ev = st.collect_events(panel, h)\n"
            "        b = ev[ev['dir']>0]['signed_ret'].to_numpy()\n"
            "        base = st.unconditional_base(panel, h)\n"
            "        ex_t.append(st.welch_t(b, base))\n"
            "else:\n"
            f"    ex_t = [{R['excess_t1']},{R['excess_t3']},{R['excess_t5']},{R['excess_t10']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "col = [GREEN if t>=2 else AMBER for t in ex_t]\n"
            "ax.bar([f'{h}d' for h in HORIZONS], ex_t, color=col)\n"
            "ax.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Welch t (bullish leg − unconditional drift)')\n"
            "ax.set_title('The bullish leg beats buy-and-hold only at 1 day, then fades to noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, t in zip(HORIZONS, ex_t): print(f'H={h:2d}  excess Welch t = {t:+.2f}')"
        ),
        md(
            f"> In plain words: over the unconditional drift, the bullish leg clears *t* = 2 only at "
            f"1 day ({R['excess_t1']:.2f}); by 3/5/10 days it is {R['excess_t3']:.2f}/{R['excess_t5']:.2f}"
            f"/{R['excess_t10']:.2f} — below the bar. The multi-day 'edge' is mostly drift you'd get for free."
        ),
        md(
            "### 4d · The placebo — could a coin-signed random pick look this good?\n\n"
            "Draw the same many bars at random, sign each by a coin, and compare to the observed mean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.placebo_pvalue(panel, 5, len(st.collect_events(panel,5)), n_draws=5000)\n"
            "    draws, obs, p = res['draws']*100, res['obs']*100, res['p_value']\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "    ax.hist(draws, bins=40, color=GREY, alpha=.7)\n"
            "    ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed: {obs:+.3f}%')\n"
            "    ax.axvline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('coin-signed random mean (%, 5-day)'); ax.set_ylabel('count')\n"
            "    ax.set_title(f'Label-shuffle placebo: p = {p:.3f}'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'placebo p (5d) = {p:.3f}')\n"
            "else:\n"
            f"    print('placebo p (5d) = {R['pplac5']:.3f}  (frozen)')"
        ),
        md(
            f"> In plain words: the placebo *p* at 5 days is {R['pplac5']:.3f} — the *pooled* mean is "
            "hard to get by chance. But the placebo only tests the pooled magnitude; the leg split and "
            "the excess-over-drift test (4b, 4c) are what demote it from 'real reversal' to 'beta'."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — pooled HAC *t* = {R['thac5']:.2f} (5d) clears the bar, but it is "
            f"one-legged: bullish *t* = {R['bull_t10']:.2f}, bearish *t* = {R['bear_t10']:.2f} "
            "(wrong-signed). Real on the long leg, busted on the short.\n"
            f"- **Tradability `MIRAGE`** — net of a 5 bps round trip + short borrow the two-sided rule "
            f"is ≈ 0 (net {R['net5']:+.3f}% at 5d).\n"
            f"- **Reliable reversal? `MISATTRIBUTED`** — the bullish leg's excess over the drift clears "
            f"*t* = 2 only at 1 day; the gain is beta, not a trend flip."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the myth-check and the cost ladder\n\n"
            "First the myth-check: candlestick lore says a harami only counts *after a trend* and with "
            "a *volume contraction* on the inside day. Does either filter rescue it (H=1)?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for label, rt, rv in [('plain',False,False),('+prior trend',True,False),\n"
            "                          ('+vol contraction',False,True),('+both',True,True)]:\n"
            "        s = st.summarize(panel, 1, placebo=False, require_trend=rt, require_volume=rv)\n"
            "        rows.append((label, s['n_events'], s['mean']*100, s['t_hac']))\n"
            "    mc = pd.DataFrame(rows, columns=['filter','n','mean%','t_hac'])\n"
            "else:\n"
            "    mc = pd.DataFrame({'filter':['plain','+prior trend','+vol contraction','+both'],\n"
            f"        'mean%':[{R['myth_plain_m']},{R['myth_trend_m']},{R['myth_vol_m']},{R['myth_both_m']}],\n"
            f"        't_hac':[{R['myth_plain_t']},{R['myth_trend_t']},{R['myth_vol_t']},{R['myth_both_t']}]}})\n"
            "print(mc.round(3).to_string(index=False))\n"
            "print('\\nNo filter lifts the 1-day signal over t=2 — the textbook confirmations do not help.')"
        ),
        md(
            "Now the cost ladder on the 5-day two-sided rule:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    g5 = st.collect_events(panel, 5)['signed_ret'].mean()\n"
            "else:\n"
            f"    g5 = {R['mean5']/100}\n"
            "net = [st.net_of_costs(g5, 5, cost_bps=c)*100 for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net per event (%, 5-day)')\n"
            "ax.set_title('A realistic 5 bps each way pushes the two-sided rule to ~zero')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'gross 5d = {{g5*100:+.3f}}%  ->  net at 5bps = {{net[2]:+.3f}}%')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding a real post-harami reversal? Plant a day-after drift in "
            "a synthetic tape (edge = 0 → must read noise; edge > 0 → must light up):"
        ),
        code(
            "for edge in (0.0, 0.004):\n"
            "    pan, truth = data.synthetic_panel(edge=edge, seed=406)\n"
            "    s = st.summarize(pan, 1, n_draws=2000)\n"
            "    print(f'planted edge={edge:+.4f}: n={s[\"n_events\"]:>5}  mean={s[\"mean\"]*100:+.3f}%  '\n"
            "          f'HAC t={s[\"t_hac\"]:+.2f}  placebo p={s[\"p_placebo\"]:.3f}  win={s[\"win\"]*100:.0f}%')\n"
            f"print('\\nNull tape: t≈{R['syn_null_t']:.2f} (noise). Planted tape: t≈{R['syn_edge_t']:.1f} — engine is faithful.')"
        ),
        md(
            f"The engine is a faithful reversal detector: on a planted tape it lights up at *t* ≈ "
            f"{R['syn_edge_t']:.0f}, and on a null tape it reads ≈ {R['syn_null_t']:.2f} (noise, "
            f"placebo *p* = {R['syn_null_p']:.2f}). So the real-tape verdict is a statement about the "
            "**market** — modern US large-caps don't reverse after a harami; the long leg just rides "
            "the drift — not about the method. Forks worth trying: the long-only bullish leg vs a "
            "buy-and-hold benchmark; intraday harami; or commodity/FX tapes where mean-reversion is "
            "more native."
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
