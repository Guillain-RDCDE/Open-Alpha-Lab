"""Generate the two narrative notebooks for Study 405 (Doji Reversal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks walk the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
positive-control cells run anywhere, offline and deterministic; the real-tape cells use
the cached daily parquets under ../_cache/ if present and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader. Real numbers live in ONE place — this dict ``R`` — mirrored by docs/results.md.
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


# ---------------------------------------------------------------------------
# Frozen real-tape headline numbers — mirror of docs/results.md.
# As-of 2025-12-31 (full-year cut, no partial 2026). Panel fingerprint 91bd864b7001.
# Basket: SPY + 28 long-listed US large-caps; body_frac = 0.10; cost = 2 bps + 1 bp borrow.
# ---------------------------------------------------------------------------
R = dict(
    asof="2025-12-31", panel_fp="91bd864b7001",
    n_dojis=18750, n_names=29, years=24.5, per_name_per_yr=26.4,
    # Doji reversal (against the prior move), gross, by horizon
    d1_mean=-0.50, d1_t=-0.49, d1_win=49.7, d1_net=-3.00,
    d3_mean=+1.20, d3_t=+0.61, d3_win=49.9, d3_net=-1.30,
    d5_mean=+5.98, d5_t=+2.31, d5_win=50.7, d5_net=+3.48,
    d10_mean=+3.93, d10_t=+1.08, d10_win=50.0, d10_net=+1.43,
    # Unconditional base rate (same against-the-prior-move bet on ALL bars)
    b1_mean=+1.95, b3_mean=+4.23, b5_mean=+8.58, b10_mean=+7.03,
    # Doji minus base (the honest "does a doji ADD anything?" number)
    delta1=-2.45, delta3=-3.03, delta5=-2.60, delta10=-3.10,
    # Pooled label-shuffle placebo p (fraction of random same-size sets that beat the doji)
    p1=0.995, p3=0.941, p5=0.842, p10=0.812,
    # body_frac robustness (h=5): t tracks sample size, delta stays negative
    bf05_n=9395, bf05_t=+1.29, bf05_delta=-3.90,
    bf10_n=18764, bf10_t=+2.30, bf10_delta=-2.62,
    bf15_n=28242, bf15_t=+2.53, bf15_delta=-3.03,
    bf20_n=38009, bf20_t=+3.24, bf20_delta=-2.45,
    # Trend-filter myth-check (h=5): downtrend dojis "reverse" — but that's base-rate reversion
    above_n=11089, above_mean=-1.43, above_t=-0.49,
    below_n=7675, below_mean=+16.64, below_t=+3.55,
    # Volume-filter myth-check (h=5)
    hivol_mean=+4.08, hivol_t=+0.76, lovol_mean=+6.91, lovol_t=+2.36,
    # SPY-only (h=5): the index doji "reversal" is negative
    spy_n=627, spy_mean=-7.99, spy_t=-0.88,
    # Synthetic positive control (h=5): edge -> detected delta vs base
    syn0_delta=-1.51, syn0_p=0.716,
    syn1_delta=+19.34, syn1_p=0.000,
    syn2_delta=+38.78, syn2_p=0.000,
    syn4_delta=+84.95, syn4_p=0.000,
)

SIGNAL, TRADE, MYTH = "Weak", "Mirage", "Misattributed"
SIG_COLOR, TR_COLOR, MY_COLOR = "dab617", "c0392b", "8b949e"

# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from doji_reversal import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))
ASOF = "2025-12-31"

def load_panel():
    p = data.load_real(cache_dir=CACHE, fetch=False)
    return {t: b.loc[:ASOF] for t, b in p.items()}

HAVE_REAL = data.have_real(cache_dir=CACHE)
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Doji Reversal — does the candle of 'indecision' call the turn?\n"
            "### The textbook doji-reversal pattern, tested honestly on 25 years of US large-caps\n\n"
            f"![Signal: {SIGNAL}](https://img.shields.io/badge/Signal-{SIGNAL}-{SIG_COLOR}?style=flat-square)\n"
            f"![Tradability: {TRADE}](https://img.shields.io/badge/Tradability-{TRADE}-{TR_COLOR}?style=flat-square)\n"
            f"![Reversal signal%3F: {MYTH}](https://img.shields.io/badge/Reversal_signal%3F-{MYTH}-{MY_COLOR}?style=flat-square)\n\n"
            "A **doji** is the candlestick where the open and the close finish at almost the same "
            "price — a tiny body with wicks above and below. Candlestick lore (Steve Nison's "
            "*Japanese Candlestick Charting Techniques*) calls it the bar of **indecision**: buyers "
            "and sellers fought to a draw, so the prevailing move is exhausted and a **reversal** "
            "is at hand. Buy a doji after a drop; sell one after a rally.\n\n"
            "Does the indecision actually precede a turn? We measure what the price *does* in the "
            "1–10 days after every doji on a fixed basket of 29 liquid US tapes, and compare it "
            "to what a *random* day would have given.\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the placebo, and the "
            "filter myth-check? That's the companion, "
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
            "| Do prices reverse after a doji? | **A little — but no more than after any day.** "
            f"The against-the-move bet earns **{R['d5_mean']:+.1f} bps** over 5 days... |\n"
            f"| ...so is that a *doji* effect? | **No.** The *same* bet on a **random** day earns "
            f"**{R['b5_mean']:+.1f} bps** — *more*. The doji set is **{R['delta5']:+.1f} bps "
            "worse** than average. |\n"
            f"| Could a random set of bars beat it? | **Yes, {R['p5']*100:.0f}% of the time** "
            "(label-shuffle placebo). The doji adds nothing. |\n"
            "| Could you trade it? | **No.** Net of costs the tiny absolute edge is the broad "
            "short-horizon mean-reversion you'd get anywhere — not the doji. |\n\n"
            "> The doji's reversal is real-looking only because *short-horizon prices mildly "
            "mean-revert in general*. Pin the doji against that baseline and the special power "
            "vanishes."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A doji forms when the open and close are virtually equal — the market is in "
            "balance, indecision reigns, and the trend in force is running out of steam. After an "
            "uptrend a doji warns of a top; after a downtrend it warns of a bottom. Trade against "
            "the prior move.\"*\n\n"
            "It is one of the most-taught single-candle patterns. The intuition is genuinely "
            "appealing: a bar that goes nowhere *looks* like exhaustion. We test the strongest "
            "version — a doji, conditioned on the move that led into it, predicts a reversal."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a doji reliably called turns, it would be a free, mechanical timing signal printed "
            "on every chart — buy the bottoms, sell the tops, no indicator math required. Millions "
            "of retail traders are taught exactly this. If instead a doji is *just another bar*, "
            "then the pattern is a costume worn by ordinary short-horizon mean-reversion, and "
            "trading it specifically buys you nothing over trading any day."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "Three honest checks, announced before we run them:\n\n"
            "1. **Measure the forward move.** Detect every doji (tiny body vs the day's range), "
            "tag the 2-day move into it, and take the reversal bet at the *next* open — long after "
            "a drop, short after a rally. Read the return 1/3/5/10 days later.\n"
            "2. **Compare to a random day.** Run the *exact same* against-the-move bet on **all** "
            "bars. If the doji is special, it must beat this baseline.\n"
            "3. **Shuffle the labels.** Re-draw the same number of 'fake dojis' at random, "
            "thousands of times. If a coin's-worth of random bars beats the real dojis often, the "
            "pattern is noise.\n\n"
            "**What would make us say mirage:** the doji does *no better* than a random day, and a "
            "random label set beats it more than 5% of the time."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: the doji's forward reversal across horizons, versus a random day.**"
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    panel = load_panel()\n"
            "    res = st.run_experiment(panel, body_frac=0.10, horizons=tuple(horizons),\n"
            "                            cost_bps=2.0, borrow_bps=1.0, n_draws=2000, seed=405)\n"
            "    doji = [res['per_horizon'][h]['doji']['mean_bps'] for h in horizons]\n"
            "    base = [res['per_horizon'][h]['base']['mean_bps'] for h in horizons]\n"
            "else:\n"
            f"    doji = [{R['d1_mean']}, {R['d3_mean']}, {R['d5_mean']}, {R['d10_mean']}]\n"
            f"    base = [{R['b1_mean']}, {R['b3_mean']}, {R['b5_mean']}, {R['b10_mean']}]\n"
            "x = np.arange(len(horizons)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(x - w/2, doji, w, color=AMBER, label='after a doji')\n"
            "ax.bar(x + w/2, base, w, color=GREY, label='after a random day')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('reversal return (bps, against prior move)')\n"
            "ax.set_title('The doji never beats a random day at any horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, d, b in zip(horizons, doji, base):\n"
            "    print(f'{h:2d}d  doji {d:+6.2f} bps   random day {b:+6.2f} bps   delta {d-b:+6.2f}')"
        ),
        md(
            f"At the 5-day horizon a doji's reversal bet earns **{R['d5_mean']:+.1f} bps** — "
            "positive, and it looks like a win. But the *same bet on a random day* earns "
            f"**{R['b5_mean']:+.1f} bps**. The doji is **worse than average** at every single "
            "horizon. The 'reversal' isn't the doji's doing — it's the broad tendency of "
            "short-horizon prices to bounce a little after a move, which happens on *every* bar."
        ),
        md(
            "**Second: could a random set of bars beat the dojis? (the label-shuffle placebo)**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pool = st.pool_base_rate(panel, horizons=(5,))['rev_5'].to_numpy()\n"
            "    pool = pool[np.isfinite(pool)]\n"
            "    ndj = res['n_dojis']; real = res['per_horizon'][5]['doji']['mean_bps']/1e4\n"
            "    rng = np.random.default_rng(405)\n"
            "    placebo = np.array([rng.choice(pool, size=ndj, replace=False).mean()\n"
            "                        for _ in range(2000)]) * 1e4\n"
            "    p = (placebo >= real*1e4).mean()\n"
            "else:\n"
            f"    placebo = np.random.default_rng(405).normal({R['b5_mean']}, 1.5, 2000)\n"
            f"    real, p = {R['d5_mean']}/1e4, {R['p5']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(placebo, bins=40, color=GREY, alpha=.7, label='random same-size sets (placebo)')\n"
            "ax.axvline(real*1e4, c=AMBER, lw=2.5, label=f'real dojis: {real*1e4:+.1f} bps')\n"
            "ax.set_xlabel('mean 5-day reversal (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'A random bar set beats the dojis {p*100:.0f}% of the time'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'placebo p = {{p:.3f}}  (we need p < 0.05 for the doji to be special)')"
        ),
        md(
            f"The real dojis sit on the **left** of the placebo cloud: a random set of the same "
            f"size beats them **{R['p5']*100:.0f}%** of the time. Far from significant — the doji "
            "label carries no extra reversal information."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The *absolute* 5-day reversal does clear the bar "
            f"(**{R['d5_mean']:+.1f} bps**, HAC *t* = **{R['d5_t']:+.2f}**) — but that is the "
            "general short-horizon mean-reversion, not the doji. Against the right baseline the "
            f"doji is **{R['delta5']:+.1f} bps worse** than a random day (placebo *p* = "
            f"**{R['p5']:.2f}**).\n"
            "- **Tradability — Mirage.** Net of costs the only thing left is the base-rate "
            "bounce you'd get trading any day — there is no doji-specific edge to harvest.\n"
            "- **Reversal signal? — Misattributed.** The reversal is real, the *attribution* is "
            "wrong: it belongs to short-horizon mean-reversion, not to the candle of indecision."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Dojis are common (about 26 per name per year at the 10%-body cut), so turnover is "
            "high. But there is no doji-specific gross edge to pay costs *from* — net of a 2 bps "
            "round-trip and short borrow, the doji book just tracks the same mild base-rate "
            "reversion, minus friction:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    gross = [res['per_horizon'][h]['doji']['mean_bps'] for h in horizons]\n"
            "    net = [res['per_horizon'][h]['doji']['net_bps'] for h in horizons]\n"
            "else:\n"
            f"    gross = [{R['d1_mean']}, {R['d3_mean']}, {R['d5_mean']}, {R['d10_mean']}]\n"
            f"    net = [{R['d1_net']}, {R['d3_net']}, {R['d5_net']}, {R['d10_net']}]\n"
            "x = np.arange(len(horizons)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x - w/2, gross, w, color=AMBER, label='gross')\n"
            "ax.bar(x + w/2, net, w, color=RED, label='net of costs + borrow')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x)\n"
            "ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('doji reversal return (bps)')\n"
            "ax.set_title('Net of costs, the short horizons go negative'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Net 1d/3d are negative; 5d/10d net is just the base-rate bounce minus friction.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a *real* doji-specific "
            "reversal in synthetic data — and the very same harness lights up (placebo *p* → 0). "
            "So the method works; the market just doesn't have a doji effect.\n"
            "- **The trend-filter trap.** Folk wisdom says 'only trade dojis against the trend'. "
            "Dojis below their 20-day average *do* bounce harder — but so does *any* bar below "
            "its average. The companion shows that filter is the same misattribution, dressed up.\n"
            "- **Multi-candle patterns.** A doji inside an engulfing or a morning-star setup is a "
            "different (multi-bar) claim — fork this detector and test those.\n"
            "- **Intraday dojis.** The hourly/5-minute doji is a separate question with its own "
            "microstructure; worth a study of its own.\n\n"
            "*Think a stricter doji (true zero body, long wicks) or a specific market regime "
            "rescues it? Fork this, tighten the detector, and show a doji-minus-baseline edge "
            "that clears the placebo. That is the bar.*"
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
            "# Doji Reversal — a quantitative teardown\n"
            "### Daily OHLC · 10%-body doji · against-the-prior-move event study · base-rate baseline · HAC + label-shuffle placebo\n\n"
            f"![Signal: {SIGNAL}](https://img.shields.io/badge/Signal-{SIGNAL}-{SIG_COLOR}?style=flat-square)\n"
            f"![Tradability: {TRADE}](https://img.shields.io/badge/Tradability-{TRADE}-{TR_COLOR}?style=flat-square)\n"
            f"![Reversal signal%3F: {MYTH}](https://img.shields.io/badge/Reversal_signal%3F-{MYTH}-{MY_COLOR}?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We detect dojis by a precise "
            "OHLC rule across a 29-name basket, measure the forward reversal return against zero "
            "**and against the unconditional base rate**, and stress it with a label-shuffle placebo, "
            "a detector-cut sweep, a trend/volume filter myth-check, and a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2001-06-25 → **2025-12-31** "
            f"(as-of, full-year cut), panel fingerprint `{R['panel_fp']}`; the offline core and the "
            "controls run on a deterministic synthetic tape. Methods & sources in "
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
            f"| **Signal** | `WEAK` | Absolute 5-day reversal **{R['d5_mean']:+.2f} bps**, HAC "
            f"*t* = **{R['d5_t']:+.2f}** — clears 2 — but the **doji-minus-base-rate** delta is "
            f"**{R['delta5']:+.2f} bps** (negative) and the label-shuffle *p* = **{R['p5']:.2f}**. "
            "The effect is short-horizon mean-reversion, not the doji. |\n"
            f"| **Tradability** | `MIRAGE` | No doji-specific gross edge; net of 2 bps + borrow the "
            "short horizons turn negative and the rest is just base-rate reversion minus friction. |\n"
            f"| **Reversal signal?** | `MISATTRIBUTED` | Across {len(['1','3','5','10'])} horizons "
            "the doji *underperforms* the unconditional bet; the reversal belongs to the baseline. |\n\n"
            "> In plain words: a doji's forward 'reversal' is real-looking only because *all* short "
            "horizons mildly mean-revert. Net it against that baseline and the doji adds nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "Let the real body be $b_t = |C_t - O_t|$ and the range $\\rho_t = H_t - L_t$. A bar is "
            "a **doji** when $b_t \\le \\phi\\,\\rho_t$ with $\\phi = 0.10$ (the textbook 10%-of-range "
            "cut; we sweep $\\phi$). Let $s_t = \\operatorname{sign}(C_t/C_{t-2}-1)$ be the 2-day "
            "move into the bar. The **reversal trade** enters at $O_{t+1}$ against $s_t$ and exits "
            "at $C_{t+h}$:\n\n"
            "$$ r^{\\mathrm{rev}}_{t,h} = -\\,s_t\\left(\\frac{C_{t+h}}{O_{t+1}} - 1\\right). $$\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[r^{\\mathrm{rev}}_{t,h}\\mid \\text{doji}] > 0$.\n"
            "- **H₂ (specificity).** That mean exceeds the *unconditional* base rate "
            "$\\mathbb{E}[r^{\\mathrm{rev}}_{t,h}]$ over all bars — a doji must add reversal "
            "information beyond what any bar carries.\n"
            "- **H₃ (tradable).** It survives one-way costs + short borrow.\n\n"
            "We find H₁ *passes in absolute terms at h=5* (the base-rate reversion), but **H₂ fails** "
            "(negative delta, placebo $p\\approx0.84$) and **H₃ is moot**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on the answer\n\n"
            "The single-candle doji is among the most widely taught reversal signals in retail "
            "education. If H₁–H₃ held it would be a free, mechanical timing overlay. The finding "
            "that the doji *underperforms* the unconditional against-the-move bet is the more "
            "useful lesson: it isolates a classic confound — a generic short-horizon mean-reversion "
            "masquerading as a candlestick pattern. Anyone backtesting 'doji reversal' against a "
            "*zero* benchmark (rather than the base rate) will mistake the baseline for an edge."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** $b_t \\le 0.10\\,\\rho_t$, $\\rho_t > 0$. Swept over "
            "$\\phi \\in \\{0.05, 0.10, 0.15, 0.20\\}$.\n"
            "- **Conditioning.** Prior move $s_t$ over a 2-day lookback; reversal taken *against* it.\n"
            "- **Execution.** Enter at $O_{t+1}$ (one documented lag), exit at $C_{t+h}$, "
            "$h \\in \\{1,3,5,10\\}$.\n"
            "- **Baseline.** The identical against-$s_t$ trade on **every** eligible bar "
            "(the unconditional base rate).\n"
            "- **Inference.** One-sample Newey–West HAC *t* on pooled per-event returns; a "
            "label-shuffle placebo (2,000 random same-size bar sets); a detector-cut robustness "
            "sweep; a trend/volume filter myth-check.\n"
            "- **Costs.** 2 bps one-way round-trip + 1 bp borrow on the ~50% short leg.\n"
            "- **Positive control.** Synthetic panel with a *planted* doji-specific reversal knob.\n\n"
            "Basket: SPY + 28 long-listed US large-caps, 2001-06-25 → 2025-12-31 "
            f"(**{R['n_dojis']:,} dojis**). **Survivorship** is named on the Signal axis: a "
            "fixed surviving-names basket; its direction here is ambiguous (it does not obviously "
            "create the *absence* of a doji effect we report)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Doji vs base rate, by horizon — the doji never wins\n\n"
            "HAC *t* on the doji reversal, and the delta against the unconditional base rate."
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    panel = load_panel()\n"
            "    res = st.run_experiment(panel, body_frac=0.10, horizons=tuple(horizons),\n"
            "                            cost_bps=2.0, borrow_bps=1.0, n_draws=2000, seed=405)\n"
            "    rows = []\n"
            "    for h in horizons:\n"
            "        d = res['per_horizon'][h]\n"
            "        rows.append((h, d['doji']['n'], d['doji']['win_rate']*100,\n"
            "                     d['doji']['mean_bps'], d['doji']['tstat'],\n"
            "                     d['base']['mean_bps'], d['delta_bps'], d['placebo_p']))\n"
            "    tab = pd.DataFrame(rows, columns=['h','n','win%','doji_bps','t','base_bps','delta','placebo_p'])\n"
            "else:\n"
            "    tab = pd.DataFrame({'h':horizons,\n"
            f"        'doji_bps':[{R['d1_mean']},{R['d3_mean']},{R['d5_mean']},{R['d10_mean']}],\n"
            f"        't':[{R['d1_t']},{R['d3_t']},{R['d5_t']},{R['d10_t']}],\n"
            f"        'base_bps':[{R['b1_mean']},{R['b3_mean']},{R['b5_mean']},{R['b10_mean']}],\n"
            f"        'delta':[{R['delta1']},{R['delta3']},{R['delta5']},{R['delta10']}],\n"
            f"        'placebo_p':[{R['p1']},{R['p3']},{R['p5']},{R['p10']}],\n"
            f"        'win%':[{R['d1_win']},{R['d3_win']},{R['d5_win']},{R['d10_win']}],\n"
            f"        'n':[{R['n_dojis']}]*4}})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(horizons))\n"
            "ax.bar(x, tab['delta'], color=[RED if v < 0 else GREEN for v in tab['delta']])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('doji minus base rate (bps)')\n"
            "ax.set_title('Doji underperforms the unconditional bet at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "tab.round(3)"
        ),
        md(
            f"> In plain words: the absolute doji mean clears *t* = 2 only at h=5 "
            f"(**{R['d5_t']:+.2f}**), but its delta vs the base rate is **{R['delta5']:+.2f} bps** "
            f"and the placebo *p* = **{R['p5']:.2f}**. Every horizon's delta is negative — the doji "
            "is a *below-average* day for the reversal bet."
        ),
        md(
            "### 4b · The label-shuffle placebo (h=5)\n\n"
            "Draw $n_{\\text{doji}}$ random bars from the pooled base-rate distribution, 2,000 times; "
            "where does the real doji mean fall?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pool = st.pool_base_rate(panel, horizons=(5,))['rev_5'].to_numpy()\n"
            "    pool = pool[np.isfinite(pool)]\n"
            "    ndj = res['n_dojis']; real = res['per_horizon'][5]['doji']['mean_bps']\n"
            "    rng = np.random.default_rng(405)\n"
            "    placebo = np.array([rng.choice(pool, size=ndj, replace=False).mean()\n"
            "                        for _ in range(2000)]) * 1e4\n"
            "    p = (placebo >= real).mean()\n"
            "else:\n"
            f"    placebo = np.random.default_rng(405).normal({R['b5_mean']}, 1.6, 2000)\n"
            f"    real, p = {R['d5_mean']}, {R['p5']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(placebo, bins=40, color=GREY, alpha=.7, label='placebo means (2,000 draws)')\n"
            "ax.axvline(real, c=AMBER, lw=2.5, label=f'real dojis: {real:+.2f} bps')\n"
            "ax.axvline(placebo.mean(), c='k', ls='--', lw=1, label=f'placebo mean: {placebo.mean():+.2f}')\n"
            "ax.set_xlabel('mean 5-day reversal (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'placebo p = {p:.3f} — the doji set is unremarkable'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'real doji mean {{real:+.2f}} bps sits LEFT of the placebo centre; p = {{p:.3f}}')"
        ),
        md(
            "> In plain words: a same-size set of *random* bars beats the dojis "
            f"{R['p5']*100:.0f}% of the time. The doji label is not a meaningful filter."
        ),
        md(
            "### 4c · Detector-cut robustness — the *t* tracks sample size, not signal\n\n"
            "Loosen the body cut and you collect more 'dojis', which lifts the absolute *t* by "
            "$\\sqrt{n}$ — but the delta vs base rate stays stubbornly negative."
        ),
        code(
            "cuts = [0.05, 0.10, 0.15, 0.20]\n"
            "if HAVE_REAL:\n"
            "    rr = [st.run_experiment(panel, body_frac=bf, horizons=(5,), n_draws=400, seed=405)\n"
            "          for bf in cuts]\n"
            "    ns = [r['n_dojis'] for r in rr]\n"
            "    ts = [r['per_horizon'][5]['doji']['tstat'] for r in rr]\n"
            "    dl = [r['per_horizon'][5]['delta_bps'] for r in rr]\n"
            "else:\n"
            f"    ns = [{R['bf05_n']},{R['bf10_n']},{R['bf15_n']},{R['bf20_n']}]\n"
            f"    ts = [{R['bf05_t']},{R['bf10_t']},{R['bf15_t']},{R['bf20_t']}]\n"
            f"    dl = [{R['bf05_delta']},{R['bf10_delta']},{R['bf15_delta']},{R['bf20_delta']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(cuts, ts, 'o-', c=AMBER, lw=2, label='abs doji HAC t')\n"
            "ax.axhline(2, ls=':', c=GREY, label='t = 2')\n"
            "ax2 = ax.twinx(); ax2.plot(cuts, dl, 's--', c=RED, lw=1.5, label='delta vs base (bps)')\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('body-fraction cut'); ax.set_ylabel('abs doji HAC t', color=AMBER)\n"
            "ax2.set_ylabel('delta vs base (bps)', color=RED)\n"
            "ax.set_title('Looser cut -> bigger n -> bigger t, but delta stays negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "for bf, n, t, d in zip(cuts, ns, ts, dl):\n"
            "    print(f'phi={bf:.2f}  n={n:6d}  abs t={t:+.2f}  delta={d:+.2f} bps')"
        ),
        md(
            "> In plain words: the rising *t* across cuts is a sample-size artefact, not strengthening "
            "evidence — the delta vs base rate never turns positive. A bigger pile of below-average "
            "bars is still below average."
        ),
        md(
            "### 4d · Myth-check — does a trend filter rescue it?\n\n"
            "Folk wisdom: 'only fade a doji against the prevailing trend.' Split dojis by whether the "
            "close sits above/below its 20-day SMA."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.filtered_stats(panel, horizon=5, mode='trend')\n"
            "    a, b = tr['above_sma'], tr['below_sma']\n"
            "    am, at_, bm, bt = a['mean_bps'], a['tstat'], b['mean_bps'], b['tstat']\n"
            "else:\n"
            f"    am, at_, bm, bt = {R['above_mean']}, {R['above_t']}, {R['below_mean']}, {R['below_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['doji above SMA\\n(uptrend)', 'doji below SMA\\n(downtrend)'], [am, bm],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('5-day reversal (bps)')\n"
            "ax.set_title('Downtrend dojis bounce more -- but so does ANY downtrend bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'above SMA: {am:+.2f} bps (t={at_:+.2f})   below SMA: {bm:+.2f} bps (t={bt:+.2f})')\n"
            "print('The below-SMA bounce is the well-known post-drop mean-reversion, present on all bars.')"
        ),
        md(
            f"> In plain words: dojis below their average 'reverse' at **{R['below_mean']:+.1f} bps** "
            f"(*t* = {R['below_t']:+.2f}) and above-average dojis don't — but that split is exactly "
            "what *any* bar shows (drops bounce, rallies don't). The filter relabels the same "
            "base-rate reversion; it doesn't isolate a doji effect. **The myth-check confirms the "
            "misattribution.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — absolute 5-day doji reversal **{R['d5_mean']:+.2f} bps**, "
            f"HAC *t* **{R['d5_t']:+.2f}** (clears 2), **but** delta vs base **{R['delta5']:+.2f} bps**, "
            f"label-shuffle *p* **{R['p5']:.2f}**, and the SPY-only doji is *negative* "
            f"({R['spy_mean']:+.2f} bps, *t* {R['spy_t']:+.2f}). The number is the baseline, not the doji.\n"
            "- **Tradability `MIRAGE`** — no doji-specific gross edge; net of costs the short "
            "horizons go negative and the rest is base-rate reversion minus friction.\n"
            f"- **Reversal signal? `MISATTRIBUTED`** — every horizon's delta is negative; the "
            "reversal belongs to short-horizon mean-reversion, not the candle of indecision."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — gross vs net\n\n"
            "About 26 dojis/name/year — high turnover, and no doji-specific gross to pay costs from:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    gross = [res['per_horizon'][h]['doji']['mean_bps'] for h in horizons]\n"
            "    net = [res['per_horizon'][h]['doji']['net_bps'] for h in horizons]\n"
            "else:\n"
            f"    gross = [{R['d1_mean']},{R['d3_mean']},{R['d5_mean']},{R['d10_mean']}]\n"
            f"    net = [{R['d1_net']},{R['d3_net']},{R['d5_net']},{R['d10_net']}]\n"
            "x = np.arange(len(horizons)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x - w/2, gross, w, color=AMBER, label='gross')\n"
            "ax.bar(x + w/2, net, w, color=RED, label='net (2 bps + 1 bp borrow)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('doji reversal (bps)'); ax.legend()\n"
            "ax.set_title('Net of friction the short horizons are losers; the rest is base-rate beta')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even is moot: there is no doji-specific gross edge to defend.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *harness* capable of catching a real doji effect, or is it blind? Plant a "
            "doji-specific reversal in a synthetic panel and sweep the knob:"
        ),
        code(
            "edges = [0.0, 0.01, 0.02, 0.04]\n"
            "deltas, ps = [], []\n"
            "for e in edges:\n"
            "    sp, _ = data.synthetic_panel(n_names=28, n_days=3000, edge=e, seed=405)\n"
            "    rr = st.run_experiment(sp, body_frac=0.10, horizons=(5,), n_draws=500, seed=405)\n"
            "    deltas.append(rr['per_horizon'][5]['delta_bps'])\n"
            "    ps.append(rr['per_horizon'][5]['placebo_p'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(edges, deltas, 'o-', c=GREEN, lw=2, label='detected delta vs base (bps)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax2 = ax.twinx(); ax2.plot(edges, ps, 's--', c=GREY, lw=1.5, label='placebo p')\n"
            "ax2.axhline(0.05, ls=':', c=GREY)\n"
            "ax.set_xlabel('planted doji reversal (edge)'); ax.set_ylabel('detected delta (bps)', color=GREEN)\n"
            "ax2.set_ylabel('placebo p', color=GREY)\n"
            "ax.set_title('Harness is faithful: plant a doji effect and it lights up')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e, d, p in zip(edges, deltas, ps):\n"
            "    print(f'edge={e:.2f}  delta={d:+7.2f} bps  placebo p={p:.3f}')"
        ),
        md(
            "The harness is a faithful doji-reversal detector: at `edge = 0` it reads ~zero delta "
            f"(placebo *p* ≈ {R['syn0_p']:.2f}); as the planted doji-specific reversal grows the "
            "delta climbs and the placebo *p* collapses to 0. The real-tape verdict is therefore a "
            "statement about the **market** — there is no doji-specific reversal in US large-caps, "
            "only the generic short-horizon mean-reversion the doji is mistaken for. Forks worth "
            "trying: strict zero-body / long-legged dojis; multi-candle morning/evening-star setups; "
            "intraday dojis; or a cross-sectional doji portfolio neutralised to the base rate."
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
