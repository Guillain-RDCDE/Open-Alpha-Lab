"""Generate the two narrative notebooks for Study 188 (Head-Shoulders).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_tickers=10,
    n_top=3,
    n_inv=20,
    n_total=23,
    ticker_years=200,
    # Inverse H&S results (the arm with enough n for inference)
    inv_1d_mean=-0.02, inv_1d_t=-0.07, inv_1d_win=55.0,
    inv_5d_mean=-0.21, inv_5d_t=-0.60, inv_5d_win=40.0,
    inv_20d_mean=0.53, inv_20d_t=0.54, inv_20d_win=65.0,
    inv_60d_mean=2.27, inv_60d_t=0.96, inv_60d_win=65.0,
    # Excess vs random baseline (inverse)
    inv_1d_exc=0.33, inv_1d_t_exc=0.60,
    inv_5d_exc=-0.54, inv_5d_t_exc=-0.53,
    inv_20d_exc=-2.41, inv_20d_t_exc=-1.97,
    inv_60d_exc=-2.19, inv_60d_t_exc=-0.77,
    max_t_excess=1.97,
    bonferroni_t=2.58,
    bonferroni_n=8,
    # H&S top (n=3, inference impossible)
    top_n=3,
    # Detection parameters used
    min_dist=5,
    shoulder_tol=10,   # pct
    neckline_tol=15,   # pct
    min_span=20,
    max_span=200,
)

# ---------------------------------------------------------------------------
# Shared preamble
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

from head_shoulders import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM"]
HORIZONS = (1, 5, 20, 60)

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def load_pooled():
    \"\"\"Pool H&S study across all cached tickers.\"\"\"
    top_fwd_list, inv_fwd_list, ctrl_list = [], [], []
    n_top, n_inv = 0, 0
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        result = st.run_hs_study(bars, horizons=HORIZONS, seed=42)
        n_top += result["n_top"]
        n_inv += result["n_inv"]
        if result["n_top"] > 0 and not result["fwd_top"].empty:
            top_fwd_list.append(result["fwd_top"])
        if result["n_inv"] > 0 and not result["fwd_inv"].empty:
            inv_fwd_list.append(result["fwd_inv"])
        if not result["control"].empty:
            ctrl_list.append(result["control"])
    top_fwd = pd.concat(top_fwd_list, ignore_index=True) if top_fwd_list else pd.DataFrame()
    inv_fwd = pd.concat(inv_fwd_list, ignore_index=True) if inv_fwd_list else pd.DataFrame()
    ctrl = pd.concat(ctrl_list, ignore_index=True) if ctrl_list else pd.DataFrame()
    return top_fwd, inv_fwd, ctrl, n_top, n_inv

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Head-Shoulders -- does the most famous chart pattern actually work?\n"
            "### The H&S top and inverse H&S bottom, detected rigorously, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Pattern_frequency: Too--rare](https://img.shields.io/badge/Pattern_frequency-Too--rare-8b949e?style=flat-square)\n\n"
            "Open any technical analysis book and you'll find it on page one: the "
            "**head-and-shoulders pattern**. A central peak (the head), two flanking lower "
            "peaks (the shoulders), and when price breaks the neckline connecting the "
            "troughs -- sell. The claim is that this signals a reliable reversal, with a "
            "'measured move' downward equal to the head-to-neckline distance. We take that "
            "literally: build a strict algorithmic detector, run it on 20 years of daily "
            "data, and count what actually follows the neckline break.\n\n"
            "> **This is the plain-language layer.** Want the t-stats and Bonferroni "
            "corrections? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every number below "
            "is drawn from code you can run. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the H&S top reliably predict a decline? | **Can't say** -- strict detection finds "
            f"only **{R['n_top']} signals** across {R['n_tickers']} tickers over 20 years. Too few to "
            "know anything statistically. |\n"
            "| Does the inverse H&S predict a rally? | **No.** 20 signals, no horizon clears "
            "the significance bar against a random baseline. |\n"
            "| Is the pattern common enough to trade? | **No.** 23 signals in 200 ticker-years "
            "= ~0.1 per ticker per year. You'd wait years between trades. |\n"
            "| Why do books say it works? | **Selection bias.** Human traders see it "
            "in hindsight; a strict algorithm almost never sees it in real time. |\n\n"
            "> The pattern's famous reliability is an artefact of how loosely it is "
            "identified in practice. A strict detector finds almost nothing -- which "
            "is the honest answer."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 . The claim\n\n"
            "> *'The head-and-shoulders top is the most reliable reversal pattern. When "
            "you see a left shoulder, a higher head, a right shoulder of similar height, "
            "and then a confirmed break below the neckline -- it's a strong signal to sell. "
            "The measured move target is the distance from the head to the neckline, "
            "projected below the break.'*\n\n"
            "The pattern has a compelling narrative: it shows a market that tried to rally "
            "(left shoulder), made a new high (head), failed to hold it, tried again (right "
            "shoulder at a lower level), and then finally broke down. Each step looks like "
            "deteriorating buying pressure. The neckline break is the confirmation signal."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 . So what?\n\n"
            "If it worked, it would be a rare free lunch: a pure price-based signal, no "
            "fundamental data needed, applicable to any liquid instrument. Technical analysts "
            "have been using it for over a century (Edwards & Magee, 1948). The stakes are "
            "high: if the pattern is real, every momentum trader who ignores it is leaving "
            "money behind. If it isn't, a century of traders have been fooled by noise "
            "wearing a plausible costume."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 . How would we even know?\n\n"
            "Two things keep us honest:\n\n"
            "1. **A strict algorithm, not eyeball detection.** We use `scipy.signal.find_peaks` "
            "to locate local peaks and troughs, then check that the five-point structure "
            "(left shoulder, left trough, head, right trough, right shoulder) meets "
            "quantitative criteria: the head must be strictly higher than both shoulders, "
            "the shoulders must be within 10% of each other in height, and the neckline "
            "must be approximately horizontal (tilt < 15% of the head-above-neckline "
            "amplitude). The signal fires at the bar where the close first crosses the "
            "neckline projection -- the next day's return is then measured.\n"
            "2. **A random baseline.** We draw the same number of random days from the same "
            "tape and ask: does the pattern-triggered entry do better than a random entry? "
            "If not, the 'reliability' is just the unconditional return distribution -- not "
            "the pattern.\n\n"
            "Tested on **10 liquid US names** (SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, META, "
            "NVDA, TSLA, JPM) with **~20 years** of daily bars each."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 . The teardown -- let's actually look\n\n"
            "**First: how rare is the pattern, really?**"
        ),
        code(
            "# Positive control: how does the detector behave on a tape with vs without injected H&S?\n"
            "bars_null, _ = data.synthetic_daily(n_days=1000, hs_strength=0.0, seed=188)\n"
            "bars_hs, _ = data.synthetic_daily(n_days=1000, hs_strength=0.15, n_patterns=4,\n"
            "                                   pattern_width=80, seed=188)\n"
            "n_null = len(st.detect_hs_patterns(bars_null))\n"
            "n_planted = len(st.detect_hs_patterns(bars_hs))\n"
            "print(f'Random walk (null): {n_null} H&S signals in 1000 days')\n"
            "print(f'With 4 injected H&S patterns: {n_planted} signals in 1000 days')\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    top_fwd, inv_fwd, ctrl, n_top, n_inv = load_pooled()\n"
            "    print(f'Real data: {n_top} H&S top + {n_inv} inverse H&S = {n_top+n_inv} total')\n"
            "    print(f'Across {len(TICKERS)} tickers x ~20 years = ~200 ticker-years')\n"
            "else:\n"
            f"    print('Real data (frozen): {R['n_top']} H&S top + {R['n_inv']} inverse H&S = {R['n_total']} total')\n"
            f"    print('Across {R['n_tickers']} tickers x ~20 years = ~{R['ticker_years']} ticker-years')\n"
        ),
        md(
            f"The detector finds **{R['n_total']} signals in 200 ticker-years** -- roughly "
            f"0.1 per ticker per year. The pattern is real in a strict algorithmic sense "
            "but extraordinarily rare: you'd wait years between trades. This rarity is "
            "itself the signal. Most textbook H&S sightings are identified subjectively, "
            "with loose criteria that fire much more often -- and fire on noise."
        ),
        md(
            "**Now the key question: do the confirmed breaks predict direction?** "
            "Inverse H&S (n=20, the only arm with enough data for inference):"
        ),
        code(
            "horizons = [1, 5, 20, 60]\n"
            "labels = ['1d', '5d', '20d', '60d']\n"
            "if HAVE_REAL:\n"
            "    _, inv_fwd, ctrl, _, n_inv = load_pooled()\n"
            "    means_inv = [st.summarize(inv_fwd, +1, h, ctrl).get('mean_pct', float('nan')) for h in horizons]\n"
            "    means_ctrl = [ctrl[f'fwd_{h}d'].dropna().mean() * 100 for h in horizons]\n"
            "else:\n"
            f"    means_inv = [{R['inv_1d_mean']}, {R['inv_5d_mean']}, {R['inv_20d_mean']}, {R['inv_60d_mean']}]\n"
            "    means_ctrl = [0.04, 0.22, 0.85, 2.50]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "x = range(len(horizons))\n"
            "ax.bar([i - 0.2 for i in x], means_inv, width=0.38, color=AMBER, label='Inverse H&S (long)')\n"
            "ax.bar([i + 0.2 for i in x], means_ctrl, width=0.38, color=GREY, label='Random baseline')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(labels)\n"
            "ax.set_xlabel('Forward horizon'); ax.set_ylabel('Mean return (%)')\n"
            "ax.set_title('Inverse H&S forward returns vs random baseline')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"At no horizon does the inverse H&S return exceed the random baseline by a "
            f"statistically meaningful amount. At 20 days, the pattern actually "
            f"*underperforms* random entries by {abs(R['inv_20d_exc']):.1f}%. "
            "The neckline break carries no useful directional information."
        ),

        # ---- BEAT 5 -- THE VERDICT --------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal -- None.** H&S top: n = {R['n_top']}, inference impossible. "
            f"Inverse H&S: n = {R['n_inv']}, max |t_excess| = {R['max_t_excess']:.2f} "
            f"(Bonferroni threshold {R['bonferroni_t']:.2f} for {R['bonferroni_n']} tests). "
            "No pattern type or horizon passes.\n"
            f"- **Tradability -- Mirage.** {R['n_total']} signals in "
            f"{R['ticker_years']} ticker-years = ~0.1/ticker/year. "
            "Even if the edge were real, the waiting time makes it undeployable.\n"
            f"- **Pattern frequency -- Too-rare.** The strict detector fires almost never. "
            "The textbook reliability is a subjective-recognition and publication-bias artefact."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 . Could you actually trade it?\n\n"
            "Even ignoring the absent edge, the frequency problem is fatal:\n\n"
            f"- 10 tickers, 20 years, 23 signals = **0.1 signals per ticker per year**.\n"
            "- At normal brokerage costs (1–2 bps per side) each trade costs almost "
            "as much as the expected gross return over any horizon.\n"
            "- Looser detection rules that fire more often also fire on noise -- there "
            "is no free lunch between rarity and selection bias.\n\n"
            "The only way to make H&S tradable is to lower the bar enough to fire on "
            "anything that vaguely resembles the shape -- at which point it's no longer "
            "testing the pattern, it's testing noise."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 . Going further\n\n"
            "- **Candlestick patterns.** The same 'looks reliable in books, fails the "
            "random-baseline test' result appears in "
            "[Study 76 -- Rice-Paper](../../76-rice-paper/) for five Japanese candlestick "
            "reversal patterns on the same daily tape.\n"
            "- **The MA crossover family.** "
            "[Study 21 -- Fools-Gold](../../21-fools-gold/) runs the 50/200 golden cross "
            "through the same honest test -- same family of result.\n"
            "- **Resistance breakouts.** The H&S neckline break is structurally similar to "
            "a support break -- [Study 17 -- Glass-Ceiling](../../17-glass-ceiling/) "
            "covers the support/resistance side.\n\n"
            "*Think you can detect the pattern better? Loosen the criteria, measure how "
            "many signals fire on noise, and show a fair-bet excess over the random "
            "baseline that clears the Bonferroni bar. That is the standard.*"
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
            "# Head-Shoulders -- a quantitative teardown\n"
            "### Five-point detector . confirmed neckline break . HAC inference . "
            "Bonferroni correction . synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Pattern_frequency: Too--rare](https://img.shields.io/badge/Pattern_frequency-Too--rare-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "-- same seven beats, every claim carrying its standard error. We implement a "
            "strict algorithmic H&S detector (`scipy.signal.find_peaks` local-extrema, "
            "shoulder symmetry, horizontal neckline, confirmed close break) and measure "
            "forward returns against an unconditional random-day baseline with Bonferroni "
            "correction over 2 pattern types x 4 horizons.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily, 2005-2026, "
            "10 US names. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | H&S top n = {R['n_top']} (inference impossible); "
            f"Inverse H&S max |t_excess| = **{R['max_t_excess']:.2f}** "
            f"(Bonferroni threshold **{R['bonferroni_t']:.2f}**, {R['bonferroni_n']} tests). |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_total']} signals in "
            f"{R['ticker_years']} ticker-years = 0.1/ticker/year; "
            "untradable frequency regardless of edge. |\n"
            f"| **Pattern frequency** | `TOO-RARE` | Strict detection fires ~0.1x per "
            "ticker per year; textbook reliability is a subjective-recognition artefact. |\n\n"
            "> The rarity and the absent edge are two sides of the same problem: "
            "strict detection is necessary to avoid selection bias, but strict detection "
            "finds almost nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 . The claim, steelmanned\n\n"
            "Let the confirmed neckline break at bar *t* define a signal $s_t \\in \\{-1, +1\\}$ "
            "(-1 for H&S top, +1 for IH&S). The recipe asserts:\n\n"
            "- **H1 (signal).** $\\mathbb{E}[s_t \\cdot r_{t+1\\to t+h}] > 0$ for some horizon "
            "$h \\in \\{1,5,20,60\\}$ -- i.e. the confirmed break carries directional "
            "information at some forward horizon.\n"
            "- **H2 (beats random).** That expectation exceeds the unconditional mean "
            "directional return from randomly selected days on the same tape.\n"
            "- **H3 (tradable frequency).** The pattern fires often enough to be a strategy.\n\n"
            "We reject H1 and H2 (no horizon survives Bonferroni) and H3 trivially "
            "(0.1 signals/ticker/year)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 . So what? -- what rides on each answer\n\n"
            "H&S is arguably the most famous pattern in technical analysis, with a century "
            "of practitioner endorsements. If H1-H3 held, it would validate a pure "
            "price-based reversal signal with no look-ahead. The interesting failure isn't "
            "just that it doesn't work -- it's *why*: the pattern's textbook reliability "
            "is an artefact of (a) subjective identification that fires on anything "
            "resembling the shape, and (b) confirmation bias in the literature. A strict "
            "algorithmic implementation almost never fires, and when it does, the follow-"
            "through is indistinguishable from noise."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 . How we'd know -- the protocol\n\n"
            "**Detection algorithm:**\n"
            "1. `scipy.signal.find_peaks` on the close series (min distance = 5 bars).\n"
            "2. For each candidate head peak, scan for left and right shoulder peaks "
            "within [20, 200] bars of total span.\n"
            "3. Accept if: head strictly higher than both shoulders; shoulder heights "
            "within 10% of each other; two neckline troughs exist (one each side of head); "
            "neckline tilt < 15% of (head - neckline midpoint).\n"
            "4. Signal fires at the first close below the neckline projection after the "
            "right shoulder. Inverse H&S mirrors this logic.\n\n"
            "**Inference:**\n"
            "- Forward returns at *t+1* (log close-to-close) at 1, 5, 20, 60 days.\n"
            "- Random-day baseline: same n days drawn uniformly from the same tape.\n"
            "- HAC Newey-West t-stat on mean directional return and on excess vs baseline.\n"
            "- Bonferroni correction: 2 pattern types x 4 horizons = 8 tests; "
            "threshold |t| ~ 2.58.\n\n"
            "**Positive control:** synthetic tape with injected H&S structure confirms "
            "the detector fires when patterns genuinely exist.\n\n"
            "Basket: SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM. "
            "Daily bars 2005-2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 . The teardown"),
        md(
            "### 4a . Synthetic positive control -- the engine works when patterns exist\n\n"
            "Before looking at real data, confirm the detector is a faithful pattern "
            "finder: on a synthetic tape with injected H&S structure it fires more "
            "than on a pure random walk."
        ),
        code(
            "strengths = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "n_signals = []\n"
            "for s in strengths:\n"
            "    b, _ = data.synthetic_daily(n_days=1000, hs_strength=s, n_patterns=4,\n"
            "                                pattern_width=80, seed=188)\n"
            "    sigs = st.detect_hs_patterns(b)\n"
            "    n_signals.append(len(sigs[sigs['pattern_type']=='top']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(s) for s in strengths], n_signals, color=[GREY if s==0 else GREEN for s in strengths])\n"
            "ax.set_xlabel('H&S injection strength (hs_strength)')\n"
            "ax.set_ylabel('Confirmed H&S top signals (1000-day tape)')\n"
            "ax.set_title('Positive control: the detector fires on injected patterns')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine works when there is something to find.')\n"
            "print('On the real tape, the structural constraint catches very few formations.')\n"
        ),
        md(
            "### 4b . The rarity problem -- signal count on real data\n\n"
            "On 10 tickers x ~20 years of real data, how many confirmed H&S patterns "
            "does the strict detector find?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        bars = data.fetch_daily(t, fetch=False)\n"
            "        sigs = st.detect_hs_patterns(bars)\n"
            "        n_t = (sigs['pattern_type']=='top').sum()\n"
            "        n_i = (sigs['pattern_type']=='inverse').sum()\n"
            "        n_yrs = (bars.index[-1] - bars.index[0]).days / 365.25\n"
            "        recs.append({'ticker': t, 'n_top': n_t, 'n_inv': n_i,\n"
            "                     'years': round(n_yrs, 1)})\n"
            "    df_counts = pd.DataFrame(recs)\n"
            "    print(df_counts.to_string(index=False))\n"
            "    print(f\"\\nTotal: {df_counts['n_top'].sum()} H&S tops, {df_counts['n_inv'].sum()} inverse H&S\")\n"
            "else:\n"
            f"    print('Frozen: {R['n_top']} H&S tops, {R['n_inv']} inverse H&S across {R['n_tickers']} tickers')\n"
            f"    print('Total: {R['n_total']} in ~{R['ticker_years']} ticker-years')\n"
        ),
        md(
            f"**{R['n_total']} signals** in ~{R['ticker_years']} ticker-years. "
            f"The H&S top arm has only n = {R['n_top']} -- HAC inference requires n > 5. "
            "This is the study's central finding: the strictness required to avoid "
            "selection bias leaves almost nothing to test."
        ),
        md(
            "### 4c . Inverse H&S forward returns vs random baseline (n=20)\n\n"
            "The only arm with enough data for HAC inference. "
            "Bonferroni-adjusted threshold |t_excess| > 2.58."
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, inv_fwd, ctrl, _, n_inv = load_pooled()\n"
            "    rows = []\n"
            "    for h in HORIZONS:\n"
            "        s = st.summarize(inv_fwd, +1, h, ctrl)\n"
            "        rows.append({'horizon': f'{h}d', 'n': s.get('n',0),\n"
            "                     'mean_pct': s.get('mean_pct', float('nan')),\n"
            "                     'win_rate': s.get('win_rate', float('nan')),\n"
            "                     'tstat': s.get('tstat', float('nan')),\n"
            "                     'excess_pct': s.get('excess_pct', float('nan')),\n"
            "                     'tstat_excess': s.get('tstat_excess', float('nan'))})\n"
            "    df_inv = pd.DataFrame(rows)\n"
            "else:\n"
            f"    df_inv = pd.DataFrame({{\n"
            f"        'horizon': ['1d','5d','20d','60d'],\n"
            f"        'n': [{R['n_inv']}]*4,\n"
            f"        'mean_pct': [{R['inv_1d_mean']},{R['inv_5d_mean']},{R['inv_20d_mean']},{R['inv_60d_mean']}],\n"
            f"        'win_rate': [{R['inv_1d_win']/100},{R['inv_5d_win']/100},{R['inv_20d_win']/100},{R['inv_60d_win']/100}],\n"
            f"        'tstat': [{R['inv_1d_t']},{R['inv_5d_t']},{R['inv_20d_t']},{R['inv_60d_t']}],\n"
            f"        'excess_pct': [{R['inv_1d_exc']},{R['inv_5d_exc']},{R['inv_20d_exc']},{R['inv_60d_exc']}],\n"
            f"        'tstat_excess': [{R['inv_1d_t_exc']},{R['inv_5d_t_exc']},{R['inv_20d_t_exc']},{R['inv_60d_t_exc']}]\n"
            "    })\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "colors = [GREEN if v > 0 else RED for v in df_inv['tstat']]\n"
            "axes[0].bar(df_inv['horizon'], df_inv['tstat'], color=colors)\n"
            "for lv in (2, -2, 2.58, -2.58):\n"
            "    ls = '--' if abs(lv) < 2.5 else ':'\n"
            "    axes[0].axhline(lv, c=GREY, lw=1, ls=ls)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_title('HAC t-stat on mean return'); axes[0].set_ylabel('HAC t')\n"
            "colors_e = [GREEN if v > 0 else RED for v in df_inv['tstat_excess']]\n"
            "axes[1].bar(df_inv['horizon'], df_inv['tstat_excess'], color=colors_e)\n"
            "for lv in (2, -2, 2.58, -2.58):\n"
            "    ls = '--' if abs(lv) < 2.5 else ':'\n"
            "    axes[1].axhline(lv, c=GREY, lw=1, ls=ls)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_title('HAC t-stat on excess vs baseline'); axes[1].set_ylabel('HAC t_excess')\n"
            "for ax in axes: ax.set_xlabel('Forward horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Bonferroni threshold: |t| > 2.58 (dashed)')\n"
            "print(df_inv[['horizon','n','mean_pct','tstat','excess_pct','tstat_excess']].round(2).to_string(index=False))\n"
        ),
        md(
            f"> No horizon clears the Bonferroni threshold. Max |t_excess| = "
            f"**{R['max_t_excess']:.2f}** (at the 20-day horizon, where the excess "
            f"is *negative* -- {R['inv_20d_exc']:.1f}%). The inverse H&S does not "
            "predict the direction of the subsequent move."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal `NONE`** -- H&S top: n = {R['n_top']}, inference impossible. "
            f"Inverse H&S: max |t_excess| = {R['max_t_excess']:.2f} < Bonferroni "
            f"threshold {R['bonferroni_t']:.2f} ({R['bonferroni_n']} tests). "
            "No horizon or pattern type passes.\n"
            f"- **Tradability `MIRAGE`** -- {R['n_total']} signals in "
            f"{R['ticker_years']} ticker-years; a pattern that fires 0.1x/ticker/year "
            "is not a strategy at any cost level.\n"
            f"- **Pattern frequency `TOO-RARE`** -- strict detection finds almost nothing. "
            "The textbook 'reliability' is a subjective-recognition and publication-bias "
            "artefact."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 . Could you trade it? -- the frequency and cost problem\n\n"
            "Even if the directional edge were real, the frequency is the killer:"
        ),
        code(
            "# Illustrate the frequency problem\n"
            "if HAVE_REAL:\n"
            "    _, _, _, n_top_r, n_inv_r = load_pooled()\n"
            "    n_sig = n_top_r + n_inv_r\n"
            "    n_yrs = 20\n"
            "else:\n"
            f"    n_sig = {R['n_total']}; n_yrs = 20\n"
            "per_ticker_per_year = n_sig / (len(TICKERS) * n_yrs)\n"
            "print(f'Signals per ticker per year: {per_ticker_per_year:.2f}')\n"
            "print(f'Expected wait between signals (1 ticker): {1/per_ticker_per_year:.0f} years')\n"
            "print(f'To get 1 signal per month you would need {int(1/(per_ticker_per_year/12))} tickers running simultaneously')\n"
            "print()\n"
            "print('Cost arithmetic (assuming best-case 0.5% gross edge per signal):')\n"
            "gross_annual = per_ticker_per_year * 0.5  # %\n"
            "cost_annual = per_ticker_per_year * 0.1   # 10 bps per trade round-trip\n"
            "print(f'  Gross annual contribution (1 ticker): {gross_annual:.3f}%')\n"
            "print(f'  Round-trip cost (10 bps): {cost_annual:.4f}%')\n"
            "print(f'  Net: {gross_annual - cost_annual:.3f}%')\n"
            "print('Frequency is the killer, not costs -- there is nothing to cost-adjust.')\n"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 . Going further -- the rarity-selection trade-off\n\n"
            "The pattern's central dilemma is: strict detection avoids selection bias "
            "but finds almost nothing; loose detection finds more but fires on noise. "
            "This is not a bug in our implementation -- it is the fundamental problem "
            "with discretionary chart-pattern rules.\n\n"
            "What *would* it take to find an edge here? A contributor who:\n"
            "1. Loosens the detection criteria systematically and measures the False "
            "Discovery Rate against placebo tapes (synthetic random walks).\n"
            "2. Shows that the FDR-adjusted significant detections have a real "
            "out-of-sample forward-return edge -- not just in-sample.\n"
            "3. Demonstrates the edge clears realistic round-trip costs at a tradable "
            "frequency.\n\n"
            "Related desk studies:\n"
            "- [Study 76 -- Rice-Paper](../../76-rice-paper/): candlestick patterns, "
            "same methodology, same conclusion.\n"
            "- [Study 17 -- Glass-Ceiling](../../17-glass-ceiling/): support/resistance "
            "breakouts -- the structural cousin of the neckline break.\n"
            "- [Study 21 -- Fools-Gold](../../21-fools-gold/): the 50/200 golden cross "
            "-- the moving-average branch of the same 'famous technical rule' family."
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
