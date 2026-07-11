"""Generate the two narrative notebooks for Study 700 (Bat-Harmonic).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily basket
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily bars,
# SPY/QQQ/AAPL/MSFT/NVDA 2001-07-10 -> 2026-06-30, TSLA 2010-06-29 -> 2026-06-30;
# pct=0.02 zigzag, AB retrace 0.382-0.50 of XA, BC retrace 0.382-0.886 of AB,
# D = A - 0.886*(A-X), 120-session touch window).
R = dict(
    basket=("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"),
    asof="2026-06-30",
    n_pivots=6055, n_cand_bat=118, n_cand_plac=118,
    n_bat=89, n_plac=92, n_base=1780, base_seeds=20,
    bat={1: dict(hit=51.7, mean=-12.41, t=-0.46), 5: dict(hit=40.4, mean=7.58, t=0.17),
         10: dict(hit=44.9, mean=-36.93, t=-0.48)},
    wilson_bat=(30.9, 50.8),
    per_instrument={
        "SPY": dict(bat_n=4, bat_mean=33.59, bat_t="n/a (n<6)",
                    base_mean=-45.95, base_t=1.61, plac_n=3, plac_mean=-66.06),
        "QQQ": dict(bat_n=10, bat_mean=-45.68, bat_t=-0.41,
                    base_mean=-32.07, base_t=-0.11, plac_n=11, plac_mean=114.72),
        "AAPL": dict(bat_n=19, bat_mean=-119.46, bat_t=-1.61,
                     base_mean=-8.70, base_t=-1.54, plac_n=19, plac_mean=-79.11),
        "MSFT": dict(bat_n=15, bat_mean=100.13, bat_t=1.01,
                     base_mean=-28.61, base_t=1.36, plac_n=16, plac_mean=125.90),
        "TSLA": dict(bat_n=14, bat_mean=202.77, bat_t=2.34,
                     base_mean=-34.00, base_t=1.73, plac_n=14, plac_mean=-86.09),
        "NVDA": dict(bat_n=27, bat_mean=-39.77, bat_t=-0.46,
                     base_mean=13.05, base_t=-0.50, plac_n=29, plac_mean=-36.99),
    },
    base_welch={1: -0.04, 5: 0.45, 10: -0.63},
    base_mean_h={1: -11.32, 5: -13.74, 10: 14.48},
    bonf_n_tests=7, bonf_thr=2.69, bonf_n_uncorrected=0, bonf_n_survive=0,
    plac_welch={1: 0.05, 5: 0.22, 10: -0.12},
    plac_mean_h={1: -14.45, 5: -7.64, 10: -23.22},
    n_beats_plac=2,
    cost_sweep={
        1: {0.0: (-12.41, -0.46), 5.0: (-22.41, -0.84), 10.0: (-32.41, -1.21)},
        3: {0.0: (2.50, 0.07), 5.0: (-7.50, -0.20), 10.0: (-17.50, -0.47)},
        5: {0.0: (7.58, 0.17), 5.0: (-2.42, -0.05), 10.0: (-12.42, -0.27)},
        10: {0.0: (-36.93, -0.48), 5.0: (-46.93, -0.61), 10.0: (-56.93, -0.74)},
        20: {0.0: (-143.17, -1.07), 5.0: (-153.17, -1.14), 10.0: (-163.17, -1.22)},
    },
    syn_null_mean=-0.12, syn_null_sd=1.24, syn_null_fire=2, syn_null_seeds=20,
    syn_planted_n=84, syn_planted_mean=126.21, syn_planted_t=5.18, syn_planted_knob=0.12,
    fp={"SPY": "58d0459a7599", "QQQ": "c48f22566e73", "AAPL": "75ce4521e4f7",
        "MSFT": "fb9333ad5b2b", "TSLA": "f8ca92e420b8", "NVDA": "1e614f1ea32c"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Most_reliable%3F: Busted](https://img.shields.io/badge/Most_reliable%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from bat_harmonic import data, strategy as st

BASKET = data.BASKET
HAVE_REAL = data.have_real()
BARS = {t: data.load_real(t) for t in BASKET} if HAVE_REAL else {}

def pooled(placebo=False, cost=0.0, seed=700, pct=0.02, horizons=(1, 5, 10)):
    \"\"\"Pool the D-touch ledger across the whole basket (offline; reads BARS).\"\"\"
    frames = []
    for t in BASKET:
        _, _, ledger = st.detect_and_scan(BARS[t], pct=pct, placebo=placebo, seed=seed,
                                          cost_bps=cost, horizons=horizons)
        if not ledger.empty:
            frames.append(ledger)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def base_rate_pool(ledger_by_ticker, n_seeds=20, horizons=(1, 5, 10), cost=0.0):
    frames = []
    for t in BASKET:
        l = ledger_by_ticker.get(t, pd.DataFrame())
        n_ev = len(l)
        if n_ev == 0:
            continue
        mix = float((l["reversal_dir"] > 0).mean())
        for s in range(n_seeds):
            br = st.base_rate_ledger(BARS[t], n_ev, mix, horizons=horizons, cost_bps=cost,
                                     seed=st._seed_from(700, t, s))
            frames.append(br)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

print("real cache present:", HAVE_REAL, "| basket:", BASKET)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is the \"safest\" chart pattern actually safe? 🦇↩️\n"
            "### The Bat harmonic — Scott Carney's self-declared *most reliable* pattern, "
            "put to the test\n\n"
            + BADGES +
            "Harmonic-pattern traders draw a five-point zig-zag on a chart — **X, A, B, C, D** — "
            "and look for very specific Fibonacci ratios between the legs. The **Bat**, named and "
            "popularized by Scott Carney, has one defining signature: point **D retraces 86.6%... "
            "no, precisely **88.6%** of the very first leg (X to A)** — a *deep* pullback, but one "
            "that (crucially) never quite reaches all the way back to X. Carney calls this the "
            "**most reliable** pattern in the whole harmonic family, because a failed Bat is cheap "
            "to bail out of: your stop just sits a hair past X.\n\n"
            "That's a strong, specific, falsifiable claim. We test it exactly as stated.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Every pivot is detected with a 2% zigzag and only counted once "
            "it's *confirmed* — never using information from the future. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price reverse at a completed Bat point D? | **Not detectably.** Betting on "
            f"the reversal (5-day hold) earned **{R['bat'][5]['mean']:+.1f} bps per event** on "
            f"average across {R['n_bat']} real detections since 2001/2010 — indistinguishable "
            "from zero, with a hit rate *below* a coin flip. |\n"
            "| Is that better than just buying random dips of the same size? | **No — the "
            f"two are statistically the same** (Welch t = {R['base_welch'][5]:+.2f} vs a "
            "drift-matched random-day control). |\n"
            "| We checked 7 different ways to slice the basket — does any of them survive? | "
            f"**No — 0 of 7**, even before the penalty for looking 7 times gets applied. |\n"
            "| Is 0.886 actually special, or would any deep pullback zone near X have worked? | "
            "**Not special.** A control using the identical pivots but a *random* nearby target "
            f"beat the real 0.886 ratio on 4 of 6 tickers. |\n"
            f"| Can you trade it? | **No — it loses money after a single 5 bps cost.** No hold "
            "length we tried survives realistic trading costs. |\n\n"
            "> The pattern is drawn with real precision. It just doesn't predict anything."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When XA, B, and C confirm the Bat's ratios — B a shallow 38-50% pullback of "
            "XA, C somewhere between 38% and 89% of AB — project D at exactly 88.6% back toward "
            "X. Because D never quite reaches X, the pattern gives you a tight, cheap invalidation "
            "point. That's why it's the *most reliable* harmonic — the risk/reward is the best in "
            "the family.\"*\n\n"
            "It's the retail-technical-analysis equivalent of a "
            "\"best in class\" product claim: not just *does it work*, but *does it work better "
            "than its cousins* (Gartley, Butterfly, Crab — see the going-further section). We test "
            "both halves."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is one of the cleanest possible trading signals in the harmonic zoo: a "
            "**specific, computable price level**, known the moment the third pivot confirms, with "
            "a tight, well-defined stop just past X. Entire harmonic-pattern trading courses and "
            "auto-scanners (TradingView, MetaTrader, Thinkorswim) are built around exactly this "
            "kind of setup — and the Bat is usually pitched as the *entry-level, safest* one to "
            "learn first. If it's just numerology dressed up with a precise-sounding decimal, "
            "though, that safety pitch is worse than useless: it's false confidence with real "
            "money behind it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Find real Bat setups.** Scan {', '.join(R['basket'])} daily bars (2001/2010→"
            "2026) for every confirmed X-A-B-C swing where AB retraces XA by 38-50%, BC retraces "
            "AB by 38-89%, then project D at 88.6% back toward X.\n"
            "- **Wait for D to actually be touched** — up to 120 trading sessions — the same way a "
            "real trader would watch and wait.\n"
            "- **Bet on the reversal**, and measure what actually happened over the next 5 days.\n"
            "- **The first control: a matched random day.** Any directional bet on a rising market "
            "picks up some of the market's own drift for free. So we also draw random entry days "
            "with the *same* bullish/bearish mix as the real Bat signals, and compare.\n"
            "- **The second control: a random target near X.** Rerun the identical scan, but swap "
            "88.6% for a random nearby retracement/extension. If 0.886 is magic, it has to beat "
            "this — not just look positive."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            f"**First, the headline.** {R['n_bat']} real Bat detections since 2001/2010 — average "
            "5-day return from betting on the reversal, at three different hold lengths."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat = pooled(placebo=False)\n"
            "    hs = [1, 5, 10]\n"
            "    ms = {h: bat[f'ret_gross_{h}'].mean() * 1e4 if len(bat) else np.nan for h in hs}\n"
            "else:\n"
            "    hs = [1, 5, 10]\n"
            "    ms = {h: R['bat'][h]['mean'] for h in hs}\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "cols = [RED if h == 5 else GREY for h in hs]\n"
            "ax.bar([f'{h}-day' for h in hs], [ms[h] for h in hs], color=cols, width=.55)\n"
            "for i,h in enumerate(hs): ax.annotate(f'{ms[h]:+.1f} bps',(i,ms[h]),ha='center',\n"
            "    va='top' if ms[h]<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg return from betting on the reversal (bps)')\n"
            "ax.set_title('The Bat fade is a coin flip at best, at every hold length')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: round(ms[h],1) for h in hs})"
        ),
        md(
            f"The 5-day headline: **{R['bat'][5]['mean']:+.2f} bps per event** — essentially "
            f"nothing — with a hit rate of **{R['bat'][5]['hit']:.1f}%**, *below* a coin flip. "
            "The 1-day and 10-day versions are outright negative.\n\n"
            "**Is that just \"no better than a coin flip\", or actually *worse* than the market's "
            "own drift?** This is where the matched random-day control earns its keep:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat_l = pooled(placebo=False)\n"
            "    base_l = base_rate_pool({t: st.detect_and_scan(BARS[t], pct=0.02, cost_bps=0.0)[2] for t in BASKET})\n"
            "    bm = bat_l['ret_gross_5'].mean()*1e4 if len(bat_l) else np.nan\n"
            "    rm = base_l['ret_gross_5'].mean()*1e4 if len(base_l) else np.nan\n"
            "else:\n"
            "    bm, rm = R['bat'][5]['mean'], R['base_mean_h'][5]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.3))\n"
            "ax.bar(['Bat D-touch\\n(n={})'.format(R['n_bat']), 'random day,\\nmatched mix'],\n"
            "       [bm, rm], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([bm, rm]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('5-day mean return (bps)')\n"
            "ax.set_title('The Bat fade is statistically the same as a random matched-mix dip-buy')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Bat {bm:+.1f} bps  vs  random-day control {rm:+.1f} bps')"
        ),
        md(
            f"Practically identical — and the difference (Welch *t* = {R['base_welch'][5]:+.2f}) "
            "isn't remotely statistically real. **We also sliced the basket seven ways** (the "
            "pooled number plus each of the six tickers on its own) — the honest thing to do is "
            "admit we looked seven times and pay the Bonferroni penalty for it:"
        ),
        code(
            "tickers = list(R['per_instrument'])\n"
            "wts = [R['per_instrument'][t]['base_t'] for t in tickers]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if abs(w) >= R['bonf_thr'] else GREY for w in wts]\n"
            "ax.bar(tickers, wts, color=cols, width=.55)\n"
            "ax.axhline(2.0, ls='--', c=AMBER, lw=1, label='uncorrected bar (t=2)')\n"
            "ax.axhline(-2.0, ls='--', c=AMBER, lw=1)\n"
            f"ax.axhline({R['bonf_thr']}, ls='--', c=RED, lw=1.4, "
            f"label='Bonferroni bar (t={R['bonf_thr']})')\n"
            f"ax.axhline(-{R['bonf_thr']}, ls='--', c=RED, lw=1.4)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (Bat vs matched random day)')\n"
            "ax.set_title('None of the 6 per-ticker tests clear even the UNCORRECTED bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(dict(zip(tickers, [round(w,2) for w in wts])))"
        ),
        md(
            f"Zero of the six per-ticker tests, plus the pooled headline, clear even the naive "
            f"*t* = 2 bar — **0 of 7 tests survive**, whether or not you apply the Bonferroni "
            "correction (which raises the bar further, to "
            f"**{R['bonf_thr']:.2f}**, once you account for having looked seven times).\n\n"
            "**Finally, the second control: is 0.886 actually the special number, or would any "
            "nearby target have done just as well?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat_l = pooled(placebo=False); plac_l = pooled(placebo=True)\n"
            "    bm = bat_l['ret_gross_5'].mean()*1e4 if len(bat_l) else np.nan\n"
            "    pm = plac_l['ret_gross_5'].mean()*1e4 if len(plac_l) else np.nan\n"
            "else:\n"
            "    bm, pm = R['bat'][5]['mean'], R['plac_mean_h'][5]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.3))\n"
            "ax.bar(['Bat (0.886)', 'placebo\\n(random nearby target)'], [bm, pm],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([bm, pm]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('5-day mean return (bps)')\n"
            "ax.set_title('0.886 shows no edge over an arbitrary nearby target')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Bat {bm:+.1f} bps  vs  placebo {pm:+.1f} bps')"
        ),
        md(
            f"The real 0.886 ratio beats its own placebo control on only **{R['n_beats_plac']} of "
            "6** tickers — no better than a coin flip. There is nothing detectably special about "
            "Carney's specific number here.\n\n"
            "**And does it at least survive costs?** No — the best gross case (5-day, "
            f"{R['bat'][5]['mean']:+.1f} bps) turns **negative** the instant you charge a single 5 "
            "bps one-way cost, and every longer hold we tried loses more, not less."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The Bat fade returned **{R['bat'][5]['mean']:+.2f} bps/event** "
            f"(HAC t = {R['bat'][5]['t']:+.2f}) with a hit rate below a coin flip, and none of 7 "
            "Bonferroni-corrected comparisons against a drift-matched random-day control survive.\n"
            "- **Tradability — Mirage.** No hold length clears significance on the positive side, "
            "and the best case turns negative at a single 5 bps cost.\n"
            "- **\"Beats a placebo, i.e. is 0.886 the most reliable ratio?\" — Busted.** The Bat's "
            "own defining number wins on only 2 of 6 tickers against an arbitrary nearby target."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **\"Most reliable\" is a comparative claim, and this study measures it directly.** "
            "Carney doesn't just claim the Bat works — he claims it works *better* than Gartley, "
            "Butterfly and Crab. See [698-abcd-harmonic](../../698-abcd-harmonic/) (the bare "
            "AB=CD skeleton), [699-butterfly-harmonic](../../699-butterfly-harmonic/) (D "
            "*overshoots* X instead of staying inside it) and "
            "[468-gartley-harmonic](../../468-gartley-harmonic/) (a shallower 0.786 D) for the "
            "rest of the comparison — none of them clears its own bar either.\n"
            "- **Sibling study:** [77-golden-mean](../../77-golden-mean/) tests plain Fibonacci "
            "retracement *levels* (not a multi-pivot pattern) on the same six tapes and reaches "
            "the identical verdict shape: Fibonacci ratios show no specificity over a randomized "
            "control.\n\n"
            "*Think you can find the real edge in a stricter version of the pattern (a tighter "
            "0.886 tolerance, an added confluence check)? Show it beats this exact placebo — same "
            "pivots, same tape, only the ratio changes — and beats the drift-matched base rate "
            "after a Bonferroni correction, and we'll take a look.*"
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
            "# The Bat Harmonic — a quantitative teardown 🔬\n"
            "### Confirmed-pivot zigzag detection · a drift-matched random-day base rate, "
            "Bonferroni-corrected · a seeded off-0.886 placebo arm · HAC/Welch splits · a cost "
            "sweep · a 20-seed synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **a completed Bat XABCD (D retraces 0.886 of the original XA leg) predicts a "
            "reversal, and is the *most reliable* harmonic** — is testable literally: detect it in "
            "real time off confirmed pivots, fade it, and compare against **two** independent "
            "controls built for the two halves of the claim.\n\n"
            "> ⚠️ **Data note.** Daily OHLC, yfinance, cached; basket "
            f"{', '.join(R['basket'])} (SPY/QQQ/AAPL/MSFT/NVDA 2001-07-10→2026-06-30, TSLA "
            "2010-06-29→2026-06-30 — identical basket to siblings "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) and "
            "[699-butterfly-harmonic](../../699-butterfly-harmonic/)). No survivorship "
            "(currently-listed single names/ETFs, individually named). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints " +
            ", ".join(f"`{v}`" for v in R["fp"].values()) + ").\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 5-day fade-at-D: **{R['bat'][5]['mean']:+.2f} bps/event**, "
            f"HAC **t = {R['bat'][5]['t']:.2f}** (n={R['n_bat']}); vs base rate Welch "
            f"t = {R['base_welch'][5]:+.2f}; **0/{R['bonf_n_tests']}** Bonferroni-corrected tests "
            f"survive (critical \\|t\\| = {R['bonf_thr']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | best gross case ({R['cost_sweep'][5][0.0][0]:+.1f} "
            f"bps, 5-day) turns net-negative at 5 bps one-way cost "
            f"({R['cost_sweep'][5][5.0][0]:+.1f} bps); 20-day hold loses "
            f"{R['cost_sweep'][20][0.0][0]:+.1f} bps gross |\n"
            f"| **Most reliable?** | `BUSTED` | Bat-vs-placebo Welch t = {R['plac_welch'][5]:+.2f} "
            f"(5-day); Bat wins on {R['n_beats_plac']}/6 tickers |\n\n"
            "> 💡 In plain words: neither half of Carney's claim survives — the pattern doesn't "
            "predict a reversal, and the specific 0.886 ratio isn't detectably better than an "
            "arbitrary nearby target."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Given four confirmed swing pivots $X, A, B, C$ (alternating high/low), let "
            "$XA = A - X$, $AB = B - A$, $BC = C - B$, retracements "
            "$\\rho_{AB} = |AB/XA|$, $\\rho_{BC} = |BC/AB|$. The Bat requires the *structural* "
            "bands $\\rho_{AB} \\in [0.382, 0.50]$ and $\\rho_{BC} \\in [0.382, 0.886]$ (shared "
            "across the XABCD zoo, not themselves under test), and projects\n\n"
            "$$D = A - \\delta \\cdot XA, \\qquad \\delta = 0.886$$\n\n"
            "— **the Bat's defining, single-number signature** — predicting a reversal (a "
            "positive return in direction $\\mathrm{sign}(XA)$) once price *touches* D.\n\n"
            "- **H₁ (reversal).** $E[\\text{fade return} \\mid D_{\\text{touch}}] \\gg 0$, robust "
            "to HAC inference, and **beats a drift-matched random-day base rate** — the specific "
            "confound named in the brief: any directional rule on a rising tape inherits some of "
            "the market's own drift.\n"
            "- **H₂ (0.886 specificity, i.e. \"most reliable\").** H₁, if true, must *beat* the "
            "identical pipeline run with $\\delta$ drawn off the 0.886 band — otherwise Carney's "
            "comparative \"most reliable\" claim has no footing: any nearby retracement/extension "
            "zone would do the same job.\n\n"
            "We find **H₁ rejected** (point estimate ≈ 0, never *t* ≥ 2 against either 0 or the "
            "base rate, 0/7 Bonferroni-corrected tests survive) and **H₂ rejected** (the placebo "
            "wins on 4/6 tickers)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Bat events on a single tape can **cluster in time** (overlapping legs share pivots), "
            "so the within-arm test is a **Newey-West (HAC)** *t* of the mean return against 0, "
            "not a naive i.i.d. *t*. Because a directional fade on an upward-drifting tape "
            "inherits some of that drift for free, the **decisive** Signal-axis test is a "
            "**Welch** *t* against a **random-day base rate** matched to the same empirical "
            "bullish/bearish mix as the real Bat touches, pooled over 20 seeds. We look at this "
            "comparison **seven ways** — the pooled headline plus each of six tickers — and apply "
            "a **Bonferroni** correction across all seven before allowing any of them to "
            "\"survive\". A second, independent **placebo** arm (identical structural pivots, "
            "randomized D-target) answers the *comparative* half of the claim: is 0.886 itself "
            "special, or would any nearby zone do?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Pivots.** A percentage-threshold zigzag (pct=2%) on daily closes; a pivot enters "
            "the record only at its **confirmation bar** — the session where price has already "
            "reversed past the threshold — never at the (earlier) extreme itself.\n"
            "- **Bat candidates.** Every consecutive confirmed quadruple $(X,A,B,C)$ with "
            "$\\rho_{AB} \\in [0.382, 0.50]$ and $\\rho_{BC} \\in [0.382, 0.886]$ (structural, "
            "identical in both arms); projects $D = A - 0.886 \\cdot XA$.\n"
            "- **Placebo candidates.** Identical structural pivots; the D-target is drawn "
            "per-candidate from a deterministic seeded uniform on $[0.55, 1.20]$, kept $\\geq "
            "0.05$ clear of 0.886.\n"
            "- **D-touch scan.** From C's confirmation bar forward, first bar in the next 120 "
            "sessions whose high-low range brackets $D$ (or closes within 0.75% of it) — "
            "real-time knowledge only.\n"
            "- **Execution.** Enter the fade at the touch bar's own close (intrabar touch, "
            "same-session close execution — one documented convention, identical in both arms); "
            "exit at close $+h$ sessions, $h \\in \\{1, 3, 5, 10, 20\\}$; net figures subtract "
            "$2\\times$ one-way cost $\\times$ NAV per round trip.\n"
            "- **Base-rate control.** $n$ random entry days per ticker (same $n$ as the real Bat "
            "touches), directional mix matched to the observed bullish/bearish split, pooled over "
            "20 seeds.\n"
            "- **Bonferroni.** 7 tests (pooled + 6 per-ticker, 5-day); critical $|t|$ raised from "
            "2.0 to 2.69 (family-wise $\\alpha=0.05$, normal approximation).\n"
            "- **Faithful-engine control.** Synthetic mean-reverting price index (tunable knob), "
            "pooled across a synthetic 6-name basket per seed; the null must not (materially) "
            "fire across 20 seeds, and a planted effect must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — Bat fade-at-D, all three horizons\n\n"
            f"Pooled confirmed pivots: **{R['n_pivots']:,}** → Bat XABCD candidates: "
            f"**{R['n_cand_bat']}** → D-touches: **{R['n_bat']}**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat_l = pooled(placebo=False)\n"
            "    rows = {h: st.summarize(bat_l, f'ret_gross_{h}') for h in (1, 5, 10)}\n"
            "    for h, s in rows.items():\n"
            "        print(f\"{h:2d}-day: n={s['n']} hit={s['hit_rate']*100:.1f}% \"\n"
            "              f\"mean={s['mean_bps']:+.2f}bps HAC t={s['t']:+.2f}\")\n"
            "    hs = list(rows); ms = [rows[h]['mean_bps'] for h in hs]\n"
            "else:\n"
            "    hs = [1, 5, 10]; ms = [R['bat'][h]['mean'] for h in hs]\n"
            "    for h in hs:\n"
            "        print(f\"{h:2d}-day: n={R['n_bat']} hit={R['bat'][h]['hit']:.1f}% \"\n"
            "              f\"mean={R['bat'][h]['mean']:+.2f}bps HAC t={R['bat'][h]['t']:+.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "cols = [RED if h == 5 else GREY for h in hs]\n"
            "ax.bar([f'{h}d' for h in hs], ms, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean return (bps)'); ax.set_title('Fade-at-D: flat at every horizon')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: at the headline 5-day horizon the Bat fade earns "
            f"**{R['bat'][5]['mean']:+.2f} bps** (HAC t = {R['bat'][5]['t']:.2f}) on a hit rate of "
            f"**{R['bat'][5]['hit']:.1f}%** (Wilson 95% [{R['wilson_bat'][0]:.1f}%, "
            f"{R['wilson_bat'][1]:.1f}%]) — a confidence interval straddling, and mostly *below*, "
            "a coin flip. The 1-day and 10-day versions are outright negative."
        ),
        md(
            "### 4b · The decisive test — Bat vs a drift-matched random-day base rate, "
            "Bonferroni-corrected\n\n"
            "The base rate draws random entry days per ticker with the **same directional mix** "
            "as the real Bat touches, pooled over 20 seeds — the exact confound named in the "
            "brief. We test the pooled headline plus all six per-ticker splits (7 tests) and "
            "correct for the multiple looks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat_l = pooled(placebo=False)\n"
            "    by_t = {t: st.detect_and_scan(BARS[t], pct=0.02, cost_bps=0.0)[2] for t in BASKET}\n"
            "    base_l = base_rate_pool(by_t)\n"
            "    tests = []\n"
            "    for h in (1, 5, 10):\n"
            "        wt = st.welch_t(bat_l[f'ret_gross_{h}'], base_l[f'ret_gross_{h}'])\n"
            "        print(f\"{h:2d}-day: Welch t (Bat vs base) = {wt:+.2f}\")\n"
            "        if h == 5: tests.append(('pooled', wt))\n"
            "    for t in BASKET:\n"
            "        l = by_t[t]\n"
            "        if len(l) == 0:\n"
            "            continue\n"
            "        mix = float((l['reversal_dir'] > 0).mean())\n"
            "        frames = [st.base_rate_ledger(BARS[t], len(l), mix, cost_bps=0.0,\n"
            "                                      seed=st._seed_from(700, t, s)) for s in range(20)]\n"
            "        br = pd.concat(frames, ignore_index=True)\n"
            "        wt = st.welch_t(l['ret_gross_5'], br['ret_gross_5'])\n"
            "        tests.append((t, wt))\n"
            "    ts = [t for t, _ in tests]; wts = [w for _, w in tests]\n"
            "else:\n"
            "    ts = ['pooled'] + list(R['per_instrument'])\n"
            "    wts = [R['base_welch'][5]] + [R['per_instrument'][t]['base_t'] for t in R['per_instrument']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if abs(w) >= R['bonf_thr'] else GREY for w in wts]\n"
            "ax.bar(ts, wts, color=cols, width=.55)\n"
            "ax.axhline(2.0, ls='--', c=AMBER, lw=1); ax.axhline(-2.0, ls='--', c=AMBER, lw=1)\n"
            f"ax.axhline({R['bonf_thr']}, ls='--', c=RED, lw=1.4)\n"
            f"ax.axhline(-{R['bonf_thr']}, ls='--', c=RED, lw=1.4)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (Bat vs matched-mix random day)')\n"
            "ax.set_title('0/7 tests clear the uncorrected bar (amber) or the Bonferroni bar (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(ts, [round(w,2) for w in wts])))"
        ),
        md(
            f"> 💡 In plain words: the pooled headline Welch *t* is only "
            f"**{R['base_welch'][5]:+.2f}**. Per-ticker, the closest call is **TSLA** at "
            f"**{R['per_instrument']['TSLA']['base_t']:+.2f}** — still short of the naive *t* = 2 "
            f"bar, let alone the Bonferroni-corrected **{R['bonf_thr']:.2f}** that 7 simultaneous "
            f"looks demand. **{R['bonf_n_survive']}/{R['bonf_n_tests']}** tests survive."
        ),
        md(
            "### 4c · Per-instrument breakdown (gross)\n\n"
            "Does the effect concentrate anywhere? TSLA is the one cell that individually clears "
            "the *uncorrected* *t* ≥ 2 bar on the raw HAC test — worth looking at directly, and "
            "worth remembering it doesn't survive 4b's correction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for t in BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=0.02, cost_bps=0.0)\n"
            "        s = st.summarize(l, 'ret_gross_5')\n"
            "        print(f\"{t:5s} n={s['n']:3d} mean={s['mean_bps']:+7.2f}bps HAC t={s['t']}\")\n"
            "else:\n"
            "    for t, d in R['per_instrument'].items():\n"
            "        print(f\"{t:5s} n={d['bat_n']:3d} mean={d['bat_mean']:+7.2f}bps HAC t={d['bat_t']}\")"
        ),
        md(
            f"> 💡 In plain words: **TSLA** shows a real-looking gross number "
            f"(+{R['per_instrument']['TSLA']['bat_mean']:.0f} bps, HAC "
            f"t = {R['per_instrument']['TSLA']['bat_t']:.2f}) — but its own drift-matched base "
            f"rate comparison is only *t* = {R['per_instrument']['TSLA']['base_t']:+.2f}, and it's "
            "one look out of seven that were never corrected for. This is exactly the "
            "data-snooping trap the Bonferroni step in 4b exists to catch."
        ),
        md(
            "### 4d · The fade-at-D timer, net of costs\n\n"
            "One round trip = 2 × one-way cost × NAV per event, across five hold lengths."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat_l5 = pooled(placebo=False, horizons=(1,3,5,10,20))\n"
            "    hs = [1,3,5,10,20]\n"
            "    means0 = [st.summarize(bat_l5, f'ret_gross_{h}')['mean_bps'] for h in hs]\n"
            "    means5 = [means0[i] - 2*5.0/1e4*1e4 for i in range(len(hs))]\n"
            "    means10 = [means0[i] - 2*10.0/1e4*1e4 for i in range(len(hs))]\n"
            "else:\n"
            "    hs = [1,3,5,10,20]\n"
            "    means0 = [R['cost_sweep'][h][0.0][0] for h in hs]\n"
            "    means5 = [R['cost_sweep'][h][5.0][0] for h in hs]\n"
            "    means10 = [R['cost_sweep'][h][10.0][0] for h in hs]\n"
            "x = np.arange(len(hs)); w = .27\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "ax.bar(x-w, means0, width=w, color=GREY, label='gross')\n"
            "ax.bar(x, means5, width=w, color=AMBER, label='net @ 5bps')\n"
            "ax.bar(x+w, means10, width=w, color=RED, label='net @ 10bps')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean return per event (bps)')\n"
            "ax.set_title('No hold length survives costs; longer holds bleed more')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(dict(zip(hs, zip(means0, means5, means10))))"
        ),
        md(
            f"> 💡 In plain words: the best gross case is the 5-day hold "
            f"(**{R['cost_sweep'][5][0.0][0]:+.2f} bps**, t = {R['cost_sweep'][5][0.0][1]:+.2f}) — "
            f"and it flips negative (**{R['cost_sweep'][5][5.0][0]:+.2f} bps**) at a single 5 bps "
            "one-way cost. The 20-day hold is a straightforward loser "
            f"(**{R['cost_sweep'][20][0.0][0]:+.2f} bps gross**, t = "
            f"{R['cost_sweep'][20][0.0][1]:+.2f}). There is no cost regime or hold length in this "
            "sweep where fading the Bat is attractive."
        ),
        md(
            "### 4e · The third axis — is 0.886 specifically better than a nearby target?\n\n"
            "Carney's \"most reliable\" claim is comparative: not just \"does the Bat's ratio "
            "work\", but \"does 0.886 beat what any other nearby retracement/extension zone would "
            "have captured\". Identical structural pivots; only the D-target differs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bat_l = pooled(placebo=False); plac_l = pooled(placebo=True)\n"
            "    rows = {}\n"
            "    for h in (1, 5, 10):\n"
            "        sf = st.summarize(bat_l, f'ret_gross_{h}'); sp = st.summarize(plac_l, f'ret_gross_{h}')\n"
            "        wt = st.welch_t(bat_l[f'ret_gross_{h}'], plac_l[f'ret_gross_{h}'])\n"
            "        rows[h] = (sf, sp, wt)\n"
            "        print(f\"{h:2d}-day: BAT mean={sf['mean_bps']:+.2f}bps  |  PLAC mean={sp['mean_bps']:+.2f}bps  |  Welch t={wt:+.2f}\")\n"
            "    hs = list(rows); bm = [rows[h][0]['mean_bps'] for h in hs]; pm = [rows[h][1]['mean_bps'] for h in hs]\n"
            "else:\n"
            "    hs = [1, 5, 10]; bm = [R['bat'][h]['mean'] for h in hs]; pm = [R['plac_mean_h'][h] for h in hs]\n"
            "    for h in hs:\n"
            "        print(f\"{h:2d}-day: BAT mean={R['bat'][h]['mean']:+.2f}bps  |  PLAC mean={R['plac_mean_h'][h]:+.2f}bps  |  Welch t={R['plac_welch'][h]:+.2f}\")\n"
            "x = np.arange(len(hs)); w = .35\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(x-w/2, bm, width=w, color=RED, label='Bat (0.886)')\n"
            "ax.bar(x+w/2, pm, width=w, color=GREY, label='Placebo (random nearby target)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean return (bps)'); ax.set_title('0.886 shows no edge over the placebo zone')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: at the headline 5-day horizon, Bat's "
            f"**{R['bat'][5]['mean']:+.2f} bps** vs the placebo's **{R['plac_mean_h'][5]:+.2f} "
            f"bps** gives Welch **t = {R['plac_welch'][5]:+.2f}** — negligible. Per-ticker, Bat "
            f"beats the placebo on only **{R['n_beats_plac']}/6** names. Carney's comparative "
            "\"most reliable\" claim has no footing here: an arbitrary nearby retracement/"
            "extension target does the same (non-)job as 0.886."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic synthetic price index with a tunable mean-reversion knob toward a slow "
            "EMA, pooled across a **synthetic 6-name basket per seed** (mirrors the real pooled "
            "sample size). The null (`mean_rev=0`) is checked over **20 seeds** — never a single "
            "stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    frames = []\n"
            "    for i in range(6):\n"
            "        sb = data.synthetic_world(mean_rev=0.0, seed=(700+s_)*1000+i, n_days=6300)\n"
            "        _,_,l = st.detect_and_scan(sb, pct=0.02, cost_bps=0.0)\n"
            "        if len(l): frames.append(l)\n"
            "    pooled_syn = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    null_ts.append(st.summarize(pooled_syn, 'ret_gross_5')['t'])\n"
            "null_ts = np.asarray(null_ts, dtype=float)\n"
            "frames = []\n"
            f"for i in range(6):\n"
            f"    sb = data.synthetic_world(mean_rev={R['syn_planted_knob']}, seed=700*1000+i, n_days=6300)\n"
            "    _,_,l = st.detect_and_scan(sb, pct=0.02, cost_bps=0.0)\n"
            "    if len(l): frames.append(l)\n"
            "planted = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "planted_t = st.summarize(planted, 'ret_gross_5')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mean_rev=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            f"           label='planted mean_rev={R['syn_planted_knob']}')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (fade-at-D vs 0)')\n"
            "ax.set_title('Control: the null rarely fires; a planted reversion lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires (|*t*| ≥ 2) in "
            f"**{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds — a touch above the nominal ~5% "
            f"false-positive rate, unsurprising given the small per-seed sample (~85 pooled "
            f"touches) — with mean t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}). A "
            f"planted mean-reversion tendency ({R['syn_planted_knob']}) lights up sharply "
            f"(n={R['syn_planted_n']}, t = {R['syn_planted_t']:.2f}). The pipeline is unbiased and "
            "has power — the flat real-tape result is the genuine article, not a broken detector. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 5-day fade-at-D: **{R['bat'][5]['mean']:+.2f} bps/event**, HAC "
            f"t = **{R['bat'][5]['t']:.2f}** (n={R['n_bat']}), hit rate "
            f"{R['bat'][5]['hit']:.1f}% (below a coin flip). Against a drift-matched random-day "
            f"base rate, pooled Welch t = {R['base_welch'][5]:+.2f}; "
            f"**{R['bonf_n_survive']}/{R['bonf_n_tests']}** Bonferroni-corrected tests (critical "
            f"|t| = {R['bonf_thr']:.2f}) survive. TSLA's individually notable gross cell "
            f"(t = {R['per_instrument']['TSLA']['bat_t']:.2f}) does not survive the correction.\n"
            f"- **Tradability `MIRAGE`** — no hold length (1/3/5/10/20-day) clears *t* ≥ 2 on the "
            f"positive side; the best gross case ({R['cost_sweep'][5][0.0][0]:+.2f} bps, 5-day) "
            f"turns negative at a single 5 bps one-way cost "
            f"({R['cost_sweep'][5][5.0][0]:+.2f} bps); the 20-day hold loses "
            f"{R['cost_sweep'][20][0.0][0]:+.2f} bps gross.\n"
            f"- **\"Beats a placebo, i.e. is 0.886 the most reliable ratio?\" `BUSTED`** — the Bat "
            f"arm beats its own placebo control (identical pivots, random nearby D-target) on "
            f"only **{R['n_beats_plac']}/6** tickers, and the pooled 5-day Welch t is a negligible "
            f"{R['plac_welch'][5]:+.2f}. There is no detectable specificity in the 0.886 ratio."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The comparative claim deserves a head-to-head.** Carney's \"most reliable\" "
            "pitch is relative to the rest of the zoo — a natural sequel pools this study's "
            "protocol against [698-abcd-harmonic](../../698-abcd-harmonic/) "
            "(no X point), [699-butterfly-harmonic](../../699-butterfly-harmonic/) (D overshoots "
            "X) and [468-gartley-harmonic](../../468-gartley-harmonic/) (a shallower 0.786 D) on "
            "an identical touch-scan protocol to rank them directly, rather than comparing "
            "verdicts study-by-study.\n"
            "- **A natural sequel** would test whether a tighter D-tolerance (a narrower band "
            "around 0.886) selects for a genuinely different, better-performing subset of setups "
            "— or whether tightening the band just shrinks the sample without changing the sign.\n"
            "- **Dedup map:** [468-gartley-harmonic](../../468-gartley-harmonic/) (B retraces "
            "0.618, D retraces 0.786 — both shallower than the Bat), "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) (the bare two-leg AB=CD skeleton, no X "
            "point), [699-butterfly-harmonic](../../699-butterfly-harmonic/) (D *extends* past X "
            "instead of retracing into it), [701-crab-harmonic](../../701-crab-harmonic/) (an "
            "even sharper 1.618 extension), [702-shark-harmonic](../../702-shark-harmonic/) (a "
            "non-standard 5-0 extension pattern), [77-golden-mean](../../77-golden-mean/) (plain "
            "Fibonacci retracement *levels*, same six tapes, same verdict shape).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
