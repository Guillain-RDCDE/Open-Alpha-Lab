"""Generate the two narrative notebooks for Study 704 (Three Drives).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (yfinance basket, as-of
# 2026-06-30; 4% ZigZag; 0.382-0.886 correction band, 1.13-2.618 extension band).
R = dict(
    asof="2026-06-30", pct=4, headline_h=20, cost_bps=5,
    corr_lo=0.382, corr_hi=0.886, ext_lo=1.13, ext_hi=2.618,
    n_cand=110,
    fade_n=110, fade_gross=45.0, fade_net=35.0, fade_win=50.9,
    fade_wilson=(41.7, 60.1), fade_t=0.62, fade_hac=0.62,
    coin_obs=45.0, coin_mean=-0.2, coin_sd=53.4, coin_p=0.207,
    # horizon -> (n, gross_bps, net_bps, win%, t)
    event={5: (110, -16.5, -26.5, 42.7, -0.42), 10: (110, 0.2, -9.8, 49.1, 0.00),
           20: (110, 45.0, 35.0, 50.9, 0.62), 40: (110, -4.3, -14.3, 45.5, -0.04)},
    grid_obs=45.0, grid_p=0.262, grid_draws=496,
    sym_n=110, sym_median_cv=0.396, sym_sym_bps=145.5, sym_asym_bps=-55.4, sym_t=1.39,
    # pct*100 -> (n, mean_bps, win%, t)
    sweep={3: (139, 12.0, 48.2, 0.23), 4: (110, 45.0, 50.9, 0.62),
           5: (97, 128.2, 57.7, 1.66), 8: (34, 62.2, 55.9, 0.52)},
    spy_n=16, spy_mean=25.4, spy_win=56.2, spy_wilson=(33.2, 76.9), spy_t=0.13,
    syn_null_mean=-0.84, syn_null_sd=0.80, syn_null_fire=0, syn_planted_t=6.87,
    syn_planted_n=27, syn_planted_mean=1401.0,
    fps={"SPY": "079eeeacb330", "QQQ": "ba4a1e34bb16", "DIA": "8484eb68cb8e",
         "IWM": "cfbd1e0dca21", "^GSPC": "cec06bce14e7", "^IXIC": "54d65c6aa4cb",
         "^DJI": "803513feb661", "GLD": "44f6ff1685e4"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Fibonacci_load--bearing%3F: Busted](https://img.shields.io/badge/Fibonacci_load--bearing%3F-Busted-8b949e?style=flat-square)\n\n"
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

from three_drives import data, strategy as st

TICKERS = [tk for tk in data.TICKERS if data.have_real(tk)]
HAVE_REAL = len(TICKERS) > 0
BASKET = data.load_basket(TICKERS) if HAVE_REAL else {}
print("real cache present:", HAVE_REAL, "| tickers:", TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Three exhausted pushes, one Fibonacci story — does it actually reverse? 🎯\n"
            "### The \"Three Drives\" chart pattern — a folklore favorite from the same family "
            "as Elliott Wave and Gartley, put through the same honest test\n\n"
            + BADGES +
            "Chartists love a pattern with a story: three pushes to a new high (or low), each "
            "one a **Fibonacci-perfect extension** of the pullback before it, are supposed to "
            "signal a trend running out of gas. Draw the lines, count the ratios, and — the "
            "story goes — point 5 is where smart money fades the crowd.\n\n"
            "We built a computer that draws those lines exactly the same way every time, on "
            "SPY and seven other broad tapes, and asked the one question that matters: **does "
            "price actually turn there, more than a coin flip would predict?**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** A 4% ZigZag marks the swing pivots; a candidate is six "
            "consecutive alternating pivots whose four leg ratios land on the widest Fibonacci "
            "band the pattern's own teachers cite. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price reverse after three Fibonacci drives? | **Not more than chance.** "
            f"Across **{R['fade_n']}** detections pooled over 8 tapes, fading the pattern at "
            f"point 5 makes **{R['fade_gross']:+.0f} bps** on average over 20 days — a coin-flip "
            f"win rate, nowhere near statistically real. |\n"
            "| Is it just bad luck with one setting? | **No — flat everywhere we look.** Every "
            "hold period from 5 to 40 days, every ZigZag threshold from 3% to 8%: no result "
            "clears the desk's bar. |\n"
            "| Do the *specific* Fibonacci numbers (0.618, 1.27...) matter? | **No.** Swap them "
            "for random ratios and the random grids do just as well. |\n"
            "| Would our test have caught a REAL pattern? | **Yes.** On a synthetic tape with a "
            "planted reversal, the same code lights up unmistakably. |\n\n"
            "> Three drives, zero edge — a genuine null, not a broken test."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Three consecutive drives to a new high (or low), each one extending the "
            "correction before it by a Fibonacci ratio, mark a trend that has run out of "
            "momentum. When the third drive completes, fade it.\"*\n\n"
            "It's part of the same Fibonacci-harmonic family as Gartley patterns and Elliott "
            "Wave counts — taught in the same courses, drawn with the same tools. The pitch is "
            "seductive precisely because the geometry really is checkable: either the ratios "
            "line up or they don't."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a free, mechanical reversal signal on any liquid chart — no news, "
            "no fundamentals, just ruler-and-Fibonacci-calculator geometry. Retail chartists "
            "trade it by hand every day. So we ask: does a computer, applying the *exact same* "
            "rule with no hindsight, find anything a coin flip wouldn't?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The detector.** A 4% ZigZag marks swing pivots; six in a row (point 0 through "
            "point 5) qualify if the two corrections retrace 38.2%-88.6% of the drive before "
            "them, the two drives extend the correction before them by 1.13x-2.618x, and each "
            "drive genuinely goes beyond the last (the \"extending drives\" rule every source "
            "agrees on).\n"
            "- **The fade.** Enter *against* the three drives at the next close after point 5 is "
            "confirmed (no look-ahead), hold 5/10/20/40 days.\n"
            "- **The luck check.** A coin flip — random timing, random direction, same count — "
            "on the identical tape.\n"
            "- **The Fibonacci check.** Swap the specific ratios for random ones; if the magic "
            "numbers matter, the real grid should win."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Fade the pattern, hold 20 days, pooled across the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = st.pooled_fade_test(BASKET, pct=0.04, horizon=20, cost_bps=5.0)\n"
            "    n = len(pooled['entries'])\n"
            "    sg = st.summarize(pooled['gross']); sn = st.summarize(pooled['net'])\n"
            "    g, ne = sg['mean_bps'], sn['mean_bps']\n"
            "else:\n"
            "    n, g, ne = R['fade_n'], R['fade_gross'], R['fade_net']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['fade the drives\\n(gross)', 'fade the drives\\n(net of costs)'], [g, ne],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "for i,v in enumerate([g, ne]): ax.annotate(f'{v:+.0f} bps',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 20-day fade return (bps)')\n"
            f"ax.set_title(f'{{n}} three-drives detections, pooled across 8 tapes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'n={n}  gross={g:+.1f} bps  net={ne:+.1f} bps')"
        ),
        md(
            f"**{R['fade_n']}** algorithmically-detected three-drives patterns across SPY and "
            "seven other broad tapes going back decades. Fading them — betting the third drive "
            f"is really exhausted — nets **{R['fade_gross']:+.0f} bps** gross over the next 20 "
            "trading days. That's a real number, but is it a *signal*, or is it just what any "
            "random bet on this tape would make?\n\n"
            "**Next, the honest comparison** — the same bet, but timed and directed by a coin."
        ),
        code(
            "if HAVE_REAL:\n"
            "    per_n = {tk: c['candidates'] for tk, c in pooled['counts'].items()}\n"
            "    pl = st.coin_placebo_pvalue(BASKET, per_n, 20, sg['mean_bps']/1e4, n_draws=300, seed=704)\n"
            "    obs, pm, psd, pp = pl['obs']*1e4, pl['placebo_mean']*1e4, pl['placebo_sd']*1e4, pl['p_value']\n"
            "else:\n"
            "    obs, pm, psd, pp = R['coin_obs'], R['coin_mean'], R['coin_sd'], R['coin_p']\n"
            "rng = np.random.default_rng(704)\n"
            "draws = rng.normal(pm, psd, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='coin-flip null (illustrative)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed fade mean {obs:+.1f} bps')\n"
            "ax.set_xlabel('mean 20-day return of a random-time, random-direction bet (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            f"ax.set_title(f'Not far from the luck cloud: p = {{pp:.3f}}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.1f} bps vs coin-flip mean {pm:+.1f} bps (sd {psd:.1f})  p={pp:.3f}')"
        ),
        md(
            f"The observed fade sits **inside** the coin-flip cloud (*p* = {R['coin_p']:.2f}) — "
            "a random bet, timed and directed by nothing but chance, does about as well. Knowing "
            "\"three Fibonacci drives just finished\" is not information the market is paying you "
            "for.\n\n"
            "**And the specific Fibonacci numbers?** Swap 0.618 / 1.27 for random ratios and see "
            "if the real grid actually earns its keep."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gp = st.ratio_grid_placebo(BASKET, pct=0.04, horizon=20, n_draws=200, seed=704)\n"
            "    gobs, gp_p = gp['obs']*1e4, gp['p_value']\n"
            "else:\n"
            "    gobs, gp_p = R['grid_obs'], R['grid_p']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['real Fibonacci grid\\n(0.618 / 1.27-ish)'], [gobs], color=AMBER, width=.4)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 20-day fade return (bps)')\n"
            f"ax.set_title(f'Random ratio grids match or beat this {{gp_p*100:.0f}}% of the time')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real grid {gobs:+.1f} bps   placebo p={gp_p:.3f}')"
        ),
        md(
            f"**{R['grid_p']*100:.0f}%** of random ratio grids do as well as the real Fibonacci "
            "grid. The 0.618s and 1.27s that make this pattern *look* like serious math carry no "
            "extra information — any similarly-shaped zig-zag would do just as well."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** {R['fade_n']} detections, fade return "
            f"**{R['fade_gross']:+.0f} bps** at *t* = {R['fade_t']:+.2f} — inside the coin-flip "
            "cloud, flat across every hold period and every ZigZag setting.\n"
            "- **Tradability — Mirage.** There's no edge to charge costs against, and the sample "
            "is thin (a handful of setups a year, pooled across 8 tapes).\n"
            "- **\"Do the Fibonacci ratios matter?\" — Busted.** Random ratio grids do just as "
            "well as the real one."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The whole harmonic-pattern family gets the same treatment on this desk.** See "
            "sibling [468-gartley-harmonic](../../468-gartley-harmonic/) (Fibonacci ratios "
            "busted there too, though a thin 60-day dip-buy effect survives) and "
            "[697-wolfe-waves](../../697-wolfe-waves/) (a converging-wedge cousin, also None).\n"
            "- **What would change our mind:** a version of the pattern with an out-of-sample, "
            "pre-registered ratio grid that a random grid genuinely can't match — we haven't "
            "found one yet, on any of these three studies.\n\n"
            "*Think you can draw a better three-drives rule? Show a net, certifiable edge against "
            "the same coin-flip and ratio-grid placebos — then we'll talk.*"
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
            "# Three Drives — a quantitative teardown 🔬\n"
            "### The fade-vs-coin-flip test · a Fibonacci ratio-grid placebo · the time-symmetry "
            "myth-check · a ZigZag threshold sweep · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Three Drives is the third member of this desk's Fibonacci-harmonic family (after "
            "[468-gartley-harmonic](../../468-gartley-harmonic/) and "
            "[697-wolfe-waves](../../697-wolfe-waves/)) — a pattern with **no price target**, "
            "just a plain reversal claim, tested against the honest base rate: a coin flip, not "
            "a beta-matched drift baseline (fade direction is generated by the pattern itself, "
            "±1, so there is no long-only drift to control for).\n\n"
            "> ⚠️ **Data note.** Daily OHLC (auto-adjusted, total-return), SPY + QQQ + DIA + IWM "
            "+ ^GSPC + ^IXIC + ^DJI + GLD, yfinance, cached; as-of **2026-06-30**. No "
            "cross-sectional survivorship (broad indices/ETFs, single-instrument pattern study). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | {R['fade_n']} detections, fade **{R['fade_gross']:+.0f} "
            f"bps** at *t* = {R['fade_t']:+.2f} (HAC *t* = {R['fade_hac']:+.2f}), coin-flip "
            f"placebo *p* = {R['coin_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | no edge to charge costs against; "
            f"~{R['fade_n']//8}/ticker over the full history |\n"
            f"| **Fibonacci ratios load-bearing?** | `BUSTED` | ratio-grid placebo "
            f"*p* = {R['grid_p']:.2f} |\n\n"
            "> 💡 In plain words: the geometry is real and checkable; the reversal it's supposed "
            "to predict isn't there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $(p_0, p_1, p_2, p_3, p_4, p_5)$ be six alternating swing pivots (a percentage "
            "ZigZag, threshold $\\theta$). Define drives $d_1 = |p_1-p_0|$, $d_2 = |p_3-p_2|$, "
            "$d_3 = |p_5-p_4|$ and corrections $c_1 = |p_1-p_2|$, $c_2 = |p_3-p_4|$. The claim:\n\n"
            "- **H₁ (geometry).** $c_1/d_1, c_2/d_2 \\in [0.382, 0.886]$ and "
            "$d_2/c_1, d_3/c_2 \\in [1.13, 2.618]$, with $d_2$ and $d_3$ each extending beyond "
            "the prior drive — the widest mechanical band the pattern's own teachers cite.\n"
            "- **H₂ (reversal).** Fading the pattern at point 5 (entered at the next close, one "
            "documented execution lag) beats a **random-time, random-direction** placebo — the "
            "honest base rate here, since the fade direction is generated by the pattern (±1), "
            "not a long-only drift bet.\n"
            "- **H₃ (Fibonacci specificity).** The *particular* ratio bands matter — a random "
            "ratio grid should do markedly worse.\n"
            "- **H₄ (symmetry).** The folklore's own word: patterns whose three drives are more "
            "evenly spaced in *time* should reverse more reliably.\n\n"
            "We find **H₁ satisfiable** (110 real-tape detections exist), **H₂ not supported** "
            "(coin-flip *p* well above 0.05), **H₃ busted** (random grids match), **H₄ not "
            "supported** either."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Each fade trade is single, non-overlapping (points 5 across the basket rarely "
            "cluster), so the planned primary is a **one-sample *t*** on the fade-return sample "
            "plus a **Newey-West (5-lag)** cross-check for any residual overlap. Because there is "
            "no natural drift baseline for a ±1-direction bet, the honest null is a **coin-flip "
            "placebo**: replay the exact same per-ticker candidate counts at random times with a "
            "random ±1 direction, 1,000 times, and ask how often the random bet's mean return "
            "beats the observed one. The **Fibonacci-grid placebo** separately asks whether the "
            "*specific* ratio bands (not just \"some zig-zag shape\") carry information."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** 4% ZigZag pivots (3/5/8% swept); six-in-a-row window; the Fibonacci "
            "grid above; \"extending drives\" ordering enforced.\n"
            "- **Fade.** Enter next close after point 5's confirmation (one lag), hold "
            f"5/10/20/40 days, {5}bps one-way costs x 2 legs.\n"
            "- **Base rate.** Random-time, random-direction coin-flip placebo, matched candidate "
            "counts per ticker.\n"
            "- **Specificity.** Random ratio-grid placebo, same machinery.\n"
            "- **Symmetry.** Coefficient-of-variation of the three drive-leg bar-counts; "
            "Welch-*t* the more- vs less-symmetric half's fade returns.\n"
            "- **Control.** Synthetic tape with *exact* Fibonacci geometry and a planted-reversal "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline fade and the coin-flip placebo\n\n"
            "Pooled across the basket, H = 20 days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = st.pooled_fade_test(BASKET, pct=0.04, horizon=20, cost_bps=5.0)\n"
            "    for tk, c in pooled['counts'].items():\n"
            "        print(f\"  {tk:>7}: pivots={c['pivots']:>4}  candidates={c['candidates']:>3}\")\n"
            "    sg = st.summarize(pooled['gross']); sn = st.summarize(pooled['net'])\n"
            "    print(f\"gross n={sg['n']} mean={sg['mean_bps']:+.1f}bps win={sg['win']*100:.1f}% \"\n"
            "          f\"t={sg['t']:+.2f} hac_t={sg['hac_t']:+.2f}\")\n"
            "    per_n = {tk: c['candidates'] for tk, c in pooled['counts'].items()}\n"
            "    pl = st.coin_placebo_pvalue(BASKET, per_n, 20, sg['mean_bps']/1e4, n_draws=300, seed=704)\n"
            "    print(f\"coin-flip placebo: observed {pl['obs']*1e4:+.1f} vs \"\n"
            "          f\"{pl['placebo_mean']*1e4:+.1f} (sd {pl['placebo_sd']*1e4:.1f})  p={pl['p_value']:.3f}\")\n"
            "else:\n"
            "    print('(no cache -- see docs/results.md for the pinned real-tape numbers)')\n"
            "    print(f\"gross mean={R['fade_gross']:+.1f}bps t={R['fade_t']:+.2f} \"\n"
            "          f\"coin p={R['coin_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['fade_n']}** patterns, mean fade **{R['fade_gross']:+.0f} "
            f"bps** at *t* = **{R['fade_t']:+.2f}** — nowhere near the *t* = 2 bar, and a "
            f"coin-flip bet with the same timing freedom does about as well "
            f"(*p* = {R['coin_p']:.2f}). If the pattern carried real reversal information, the "
            "real fade should beat blind timing+direction far more often than this."
        ),
        md(
            "### 4b · The fade timer — every horizon\n\n"
            "Does a longer or shorter hold change the answer?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in st.HORIZONS:\n"
            "        p_h = st.pooled_fade_test(BASKET, pct=0.04, horizon=h, cost_bps=5.0)\n"
            "        sgh = st.summarize(p_h['gross'])\n"
            "        rows.append((h, sgh['n'], sgh['mean_bps'], sgh['t'], sgh['hac_t']))\n"
            "    hs = [r[0] for r in rows]; ms = [r[2] for r in rows]; ts = [r[3] for r in rows]\n"
            "else:\n"
            "    hs = sorted(R['event']); ms = [R['event'][h][1] for h in hs]\n"
            "    ts = [R['event'][h][4] for h in hs]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True)\n"
            "a1.bar([str(h) for h in hs], ms, color=GREY, width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean fade return (bps)')\n"
            "a1.set_title('Flat at every horizon')\n"
            "a2.bar([str(h) for h in hs], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.55)\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('one-sample t'); a2.set_xlabel('holding period (trading days)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(list(zip(hs, [round(m,1) for m in ms], [round(t,2) for t in ts])))"
        ),
        md(
            "> 💡 In plain words: nothing crosses **±2** at any horizon. If the pattern's edge "
            "were merely showing up at the wrong hold length, one of these four bars would say "
            "so; none does."
        ),
        md(
            "### 4c · Is the *specific* Fibonacci grid doing anything?\n\n"
            "Randomize the correction/extension bands, keep everything else identical."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gp = st.ratio_grid_placebo(BASKET, pct=0.04, horizon=20, n_draws=200, seed=704)\n"
            "    gobs, gpp, gdraws = gp['obs']*1e4, gp['p_value'], gp['n_draws']\n"
            "else:\n"
            "    gobs, gpp, gdraws = R['grid_obs'], R['grid_p'], R['grid_draws']\n"
            "print(f'real Fibonacci grid: {gobs:+.1f} bps')\n"
            "print(f'random grids matching or beating it: {gpp*100:.1f}%  ({gdraws} valid draws)')"
        ),
        md(
            f"> 💡 In plain words: **{R['grid_p']*100:.0f}%** of random ratio grids do as well as "
            "the folklore's own numbers. The 0.618s and 1.27s are decoration on a generic "
            "zig-zag shape, not load-bearing forecasting content — echoing the same finding on "
            "sibling [468-gartley-harmonic](../../468-gartley-harmonic/)."
        ),
        md(
            "### 4d · The time-symmetry myth-check\n\n"
            "The folklore's own word is \"symmetric\". Split detections at the median "
            "coefficient-of-variation of the three drive-leg bar-counts (lower = more evenly "
            "spaced in time) and compare fade returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sym = st.symmetry_split_test(pooled['entries'], pooled['gross'])\n"
            "else:\n"
            "    sym = {'n': R['sym_n'], 'median_cv': R['sym_median_cv'],\n"
            "           'mean_sym_bps': R['sym_sym_bps'], 'mean_asym_bps': R['sym_asym_bps'],\n"
            "           'welch_t': R['sym_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['more time-symmetric\\nhalf', 'less time-symmetric\\nhalf'],\n"
            "       [sym['mean_sym_bps'], sym['mean_asym_bps']], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 20-day fade return (bps)')\n"
            "ax.set_title(f\"Welch t = {sym['welch_t']:+.2f} -- symmetry carries no signal\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sym)"
        ),
        md(
            f"> 💡 In plain words: the more time-symmetric half of detections returns "
            f"**{R['sym_sym_bps']:+.0f} bps**, the less symmetric half **{R['sym_asym_bps']:+.0f} "
            f"bps** (Welch *t* = {R['sym_t']:+.2f}). \"Symmetric\" is doing no predictive work "
            "either — it's a description of the geometry we already required, not a forecast."
        ),
        md(
            "### 4e · Robustness — the ZigZag threshold sweep\n\n"
            "Does a looser or tighter swing filter change the story?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    for pct in (0.03, 0.04, 0.05, 0.08):\n"
            "        p2 = st.pooled_fade_test(BASKET, pct=pct, horizon=20, cost_bps=0.0)\n"
            "        s2 = st.summarize(p2['gross'])\n"
            "        print(f\"pct={pct:.2f}: n={s2['n']:>4}  mean={s2['mean_bps']:>7.1f}bps  \"\n"
            "              f\"win={s2['win']*100:>5.1f}%  t={s2['t']:>6.2f}\")\n"
            "else:\n"
            "    for pct, (n, m, w, t) in R['sweep'].items():\n"
            "        print(f\"pct={pct/100:.2f}: n={n:>4}  mean={m:>7.1f}bps  win={w:>5.1f}%  t={t:>6.2f}\")"
        ),
        md(
            "> 💡 In plain words: no ZigZag threshold, loose or tight, turns up a certifiable "
            "edge. This isn't a knob that happens to have missed it."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic panel: exact Fibonacci geometry by construction, TUNABLE planted reversal "
            "after point 5. The null (edge = 0) is checked over **20 seeds** — never a single "
            "stream. Note a small, disclosed **selection effect** shared with sibling "
            "697-wolfe-waves: entry requires a ZigZag-confirmed point 5, i.e. price has already "
            "moved against the drives by the confirmation threshold *before* entry — the null "
            "centers modestly below zero rather than exactly at it."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    bars, _ = data.synthetic_panel(edge=0.0, seed=704 + s_, n_days=9000)\n"
            "    piv = st.zigzag(bars['close'].to_numpy(float), pct=0.04)\n"
            "    ent = st.three_drives_candidates(piv)\n"
            "    r = st.forward_returns(bars['close'], ent, 20)\n"
            "    null_ts.append(st.summarize(r)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "bars, truth = data.synthetic_panel(edge=0.30, seed=704, n_days=9000)\n"
            "piv = st.zigzag(bars['close'].to_numpy(float), pct=0.04)\n"
            "ent = st.three_drives_candidates(piv)\n"
            "r = st.forward_returns(bars['close'], ent, 20)\n"
            "planted_t = st.summarize(r)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted edge = 0.30')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (fade return, H=20)')\n"
            "ax.set_title('Control: the null stays quiet; a planted reversal is unmistakable')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f} '\n"
            "      f'(n={len(ent)}, {truth[\"n_planted\"]} structures planted)')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and stays quiet; a "
            f"planted reversal reads t = {R['syn_planted_t']:.1f}. The machinery is live and "
            "unbiased — the flat real-tape result above is a genuine null, not a broken "
            "detector. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — {R['fade_n']} algorithmically-detected three-drives patterns, "
            f"pooled fade return **{R['fade_gross']:+.0f} bps** at one-sample *t* = "
            f"**{R['fade_t']:+.2f}** (HAC *t* = {R['fade_hac']:+.2f}), a coin-flip placebo *p* = "
            f"**{R['coin_p']:.2f}**, flat across every horizon and every ZigZag threshold. A "
            "synthetic control proves the harness *would* detect a real planted reversal, so "
            "this is a genuine null.\n"
            "- **Tradability `MIRAGE`** — no edge to charge costs against; thin sample "
            f"(~{R['fade_n']//8} candidates/ticker over the whole history).\n"
            f"- **\"Do the Fibonacci ratios matter?\" `BUSTED`** — random ratio grids match the "
            f"real one **{R['grid_p']*100:.0f}%** of the time. The pattern's own word "
            "\"symmetric\" fares no better: the more time-symmetric half of detections doesn't "
            f"out-forecast the less symmetric half (Welch *t* = {R['sym_t']:+.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The whole Fibonacci-harmonic family, side by side:** "
            "[468-gartley-harmonic](../../468-gartley-harmonic/) (a thin, ratio-agnostic 60-day "
            "dip-buy effect survives; the Fibonacci ratios themselves are busted there too), "
            "[697-wolfe-waves](../../697-wolfe-waves/) (a converging wedge, also a clean None), "
            "and this study (a plain reversal claim, also None). Three different Fibonacci "
            "five-point structures, three honest teardowns, no certifiable edge in any of them.\n"
            "- **What we didn't try:** conditioning the fade on a trend/volatility regime filter, "
            "or requiring the drives to be genuinely time-symmetric *ex ante* (rather than "
            "measuring it post hoc) — worth a PR if someone wants to steelman the pattern "
            "further.\n\n"
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
