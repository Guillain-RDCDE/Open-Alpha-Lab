"""Generate the two narrative notebooks for Study 404 (Shooting Star).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
control runs anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-23,
# fingerprint d488237f4299). Basket = 26 liquid US large-caps + SPY, yfinance daily,
# un-adjusted OHLC, price-only forward returns, full history (longest back to 1962).
R = dict(
    fingerprint="d488237f4299", asof="2026-06-23", n_names=26,
    n_stars=10746, n_up_stars=5672,
    # SHOOTING STAR (prior uptrend), traded SHORT, gross edge vs short base rate
    star_h1_n=5672, star_h1_edge=-3.1, star_h1_t=-1.39, star_h1_p=0.889, star_h1_win=45.3,
    star_h3_n=5668, star_h3_edge=-3.9, star_h3_t=-0.97, star_h3_p=0.816, star_h3_win=46.1,
    star_h5_n=5668, star_h5_edge=-6.1, star_h5_t=-1.12, star_h5_p=0.870, star_h5_win=45.2,
    star_h10_n=5666, star_h10_edge=0.6, star_h10_t=0.07, star_h10_p=0.465, star_h10_win=45.3,
    # ANY star geometry, traded SHORT — the "busted" finding (significant WRONG sign)
    any_h1_n=10744, any_h1_edge=-3.9, any_h1_t=-2.15, any_h1_p=0.981, any_h1_win=45.2,
    any_h3_edge=-7.0, any_h3_t=-2.23,
    any_h5_edge=-8.7, any_h5_t=-2.05,
    # Myth-check: strong-uptrend filter (star, short, H=5)
    f00_edge=-6.1, f00_t=-1.12, f00_n=5668,
    f05_edge=3.8, f05_t=0.35, f05_n=1759,
    f10_edge=-6.2, f10_t=-0.23, f10_n=477,
    f15_edge=0.8, f15_t=0.02, f15_n=143,
    # Per-name star short edge H=5 — lone positive outlier among 26 names
    pg_t=2.38, n_names_positive=9, best_neg_gs=-93.1, best_neg_gs_t=-1.50,
    # Cost sweep (star short, H=5; borrow 2bps/day x 5 = 10bps)
    net_c0=-16.1, net_c5=-26.1, net_c10=-36.1,
    # Synthetic positive control (H=1, the planted horizon)
    syn_null_edge=0.1, syn_null_t=0.03, syn_null_p=0.49,
    syn_plant_edge=63.1, syn_plant_t=14.57, syn_plant_p=0.0,
)


# --------------------------------------------------------------------------- #
# Shared analysis preamble — imports, the basket, small pooled helpers.
# --------------------------------------------------------------------------- #
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

from shooting_star import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))
HAVE_REAL = data.have_real(cache_dir=CACHE)

def real_panel():
    return data.load_real(cache_dir=CACHE, fetch=False)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Shooting Star — does the long upper wick call the top?\n"
            "### The bearish candlestick reversal, tested honestly on US large-caps\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Calls the top%3F: Busted](https://img.shields.io/badge/Calls_the_top%3F-Busted-8b949e?style=flat-square)\n\n"
            "Open any candlestick guide and you'll meet the **shooting star**: a small body at "
            "the bottom of the day's range with a long wick sticking up. The story is vivid — "
            "buyers shoved price higher all session, then *lost the entire gain* by the close. "
            "Exhaustion at the top. A reversal is coming: **sell**. After a nice run-up, the "
            "shooting star is supposed to mark the high. Does it?\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the label-shuffle placebo, "
            "and the trend-filter myth-check? That's the companion, "
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
            f"| Does shorting the star after a run-up pay? | **No.** The short edge over a "
            f"buy-and-short baseline is **{R['star_h5_edge']:+.1f} bps** at 5 days — "
            f"HAC *t* = **{R['star_h5_t']:+.2f}**, nowhere near significant. |\n"
            f"| Does the star call the *top*? | **No — if anything, the opposite.** Pooling "
            f"every star, shorting it loses *significantly* (*t* = **{R['any_h1_t']:+.2f}**): the "
            "geometry leans toward continuation **up**, not reversal down. |\n"
            "| Does 'only fade a star at the top of a strong rally' fix it? | **No.** Adding a "
            "trend-strength filter just shrinks the sample; the edge stays at noise. |\n"
            "| Could you trade it? | **No.** Gross is already negative, and shorts pay spread "
            "*and* borrow — costs only deepen the hole. |\n\n"
            "> The picture is pretty; the prediction isn't there. On modern US large-caps the "
            "shooting star does **not** mark the top."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A shooting star is a small body at the bottom of the range with a long upper "
            "shadow, appearing after an uptrend. The long wick shows buyers pushed price up "
            "and then completely failed to hold it — the rally is exhausted. Sell: a reversal "
            "down is due.\"*\n\n"
            "It's the bearish mirror of the **hammer** (which we tear down in "
            "[Study 403](../../403-hammer-hanging-man/)). Same intuition, flipped: the hammer's "
            "long *lower* wick is a floor; the shooting star's long *upper* wick is a ceiling. "
            "We test whether that ceiling is real on 26 liquid US large-caps + SPY, decades of "
            "daily bars."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a one-day candle reliably called tops, it would be an extraordinarily cheap "
            "timing tool — no indicators, no parameters, just a shape you can spot by eye. "
            "Millions of retail traders are taught exactly this. If it's wrong — if "
            "'exhaustion' candles are more often just noise inside an ongoing trend — then "
            "people are shorting strength and paying spread *and* borrow to do it."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "We announce the test before running it:\n\n"
            "1. **Detect the geometry by exact OHLC rules** — upper wick at least 2x the body, "
            "body near the bottom of the range, small body relative to the day's range.\n"
            "2. **Split by prior trend** — the *shooting star* is the shape after an uptrend "
            "(the bearish claim). Then short it: enter at the *next* close, hold 1/3/5/10 days.\n"
            "3. **Compare to the base rate** — what would shorting *any* random day in the same "
            "name earn over the same window? The edge is the conditional return minus that base "
            "rate, name by name (so each name's own drift is netted out).\n"
            "4. **A placebo** — could the same *number* of random days look this good by luck?\n\n"
            "**Mirage trigger:** if the short edge doesn't clear HAC *t* = +2 — or worse, is "
            "negative — the star doesn't call the top."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: does shorting the star pay, across holding windows?** A positive bar means "
            "the short made money beyond the base rate; the dashed line is the *t* = +2 bar a "
            "real signal would have to clear."
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    panel = real_panel()\n"
            "    res = st.run_experiment(panel, side='star', n_draws=2000)\n"
            "    edges = [res[h]['edge_mean']*1e4 for h in horizons]\n"
            "    ts = [res[h]['t'] for h in horizons]\n"
            "else:\n"
            f"    edges = [{R['star_h1_edge']},{R['star_h3_edge']},{R['star_h5_edge']},{R['star_h10_edge']}]\n"
            f"    ts = [{R['star_h1_t']},{R['star_h3_t']},{R['star_h5_t']},{R['star_h10_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if t >= 2 else RED if e < 0 else GREY for e, t in zip(edges, ts)]\n"
            "ax.bar([str(h)+'d' for h in horizons], edges, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('holding window'); ax.set_ylabel('short edge vs base rate (bps/event)')\n"
            "ax.set_title('Shorting the shooting star: no edge at any horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, e, t in zip(horizons, edges, ts):\n"
            "    print(f'{h:2d}d: edge {e:+.1f} bps   HAC t {t:+.2f}')"
        ),
        md(
            f"At 5 days the short edge is **{R['star_h5_edge']:+.1f} bps** (*t* = "
            f"**{R['star_h5_t']:+.2f}**); at 10 days it's a flat **{R['star_h10_edge']:+.1f} bps**. "
            "Nothing reaches the +2 bar. The 'reversal' the star promises simply isn't in the "
            "forward returns."
        ),
        md(
            "**Now the sharper test: does the star geometry call the top *at all*?** Pool every "
            "star-shaped bar (ignore the trend label) and short it. If the wick is bearish, this "
            "should at least lean positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    anyr = st.run_experiment(panel, side='any', horizons=[1], n_draws=2000)\n"
            "    star_short = res[1]['edge_mean']*1e4\n"
            "    any_short = anyr[1]['edge_mean']*1e4\n"
            "    any_t = anyr[1]['t']\n"
            "else:\n"
            f"    star_short, any_short, any_t = {R['star_h1_edge']}, {R['any_h1_edge']}, {R['any_h1_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['Star\\n(uptrend, 1d)', 'Any star\\ngeometry (1d)'],\n"
            "       [star_short, any_short], color=[RED, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('short edge (bps/event)')\n"
            "ax.set_title('Shorting the star loses — the wick predicts continuation up')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Any-geometry short edge: {{any_short:+.1f}} bps, HAC t = {{any_t:+.2f}} '\n"
            "      f'(significant — but the WRONG sign for the claim)')"
        ),
        md(
            f"Shorting any star loses **{R['any_h1_edge']:+.1f} bps** at *t* = "
            f"**{R['any_h1_t']:+.2f}** — *significant, and on the wrong side*. A long upper wick, "
            "if it says anything, says the stock keeps climbing. That's the opposite of the "
            "folklore.\n\n"
            "> Why? A long upper wick on a large-cap is often just an intraday spike on news that "
            "the stock *grows into* over the following days — momentum, not exhaustion."
        ),
        md(
            "**The believer's rescue: 'only fade a star at the top of a *strong* rally.'** "
            "We keep only stars after a big trailing run-up and re-measure."
        ),
        code(
            "strengths = [0.0, 0.05, 0.10, 0.15]\n"
            "if HAVE_REAL:\n"
            "    fe, ft, fn = [], [], []\n"
            "    for ms in strengths:\n"
            "        cr = st.conditional_returns(panel, 5, side='star', min_strength=ms)\n"
            "        fe.append(cr['edge_mean']*1e4); ft.append(st.hac_t(cr['edge'])); fn.append(cr['n'])\n"
            "else:\n"
            f"    fe = [{R['f00_edge']},{R['f05_edge']},{R['f10_edge']},{R['f15_edge']}]\n"
            f"    ft = [{R['f00_t']},{R['f05_t']},{R['f10_t']},{R['f15_t']}]\n"
            f"    fn = [{R['f00_n']},{R['f05_n']},{R['f10_n']},{R['f15_n']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([f'>{int(s*100)}%' for s in strengths], fe, color=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('require trailing run-up of at least'); ax.set_ylabel('short edge (bps, 5d)')\n"
            "ax.set_title('The strong-rally filter just shrinks the sample — no edge appears')\n"
            "plt.tight_layout(); plt.show()\n"
            "for s, e, t, n in zip(strengths, fe, ft, fn):\n"
            "    print(f'run-up >{int(s*100):2d}%: n={n:5d}  edge {e:+.1f} bps  t {t:+.2f}')"
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Shorting the shooting star earns {R['star_h5_edge']:+.1f} bps "
            f"at 5 days (HAC *t* {R['star_h5_t']:+.2f}); the placebo says random days do this "
            f"easily (*p* = {R['star_h5_p']:.2f}). Nothing clears +2 at any horizon.\n"
            f"- **Tradability — Mirage.** The gross edge is negative or zero, and the short pays "
            "spread *and* borrow. There is no positive break-even cost.\n"
            f"- **Calls the top? — Busted.** Pooling every star, the short loses significantly "
            f"(*t* {R['any_h1_t']:+.2f}). The long upper wick leans toward *continuation up* — the "
            "exact opposite of the 'exhaustion' story."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Shorting is the *expensive* side: you pay the spread to get in and out, and you pay "
            "borrow every day you hold. Starting from a gross edge that's already below zero, "
            "every cost just digs deeper:"
        ),
        code(
            "costs = [0.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    cr = st.conditional_returns(panel, 5, side='star')\n"
            "    nets = [st.net_of_costs(cr['edge_mean'], cost_bps=c, borrow_bps=2.0*5)*1e4 for c in costs]\n"
            "else:\n"
            f"    nets = [{R['net_c0']}, {R['net_c5']}, {R['net_c10']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, nets, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net short edge (bps/event, 5d)')\n"
            "ax.set_title('Below zero before costs; borrow + spread only make it worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No positive break-even cost exists — the gross is already negative.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a *real* post-star decline "
            "in synthetic data and shows the harness catches it instantly (HAC *t* ≈ 15). The "
            "method works — the market just doesn't have the effect.\n"
            "- **The hammer.** The bullish mirror, same machinery, [Study 403](../../403-hammer-hanging-man/) "
            "— also no edge.\n"
            "- **Confirmation candles.** Some traders only act on a star if the *next* bar closes "
            "down. That's a different (two-bar) pattern — a fair fork, but beware it bakes in part "
            "of the move you're trying to predict.\n"
            "- **Intraday / other assets.** Lower timeframes or futures might behave differently; "
            "the on-domain test is yours to run.\n\n"
            "*Think the star works with a confirmation bar, a different threshold, or on another "
            "market? Fork this, change the rule, and show a short edge that clears HAC *t* = +2 and "
            "survives the placebo. That's the bar.*"
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
            "# Shooting Star — a quantitative teardown\n"
            "### Daily OHLC · exact-geometry detector · trend split · short-side forward returns · HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Calls the top%3F: Busted](https://img.shields.io/badge/Calls_the_top%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We detect the shooting-star "
            "geometry by precise OHLC rules, split by prior trend, trade it short, and test the "
            "conditional forward return against each name's own short base rate with a HAC *t*, a "
            "label-shuffle placebo, a trend-strength filter, and a synthetic positive control.\n\n"
            f"> **Not investment advice.** Real data: yfinance daily OHLC (un-adjusted, **price-only** "
            f"forward returns) for 26 US large-caps + SPY, full history, as-of **{R['asof']}**, "
            f"fingerprint `{R['fingerprint']}`. The offline core and the synthetic control are "
            "deterministic. Methods & sources in [`docs/references.md`](../docs/references.md); "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Star short edge **{R['star_h5_edge']:+.1f} bps** at 5d, HAC "
            f"*t* = **{R['star_h5_t']:+.2f}**; placebo *p* = **{R['star_h5_p']:.2f}**. No horizon clears +2. |\n"
            f"| **Tradability** | `MIRAGE` | Gross edge ≤ 0; the short pays spread **and** borrow. "
            f"Net at 5d ≈ **{R['net_c5']:.1f} bps**. No positive break-even cost. |\n"
            f"| **Calls the top?** | `BUSTED` | Pooling all stars, the short edge is "
            f"**{R['any_h1_edge']:+.1f} bps** at HAC *t* = **{R['any_h1_t']:+.2f}** — *significant, "
            "wrong sign*. The wick leans continuation-up. |\n\n"
            "> In plain words: the candle's silhouette looks like exhaustion, but the next few "
            "days don't agree. On large-caps a long upper wick is closer to a momentum spike "
            "than a top."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "For bar $t$ with open $O$, high $H$, low $L$, close $C$:\n\n"
            "$$\\text{body}=|C-O|,\\quad \\text{range}=H-L,\\quad "
            "u=H-\\max(O,C),\\quad \\ell=\\min(O,C)-L.$$\n\n"
            "The **shooting-star geometry** (steelmanned retail definition):\n\n"
            "$$u \\ge 2\\,\\text{body}, \\qquad \\ell \\le 0.5\\,\\text{body}, \\qquad "
            "\\text{body} \\le 0.35\\,\\text{range},$$\n\n"
            "with a small body floor so a doji can't trivially qualify. The folklore assigns the "
            "name by **prior trend** ($\\operatorname{sgn}(C_t/C_{t-10}-1)>0$ = uptrend = bearish "
            "shooting star). The traded claim:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[-r_{t+1\\to t+1+k}\\mid \\text{star}] - "
            "\\text{base}^{\\text{short}} > 0$ — shorting the star beats shorting a random day.\n"
            "- **H₂ (beats placebo).** That edge exceeds the same statistic on random entry days.\n"
            "- **H₃ (tradable).** It survives one-way costs ×2 **plus short borrow**.\n\n"
            "We will fail to reject the null on H₁/H₂ and find H₃ moot — and, pooling the "
            "geometry, find a *significant wrong-sign* short return."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The shooting star is one of the most-taught single-bar reversal signals. If H₁–H₃ "
            "held, it would be a zero-parameter top-timer. The finding that the short edge is "
            "*negative and significant* on pooled geometry matters: it says retail traders acting "
            "on 'exhaustion' candles are systematically shorting into continuation, paying spread "
            "and borrow for the privilege — the same lesson the hammer teardown delivered on the "
            "long side."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** Exact OHLC geometry above; `is_star_shape` (params $2.0 / 0.5 / 0.35$).\n"
            "- **Trend split.** `trend_at` = sign of the trailing 10-day return at close $t$ "
            "(no look-ahead). Star = geometry ∧ uptrend.\n"
            "- **Timing.** One execution lag: signal at close $t$, enter the short at close "
            "$t{+}1$, exit at close $t{+}1{+}k$ for $k\\in\\{1,3,5,10\\}$.\n"
            "- **Edge.** Conditional short return minus each name's own short base rate "
            "(unconditional mean short return at that horizon) — controls for each name's drift.\n"
            "- **Inference.** Newey–West HAC *t* on the pooled edge (overlapping windows ⇒ i.i.d. "
            "*t* would overstate); a per-name label-shuffle placebo (`p = P[shuffled ≥ observed]`).\n"
            "- **Myth-check.** A trend-strength filter (`min_strength`) keeps only stars after a "
            "big run-up — the believer's 'fade only strong rallies' refinement.\n"
            "- **Positive control.** Synthetic OHLC with a *planted* post-star decline.\n\n"
            "Basket: 26 US large-caps + SPY (full yfinance history). **Survivorship** is named on "
            "the Signal axis — and for a *bearish* claim it cuts *against* the short (survivors "
            "kept rising), so a null short edge is partly mechanical."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Horizon sweep — the short edge and its HAC *t*\n\n"
            "If the reversal were real, some horizon's edge should clear *t* = +2."
        ),
        code(
            "horizons = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    panel = real_panel()\n"
            "    res = st.run_experiment(panel, side='star', n_draws=5000)\n"
            "    edges = [res[h]['edge_mean']*1e4 for h in horizons]\n"
            "    ts = [res[h]['t'] for h in horizons]\n"
            "    ps = [res[h]['p_placebo'] for h in horizons]\n"
            "    ns = [res[h]['n'] for h in horizons]\n"
            "else:\n"
            f"    edges = [{R['star_h1_edge']},{R['star_h3_edge']},{R['star_h5_edge']},{R['star_h10_edge']}]\n"
            f"    ts = [{R['star_h1_t']},{R['star_h3_t']},{R['star_h5_t']},{R['star_h10_t']}]\n"
            f"    ps = [{R['star_h1_p']},{R['star_h3_p']},{R['star_h5_p']},{R['star_h10_p']}]\n"
            f"    ns = [{R['star_h1_n']},{R['star_h3_n']},{R['star_h5_n']},{R['star_h10_n']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(h)+'d' for h in horizons], ts, color=[GREEN if t>=2 else GREY for t in ts])\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylim(-3, 3)\n"
            "ax.set_xlabel('holding window'); ax.set_ylabel('HAC t on short edge')\n"
            "ax.set_title('Shooting-star short: HAC t never approaches +2')\n"
            "plt.tight_layout(); plt.show()\n"
            "tbl = pd.DataFrame({'horizon': horizons, 'n': ns, 'edge_bps': np.round(edges,1),\n"
            "                    'HAC_t': np.round(ts,2), 'placebo_p': np.round(ps,3)})\n"
            "tbl"
        ),
        md(
            f"> In plain words: {R['star_h1_n']:,} uptrend stars, and the short edge sits at noise "
            f"({R['star_h5_edge']:+.1f} bps at 5d, *t* {R['star_h5_t']:+.2f}). The placebo *p*-values "
            "are all far above 0.05 — random days clear this bar with ease."
        ),
        md(
            "### 4b · The pooled geometry — a significant *wrong-sign* short\n\n"
            "Drop the trend label and short every star. The bearish story predicts a positive "
            "short edge; the tape says the opposite."
        ),
        code(
            "if HAVE_REAL:\n"
            "    anyr = st.run_experiment(panel, side='any', horizons=[1,3,5], n_draws=5000)\n"
            "    ae = [anyr[h]['edge_mean']*1e4 for h in [1,3,5]]\n"
            "    at = [anyr[h]['t'] for h in [1,3,5]]\n"
            "else:\n"
            f"    ae = [{R['any_h1_edge']},{R['any_h3_edge']},{R['any_h5_edge']}]\n"
            f"    at = [{R['any_h1_t']},{R['any_h3_t']},{R['any_h5_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['1d','3d','5d'], at, color=RED)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylim(-3,3)\n"
            "ax.set_ylabel('HAC t on short edge (all stars)')\n"
            "ax.set_title('Shorting any star: significantly NEGATIVE — the wrong side')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, e, t in zip([1,3,5], ae, at):\n"
            "    print(f'{h}d: short edge {e:+.1f} bps, HAC t {t:+.2f}')"
        ),
        md(
            f"> In plain words: at 1/3/5 days the all-star short edge is {R['any_h1_edge']:+.1f} / "
            f"{R['any_h3_edge']:+.1f} / {R['any_h5_edge']:+.1f} bps with HAC *t* of "
            f"{R['any_h1_t']:+.2f} / {R['any_h3_t']:+.2f} / {R['any_h5_t']:+.2f}. A long upper wick "
            "predicts *continuation up* — fading it is a reliable way to be wrong."
        ),
        md(
            "### 4c · Per-name breakdown — no name carries the short\n\n"
            "26 names. If a true effect existed, several should line up positive. They don't — "
            "and the single positive *t* > 2 is exactly what 26 independent draws produce by chance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for tk, bars in panel.items():\n"
            "        cr = st.conditional_returns({tk: bars}, 5, side='star')\n"
            "        recs.append((tk, cr['n'], cr['edge_mean']*1e4, st.hac_t(cr['edge'])))\n"
            "    per = pd.DataFrame(recs, columns=['ticker','n','edge_bps','t']).sort_values('t')\n"
            "    npos = int((per['t'] > 2).sum())\n"
            "else:\n"
            "    per = pd.DataFrame({'ticker':['GS','...','PG'], 'edge_bps':[-93.1,0,34.1], 't':[-1.50,0,2.38]})\n"
            f"    npos = 1\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(per['ticker'], per['t'], color=[GREEN if t>2 else GREY for t in per['t']])\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); plt.xticks(rotation=90)\n"
            "ax.set_ylabel('HAC t (star short, 5d)')\n"
            "ax.set_title(f'Per-name: only {npos} of 26 clears +2 — pure multiple testing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{npos} of 26 names have HAC t > 2 (chance alone produces ~1 at the 5% level)')"
        ),
        md(
            f"> In plain words: one name (PG, *t* = {R['pg_t']:.2f}) is positive-significant out of "
            "26 — the textbook signature of multiple testing, not a real effect. The big movers "
            f"are *negative* (GS {R['best_neg_gs']:+.0f} bps)."
        ),
        md(
            "### 4d · Myth-check — does 'only fade strong rallies' rescue it?\n\n"
            "Keep only stars after a large trailing run-up."
        ),
        code(
            "strengths = [0.0, 0.05, 0.10, 0.15]\n"
            "if HAVE_REAL:\n"
            "    fe, ft, fn = [], [], []\n"
            "    for ms in strengths:\n"
            "        cr = st.conditional_returns(panel, 5, side='star', min_strength=ms)\n"
            "        fe.append(cr['edge_mean']*1e4); ft.append(st.hac_t(cr['edge'])); fn.append(cr['n'])\n"
            "else:\n"
            f"    fe=[{R['f00_edge']},{R['f05_edge']},{R['f10_edge']},{R['f15_edge']}]\n"
            f"    ft=[{R['f00_t']},{R['f05_t']},{R['f10_t']},{R['f15_t']}]\n"
            f"    fn=[{R['f00_n']},{R['f05_n']},{R['f10_n']},{R['f15_n']}]\n"
            "tbl = pd.DataFrame({'min_runup': strengths, 'n': fn, 'edge_bps': np.round(fe,1),\n"
            "                    'HAC_t': np.round(ft,2)})\n"
            "print(tbl.to_string(index=False))\n"
            "print('\\nThe filter only thins the sample; the edge stays indistinguishable from 0.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — star short edge {R['star_h5_edge']:+.1f} bps at 5d, HAC *t* "
            f"{R['star_h5_t']:+.2f}, placebo *p* {R['star_h5_p']:.2f}; no horizon, no filter, "
            "no name carries a positive short edge past +2 (the one that does is multiple testing).\n"
            f"- **Tradability `MIRAGE`** — gross ≤ 0 and the short pays spread + borrow; net at 5d "
            f"≈ {R['net_c5']:.1f} bps. No positive break-even cost.\n"
            f"- **Calls the top? `BUSTED`** — pooled-geometry short edge {R['any_h1_edge']:+.1f} bps "
            f"at HAC *t* {R['any_h1_t']:+.2f} (significant, wrong sign): the long upper wick leans "
            "toward continuation up."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "The star is a short, the most cost-laden side: spread twice plus borrow per day held. "
            "Starting below zero, costs only deepen the loss:"
        ),
        code(
            "costs = [0.0, 1.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    cr = st.conditional_returns(panel, 5, side='star')\n"
            "    nets = [st.net_of_costs(cr['edge_mean'], cost_bps=c, borrow_bps=2.0*5)*1e4 for c in costs]\n"
            "else:\n"
            f"    nets = [{R['net_c0']}, -18.1, {R['net_c5']}, {R['net_c10']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.fill_between(costs, nets, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)  [+ 10 bps borrow over 5d]')\n"
            "ax.set_ylabel('net short edge (bps/event, 5d)')\n"
            "ax.set_title('No positive break-even cost — the gross is already negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even cost: none — the signal, not the cost, is the problem.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the engine even capable of finding a post-star decline? Plant one in synthetic "
            "OHLC and check the harness recovers it (and stays silent when it's absent):"
        ),
        code(
            "p0, _ = data.synthetic_panel(edge=0.0, seed=404)\n"
            "p1, _ = data.synthetic_panel(edge=0.01, seed=404)\n"
            "r0 = st.run_experiment(p0, side='star', horizons=[1], n_draws=1000)[1]\n"
            "r1 = st.run_experiment(p1, side='star', horizons=[1], n_draws=1000)[1]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['null (edge=0)','planted decline'], [r0['t'], r1['t']],\n"
            "       color=[GREY, GREEN])\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t on planted short edge (1d)')\n"
            "ax.set_title('The harness catches a real top when one exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"null:    edge {r0['edge_mean']*1e4:+.1f} bps, t {r0['t']:+.2f}, p {r0['p_placebo']:.3f}\")\n"
            "print(f\"planted: edge {r1['edge_mean']*1e4:+.1f} bps, t {r1['t']:+.2f}, p {r1['p_placebo']:.3f}\")"
        ),
        md(
            "The engine is a faithful detector: on a planted post-star decline it returns HAC "
            f"*t* ≈ {R['syn_plant_t']:.0f} and placebo *p* = 0; on a pure random walk it sits at "
            f"*t* ≈ {R['syn_null_t']:.2f}, *p* ≈ {R['syn_null_p']:.2f}. The real-tape `NONE` is "
            "therefore a statement about the **market** (large-cap upper wicks aren't tops), not "
            "the method. Forks worth trying: a confirmation-bar variant, intraday timeframes, or "
            "single-name small-caps where reversals may be sharper — show a short edge clearing "
            "HAC *t* = +2 that survives the placebo."
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
