"""Generate the two narrative notebooks for Study 308 (Cocoa-Squeeze).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
blow-off figures run anywhere, offline and deterministic; the real-tape cells use the
cached daily parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-29).
R = dict(
    asof="2026-05-29", fp="077d24a7d94b", n_days=6622,
    trough_date="2024-01-08", trough_px=4094, peak_date="2024-12-18", peak_px=12565,
    runup=3.1, fwd_dd=-77.7, fwd20=-11.1, fwd60=-36.2, fwd250=-52.3,
    # Momentum ("ride it") on the full real tape
    mom_t=0.08, mom_ann=0.1, mom_active=131,
    mom_cost5_t=-0.05, mom_cost10_t=-0.18, mom_cost20_t=-0.44,
    mom_2024_ann=12.7, mom_2024_t=0.91, mom_2024_active=33,
    # Crack-fade ("short it") on the full real tape (shorts pay 300 bps/yr borrow)
    fade_t=-1.99, fade_ann=-2.6, fade_active=111, fade_mean_bps=-61.5,
    fade_ci_lo=-121.8, fade_ci_hi=-12.6,
    bh_ann=6.1,
    # Synthetic positive control (machinery proof only)
    synth_mom_t=3.76, synth_mom_ann=11.8, synth_null_t=1.52,
    synth_runup=2.0, synth_fwd_dd=-92.1,
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

from cocoa_squeeze import data, strategy as st

TICKER = "CC=F"

def _have_cache():
    return os.path.exists(data._cache_path(TICKER, data.DEFAULT_CACHE))

def load_real():
    \"\"\"Real cocoa tape if cached/fetchable; else None (cells fall back to synthetic).\"\"\"
    try:
        b = data.fetch_daily(TICKER, fetch=False)
        return b[b.index <= "2026-05-31"]
    except Exception:
        return None

REAL = load_real()
HAVE_REAL = REAL is not None
print("real cocoa cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cocoa-Squeeze — should you have ridden it, or shorted it? 🍫\n"
            "### After cocoa tripled and then crashed 78%, was there an edge — or just a freak?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_repeatable_edge%3F: Not_supported](https://img.shields.io/badge/A_repeatable_edge%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "In 2024 cocoa futures went **vertical** — from about $4,100 a tonne to an all-time high "
            "near **$12,600** — and then gave back roughly three-quarters of the move. Looking at the "
            "chart afterward, two voices argue in your head: *'trend is your friend, you should have "
            "ridden it'* and *'parabolas always crash, you should have shorted it.'* This notebook "
            "asks the only question that matters: **would either trade actually have made money?**\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap intervals "
            "and the borrow costs? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Was the move real? | **Spectacularly.** Cocoa ran **{R['runup']:.1f}×** "
            f"(${R['trough_px']:,}→${R['peak_px']:,}/t) then crashed **{R['fwd_dd']:.0f}%** from "
            "the top. A genuine, once-in-a-generation supply shock. |\n"
            f"| Should you have *ridden* it? | **No edge.** A 'long while it's stretched and rising' "
            f"timer earned **+{R['mom_ann']:.1f}%/yr** (*t* = +{R['mom_t']:.2f}) — a coin flip — and "
            "goes **negative** the moment you pay any cost. You whipsaw in and out of the gaps. |\n"
            f"| Should you have *shorted* it? | **It loses money.** A disciplined 'short the crack' "
            f"fade earned **{R['fade_ann']:.1f}%/yr** (*t* = {R['fade_t']:.2f}); the bear-market "
            "rallies on the way down stop you out. The widow-maker is real. |\n"
            "| So what is it? | A **once-off freak**, not a strategy. One vertical chart can't "
            "certify any signal — and both ways to bet on it failed. |\n\n"
            "> The cocoa squeeze is the perfect trap: the hindsight chart screams 'obvious trade', "
            "but ride-it earned nothing and fade-it lost money. A single, unrepeatable event is a "
            "story, not an edge."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Two pieces of trading folklore, both stated at full strength:\n\n"
            "> *\"**Trend is your friend.** When a market is making new highs and accelerating, you "
            "ride it — momentum works (Moskowitz–Ooi–Pedersen 2012).\"*\n\n"
            "> *\"**Parabolas always crash.** When a market goes vertical and grotesquely far above "
            "trend, you fade it — overreaction reverses (De Bondt–Thaler 1985; Sornette).\"*\n\n"
            "Cocoa in 2024 is the ultimate test case: a real, fundamental shock (back-to-back West "
            "African crop failures) drove a clean three-bagger and a clean crash. If *either* folk "
            "rule has teeth, this is where it should bite."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Everyone *feels* they should have traded the cocoa chart — it's the most-screenshotted "
            "commodity move in years. If 'ride the squeeze' or 'fade the blow-off' actually paid, "
            "you'd have a rule for the *next* parabola (lumber, eggs, uranium, the next meme stock). "
            "And if neither pays — if the famous chart is a mirage — that's the more valuable lesson, "
            "because it inoculates you against the most seductive hindsight trade there is."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the 'stretch.'** How far is the price above its trend, in volatility "
            "units? That's our blow-off gauge.\n"
            "2. **Ride it.** Go long while the stretch is high *and* still rising — then check, "
            "honestly (one-day execution lag, real costs), whether it made money.\n"
            "3. **Fade it.** Short the crack once the parabola rolls over — and make the short pay "
            "borrow, like a real short does.\n"
            "4. **Sanity-check the engine.** Build a *fake* parabola where we *know* there's an edge, "
            "and confirm our rules find it. If they can find a planted edge but not cocoa's, then "
            "cocoa simply didn't have one.\n\n"
            "Tape: cocoa front-month futures (CC=F) daily, since 2000. **Mirage test:** if both "
            "trades are coin flips or losers once costs are paid, it's a mirage."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the chart everyone remembers.** Here is cocoa, with the run-up and the crack:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = REAL['close']\n"
            "    info = st.find_blowoff_peak(close)\n"
            "    runup, fwd_dd = info['runup_mult'], info['fwd_drawdown']\n"
            "    peak_date, peak_px = info['peak_date'], info['peak_px']\n"
            "    seg = close['2022-01-01':]\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "    ax.plot(seg.index, seg.values, color=RED, lw=1.6)\n"
            "    ax.axvline(peak_date, ls='--', c=GREY)\n"
            "    ax.annotate(f'peak ${peak_px:,.0f}/t', (peak_date, peak_px),\n"
            "                xytext=(-90, -4), textcoords='offset points', color=GREY)\n"
            "    ax.set_ylabel('cocoa front-month ($/t, price-only)')\n"
            "    ax.set_title(f'Cocoa: a {runup:.1f}x parabola, then {fwd_dd*100:.0f}% give-back')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Run-up {runup:.1f}x to ${peak_px:,.0f}; worst forward drawdown {fwd_dd*100:.0f}%')\n"
            "else:\n"
            "    # OFFLINE: draw the planted synthetic blow-off as a stand-in, clearly bannered.\n"
            "    bb, truth = data.synthetic_blowoff(bubble=1.0, seed=308)\n"
            "    info = st.find_blowoff_peak(bb['close'])\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "    ax.plot(bb.index, bb['close'].values, color=GREY, lw=1.4)\n"
            "    ax.set_title('[SYNTHETIC STAND-IN — no real cache] a planted parabola')\n"
            "    ax.set_ylabel('synthetic price'); plt.tight_layout(); plt.show()\n"
            f"    print('OFFLINE — real cocoa ran {R['runup']:.1f}x to ${R['peak_px']:,}/t, then {R['fwd_dd']:.0f}% (docs/results.md)')"
        ),
        md(
            f"A **{R['runup']:.1f}×** run to **${R['peak_px']:,}/t** by December 2024, then a "
            f"**{R['fwd_dd']:.0f}%** collapse. In hindsight it looks like free money in both "
            "directions. Now let's see if you could have *taken* it."
        ),
        md(
            "**Could you have ridden it?** Long while cocoa is stretched above trend and still "
            "climbing — gross, and then net of a realistic cost:"
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    mom = st.momentum_signal(REAL['close'])\n"
            "    tser = [st.hac_tstat(st.book_returns(REAL['close'], mom, cost_bps=c).pipe(lambda s: s[s!=0]).to_numpy()) for c in costs]\n"
            "else:\n"
            f"    tser = [{R['mom_t']}, {R['mom_cost5_t']}, {R['mom_cost10_t']}, {R['mom_cost20_t']}]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, tser, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='|t|=2 (real)'); ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(-2, ls='--', c=RED)\n"
            "ax.set_xlabel('round-trip cost (one-way bps)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_ylim(-3, 3); ax.legend()\n"
            "ax.set_title('Riding the parabola: a coin flip gross, a loser net')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Gross t = {{tser[0]:+.2f}} (≈0); goes negative at the first basis point of cost.')"
        ),
        md(
            f"The 'ride it' timer earns HAC *t* = **+{R['mom_t']:.2f}** gross — a coin flip — and "
            "turns **negative** at the very first basis point of cost. Even inside the 2024 blow-off "
            f"window it was *t* = +{R['mom_2024_t']:.2f} (on {R['mom_2024_active']} days): one lucky "
            "episode, not an edge. Parabolas gap both ways; the timer gets chopped to bits."
        ),
        md(
            "**Could you have shorted it?** The disciplined fade — short the crack *after* it rolls "
            "over (you don't stand in front of a rising parabola), with the short paying borrow:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ck = st.crack_signal(REAL['close'])\n"
            "    r = st.book_returns(REAL['close'], ck, borrow_bps_yr=300)\n"
            "    s = st.summarize(r); fade_ann, fade_t = s['ann_return_pct'], s['tstat']\n"
            "    lo, hi = s['ci_lo_bps'], s['ci_hi_bps']\n"
            "else:\n"
            f"    fade_ann, fade_t = {R['fade_ann']}, {R['fade_t']}\n"
            f"    lo, hi = {R['fade_ci_lo']}, {R['fade_ci_hi']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['Crack-fade short'], [(lo+hi)/2], yerr=[[(lo+hi)/2-lo],[hi-(lo+hi)/2]],\n"
            "       color=RED, width=.4, capsize=8)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean daily return (bps), 95% CI')\n"
            "ax.set_title('Fading the blow-off: the whole interval is BELOW zero — it lost money')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Fade: {fade_ann:+.1f}%/yr, t={fade_t:+.2f}; mean-return 95% CI = [{lo:+.1f}, {hi:+.1f}] bps/day')"
        ),
        md(
            f"The fade **lost money** — **{R['fade_ann']:.1f}%/yr** (*t* = {R['fade_t']:.2f}), and its "
            f"95% confidence interval for the average day is **entirely below zero** "
            f"([{R['fade_ci_lo']:.0f}, {R['fade_ci_hi']:.0f}] bps). The crash was real, but it came "
            "with vicious bear-market rallies that stop a short out. This is the widow-maker the old "
            "hands warn about."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Ride-it *t* = +{R['mom_t']:.2f}, fade-it *t* = {R['fade_t']:.2f} "
            "(a loss). One blow-off is one data point — there's no cross-section to certify anything.\n"
            "- **Tradability — Mirage.** Momentum dies at the first basis point; the fade's whole "
            "confidence interval is negative once shorts pay borrow. Nothing survives costs.\n"
            f"- **A repeatable edge? — Not supported.** The {R['fwd_dd']:.0f}% give-back looks "
            "shortable in hindsight, but it was a crop-failure supply shock — a freak, not a pattern."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and notice it's a *double* failure. Most desk mirages survive on paper and die on "
            "costs; this one barely survives even gross. Ride-it is a coin flip you can't compound; "
            f"fade-it loses money outright. And the deeper problem is **capacity of evidence**: with "
            "a *single* event there is nothing to size, nothing to repeat, and no honest *t*-stat. "
            "The only winning trade was buy-and-hold cocoa from the 2024 low to the peak — which is "
            "to say, you'd have needed to know the supply shock in advance. That's a weather forecast, "
            "not a strategy."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is the *engine* broken, or did cocoa just not cooperate?** The next notebook plants "
            "a *fake* parabola with a known edge and confirms our rules find it — so the null result "
            "on cocoa is real, not a bug.\n"
            "- **The right way to harvest reversion** is a *basket*, not a single chart — see "
            "[Study 196 — Long-Term-Reversal](../../196-long-term-reversal/).\n"
            "- **The single-commodity widow-maker is a genre** — see "
            "[Study 227 — Natgas-Winter](../../227-natgas-winter/), the winter-gas spike that's "
            "equally famous and equally untradable.\n\n"
            "*Think you can time a parabola? Fork this, add your favourite blow-off rule, and show "
            "it clearing |t|=2 across more than one event — that's the bar.*"
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
            "# Cocoa-Squeeze — a quantitative teardown 🔬\n"
            "### Single-asset event · one execution lag · HAC *t* · block-bootstrap CIs · "
            "borrow-aware shorts · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_repeatable_edge%3F: Not_supported](https://img.shields.io/badge/A_repeatable_edge%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether either folk "
            "reaction to the 2024 cocoa parabola (ride the momentum / fade the crack) survives honest "
            "inference, and we use a planted synthetic blow-off to prove the engine is a faithful "
            "detector — so the null on cocoa is a fact about cocoa, not a broken pipeline.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, cocoa front-month `CC=F` "
            "(a **price-only** continuous roll, *not* total return) since 2000; as-of 2026-05-29. "
            "The offline core and tests run on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Real tape: momentum HAC *t* = **+{R['mom_t']:.2f}**, "
            f"crack-fade *t* = **{R['fade_t']:.2f}** (losing, CI entirely < 0). n = 1 blow-off → no "
            f"cross-section. Synthetic control *t* = +{R['synth_mom_t']:.2f} proves *machinery only*. |\n"
            f"| **Tradability** | `MIRAGE` | Momentum negative at ≥ 1 bp cost; fade mean-return 95% CI "
            f"= [{R['fade_ci_lo']:.0f}, {R['fade_ci_hi']:.0f}] bps/day after borrow. |\n"
            f"| **A repeatable edge?** | `NOT SUPPORTED` | {R['fwd_dd']:.0f}% give-back is a "
            "crop-failure supply shock — one event, not a harvestable pattern. |\n\n"
            "> 💡 In plain words: a vertical chart is not a signal. Both ways to bet on the cocoa "
            "parabola failed on the real tape, and a single event cannot certify anything — the "
            "synthetic control exists precisely so we can say the detector works and the market "
            "still gave us nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $z_t = (\\log P_t - \\mathrm{EMA}_{100}(\\log P)_t)/\\sigma_t$ be the volatility-"
            "scaled *stretch* above trend. Two competing hypotheses:\n\n"
            "- **H₁ (momentum / ride).** $\\mathbb{E}[r_{t+1}\\mid z_t>1,\\ z_t>z_{t-5}]>0$ — long "
            "while stretched and rising pays.\n"
            "- **H₂ (reversion / fade).** $\\mathbb{E}[r_{t+1}\\mid \\text{post-peak crack}]<0$ — "
            "shorting the roll-over pays.\n"
            "- **H₃ (machinery).** On a tape with a *planted* mean-reverting blow-off, the momentum "
            "engine recovers a robust edge (and ≈ 0 on a random walk).\n\n"
            "We find **H₁ not supported** (*t* ≈ 0 gross, negative net), **H₂ rejected** (the fade "
            "*loses*, CI < 0), **H₃ confirmed** — the detector works, cocoa just had no harvestable "
            "edge."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Time-series momentum (Moskowitz–Ooi–Pedersen 2012) and commodity momentum (Miffre–Rallis "
            "2007) are *diversified, many-market* premia; the interesting question is whether they "
            "collapse on a *single* parabolic name. Greenwood–Shleifer–You (2019) show run-ups raise "
            "crash odds but are hard to time — the steelman for fading, and the reason it's a "
            "widow-maker. Settling both on the most extreme commodity move in years, with honest "
            "inference, is worth more than the binary verdict: it shows the famous chart is a "
            "selection artefact."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Stretch.** $z_t$ from closes up to and including $t$ (no look-ahead in the gauge).\n"
            "- **Entry/exit.** Momentum: long while $z_t>1$ and rising. Fade: short 20 days after a "
            "≥7% pullback from a multi-month high.\n"
            "- **Execution.** One lag — the position at the close of $t$ earns $r_{t+1}$ (a single "
            "`shift`). Costs **one-way × |Δposition| × NAV**; shorts pay a 300 bps/yr borrow leg.\n"
            "- **Inference.** Newey–West HAC *t* on daily returns; circular **block-bootstrap** 95% "
            "CI of the mean (block = 20); cost sweep.\n"
            "- **Positive control.** Deterministic `synthetic_blowoff`: a planted parabola that "
            "mean-reverts (`bubble=1`) vs a pure random walk (`bubble=0`, the null).\n\n"
            "One asset (`CC=F`), so Sharpe is on the strategy's own returns — there is no cross-asset "
            "race to run excess-vs-excess against."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Anatomy of the blow-off\n\n"
            "Locate the parabola: pre-blowoff trough, the price top, the run-up multiple, the forward "
            "drawdown."
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = REAL['close']; info = st.find_blowoff_peak(close)\n"
            "    print(f\"trough {info['trough_date'].date()} ${info['trough_px']:,.0f}  ->  \"\n"
            "          f\"peak {info['peak_date'].date()} ${info['peak_px']:,.0f}\")\n"
            "    print(f\"run-up {info['runup_mult']:.1f}x ; worst forward drawdown {info['fwd_drawdown']*100:.0f}%\")\n"
            "    z = st.stretch_z(close)\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    seg = close['2022-01-01':]; zseg = z['2022-01-01':]\n"
            "    a1.plot(seg.index, seg.values, c=RED, lw=1.5); a1.set_ylabel('$/t')\n"
            "    a1.set_title('Cocoa front-month (price-only)')\n"
            "    a2.plot(zseg.index, zseg.values, c=GREY, lw=1.2); a2.axhline(0, c='k', lw=.8)\n"
            "    a2.axhline(1, ls='--', c=AMBER); a2.set_ylabel('stretch z')\n"
            "    a2.set_title('Stretch above trend (vol units)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    bb, truth = data.synthetic_blowoff(bubble=1.0, seed=308)\n"
            "    info = st.find_blowoff_peak(bb['close'])\n"
            "    print('[OFFLINE — synthetic stand-in]')\n"
            f"    print('real: trough {R['trough_date']} ${R['trough_px']:,} -> peak {R['peak_date']} ${R['peak_px']:,}; '\n"
            f"          'run-up {R['runup']:.1f}x, fwd dd {R['fwd_dd']:.0f}% (docs/results.md)')\n"
            "    fig, ax = plt.subplots(figsize=(10, 3.6))\n"
            "    ax.plot(bb.index, bb['close'].values, c=GREY); ax.set_title('synthetic planted parabola')\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: a **{R['runup']:.1f}×** run to **${R['peak_px']:,}/t** and a "
            f"**{R['fwd_dd']:.0f}%** crash. The stretch gauge peaks *during* the steepest ascent (the "
            "vol denominator inflates near the very top) — already a hint that timing this is hard."
        ),
        md(
            "### 4b · H₁ — ride the momentum (the cost sweep)\n\n"
            "HAC *t* of the long-while-stretched-and-rising book, gross and net."
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    mom = st.momentum_signal(REAL['close'])\n"
            "    rows = []\n"
            "    for c in costs:\n"
            "        r = st.book_returns(REAL['close'], mom, cost_bps=c); s = st.summarize(r)\n"
            "        rows.append((c, s['ann_return_pct'], s['tstat']))\n"
            "    dfm = pd.DataFrame(rows, columns=['cost_bps','ann%','t'])\n"
            "else:\n"
            "    dfm = pd.DataFrame({'cost_bps':costs,\n"
            f"        'ann%':[{R['mom_ann']}, -0.1, -0.2, -0.5],\n"
            f"        't':[{R['mom_t']}, {R['mom_cost5_t']}, {R['mom_cost10_t']}, {R['mom_cost20_t']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "col = [GREEN if t>=2 else (RED if t<=-0 else AMBER) for t in dfm['t']]\n"
            "ax.bar(dfm['cost_bps'].astype(str), dfm['t'], color=col, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREEN); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('HAC t'); ax.set_ylim(-1, 2.5)\n"
            "ax.set_title('Momentum on cocoa: never near |t|=2, negative once costed')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfm.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: gross HAC *t* = **+{R['mom_t']:.2f}**, nowhere near the bar, and "
            f"negative net. Even confined to the 2024 episode it was *t* = +{R['mom_2024_t']:.2f} on "
            f"{R['mom_2024_active']} days. **H₁ not supported.**"
        ),
        md(
            "### 4c · H₂ — fade the crack (with borrow), and its bootstrap interval\n\n"
            "The disciplined short, paying borrow, with a block-bootstrap CI of the mean daily return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ck = st.crack_signal(REAL['close'])\n"
            "    r = st.book_returns(REAL['close'], ck, borrow_bps_yr=300)\n"
            "    s = st.summarize(r)\n"
            "    ann, t, lo, hi, na = s['ann_return_pct'], s['tstat'], s['ci_lo_bps'], s['ci_hi_bps'], s['n_active']\n"
            "else:\n"
            f"    ann, t, lo, hi, na = {R['fade_ann']}, {R['fade_t']}, {R['fade_ci_lo']}, {R['fade_ci_hi']}, {R['fade_active']}\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "mid = (lo+hi)/2\n"
            "ax.barh(['Crack-fade short'], [mid], xerr=[[mid-lo],[hi-mid]], color=RED, capsize=8, height=.4)\n"
            "ax.axvline(0, c='k', lw=1); ax.set_xlabel('mean daily return (bps), 95% block-bootstrap CI')\n"
            "ax.set_title('The fade lost money — the whole interval sits below zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Fade: {na} active days, {ann:+.1f}%/yr, HAC t={t:+.2f}, mean-return CI [{lo:+.1f},{hi:+.1f}] bps/day')"
        ),
        md(
            f"> 💡 In plain words: the fade is **{R['fade_ann']:.1f}%/yr** (*t* = {R['fade_t']:.2f}); "
            f"its mean-return 95% CI **[{R['fade_ci_lo']:.0f}, {R['fade_ci_hi']:.0f}] bps/day** lies "
            "entirely below zero. The crash was real but punctuated by violent rallies — shorting it "
            "is a reliable way to lose money. **H₂ rejected.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — momentum *t* = +{R['mom_t']:.2f}, fade *t* = {R['fade_t']:.2f} "
            "(losing). A single blow-off is one observation; no honest *t*-stat certifies a signal "
            "from n = 1.\n"
            f"- **Tradability `MIRAGE`** — momentum negative at the first basis point; fade CI "
            f"[{R['fade_ci_lo']:.0f}, {R['fade_ci_hi']:.0f}] bps/day after borrow. Nothing survives.\n"
            f"- **A repeatable edge? `NOT SUPPORTED`** — the {R['fwd_dd']:.0f}% give-back was a "
            "crop-failure supply shock, not a recurring pattern."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity-of-evidence problem\n\n"
            "The usual desk capacity question (square-root impact, break-even cost) is almost beside "
            "the point here, because the binding constraint is upstream: **there is one event.** "
            "Cocoa B&H returned ~{bh}%/yr over the full sample, but that's hindsight on a supply "
            "shock you'd have had to forecast. You can't size, repeat, or compound a single chart, "
            "and the only trades you *could* systematise (ride / fade) earned nothing and lost money "
            "respectively.".replace("{bh}", f"{R['bh_ann']:.1f}")
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            f"vals = [{R['mom_ann']}, {R['fade_ann']}, {R['bh_ann']}]\n"
            "labels = ['Ride momentum', 'Fade crack\\n(after borrow)', 'Buy & hold\\n(hindsight)']\n"
            "col = [AMBER, RED, GREY]\n"
            "ax.bar(labels, vals, color=col, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %/yr')\n"
            "ax.set_title('Systematic trades earned nothing or lost; only hindsight B&H \"won\"')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Ride {R['mom_ann']:+.1f}%/yr  Fade {R['fade_ann']:+.1f}%/yr  B&H {R['bh_ann']:+.1f}%/yr (full sample, hindsight)')"
        ),
        md(
            "> 💡 In plain words: the only winner needed a weather forecast, not a rule. That is the "
            "signature of a **Mirage**: a chart that looks like alpha and a backtest that delivers a "
            "coin flip and a loss."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* faithful, or does it always print ≈ 0? Plant a parabola that genuinely "
            "mean-reverts and confirm the momentum book banks it — and that on a pure random walk it "
            "does not."
        ),
        code(
            "rows = []\n"
            "for bub, lab in [(0.0, 'Random walk (null)'), (1.0, 'Planted blow-off')]:\n"
            "    bb, _ = data.synthetic_blowoff(n_days=2500, bubble=bub, seed=308)\n"
            "    r = st.book_returns(bb['close'], st.momentum_signal(bb['close']))\n"
            "    s = st.summarize(r); rows.append((lab, s['ann_return_pct'], s['tstat']))\n"
            "dfs = pd.DataFrame(rows, columns=['tape','ann%','t'])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "col = [GREY, GREEN]\n"
            "ax.bar(dfs['tape'], dfs['t'], color=col, width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('momentum HAC t'); ax.set_title('The engine finds a PLANTED blow-off, not noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfs.round(2).to_string(index=False))"
        ),
        md(
            f"The momentum engine clears the bar on the planted blow-off (*t* = +{R['synth_mom_t']:.2f}, "
            f"~+{R['synth_mom_ann']:.0f}%/yr) and sits below it on the random walk "
            f"(*t* = +{R['synth_null_t']:.2f}). **The detector works** — so cocoa's null is a "
            "statement about cocoa, not a broken pipeline. *A synthetic control proves machinery, "
            "never markets; it can never back a Signal stamp.*\n\n"
            "For the *right* way to harvest reversion (a cross-section, not one chart) see "
            "[Study 196 — Long-Term-Reversal](../../196-long-term-reversal/); for the single-commodity "
            "widow-maker genre, [Study 227 — Natgas-Winter](../../227-natgas-winter/)."
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
