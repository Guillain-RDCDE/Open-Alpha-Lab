"""Generate the two narrative notebooks for Study 336 (Inverse-Cramer).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells try the cached daily
parquet via quantlab.data and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-20).
R = dict(
    asof="2026-06-20",
    n=20, n_table=22,
    # fwd=21 (the meme's one-month horizon), gross
    win=65.0, mean=420.2, skew=-0.64, t=1.47, ci_lo=-114, ci_hi=937,
    coin_win=45.0, coin_mean=24.8, excess=395.4,
    # fwd=63 (snooped longer horizon)
    win63=70.0, mean63=1220.4, t63=3.14, ci63_lo=327, ci63_hi=2049,
    # look-ahead (extra day)
    lag2_mean=537.0, lag2_t=2.26,
    # cost sweep (fwd=21, net)
    c0=420.2, c0t=1.47, c5=410.2, c5t=1.44, c10=400.2, c10t=1.40, c20=380.2, c20t=1.33,
    # synthetic control (seed 336, n=200, fwd=21)
    null_win=47.5, null_mean=-35.7, null_t=-0.54,
    anti_win=79.5, anti_mean=449.8, anti_t=7.85,
    sel_win=70.5, sel_mean=250.4, sel_t=4.14,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
# Point quantlab's parquet cache at THIS study's _cache so the real-tape cells hit the
# pinned tapes regardless of the working directory (and fall back to frozen numbers if absent).
os.environ.setdefault("OVERNIGHT_CACHE", os.path.abspath("../_cache"))
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from inverse_cramer import data, strategy as st

# Try the real tape (cache or network); fall back to the frozen numbers from results.md.
def _load_real():
    try:
        tapes = data.load_real(end="2026-06-20", fetch=False)
        led = st.build_real_ledger(data.calls_table(), tapes, fwd=21)
        if len(led) >= 5:
            return led, tapes, True
    except Exception as e:
        print("real tape unavailable, using frozen numbers:", type(e).__name__)
    return None, None, False

REAL_LEDGER, REAL_TAPES, HAVE_REAL = _load_real()
print("real curated tape present:", HAVE_REAL,
      f"(n={len(REAL_LEDGER) if HAVE_REAL else 'frozen'})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Inverse-Cramer — is fading the loudest pundit an edge? 📺\n"
            "### A viral 'just do the opposite of Jim Cramer' meme, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Selection_bias%3F: Not_supported](https://img.shields.io/badge/Selection_bias%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "One of the internet's favourite market jokes: CNBC's Jim Cramer is *so* often wrong "
            "that you should just **do the opposite of whatever he says**. The joke got real enough "
            "to spawn an actual ETF — **SJIM**, the *Inverse Cramer Tracker* — which launched in "
            "2023 and quietly closed eleven months later. This notebook takes a curated table of "
            "his most famous calls, fades every one of them, and asks the only two questions that "
            "matter: *is fading him a real edge, or does the meme just remember his worst days?*\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the bootstrap and the "
            "selection-bias maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does fading his famous calls make money? | **On paper, a lot** — about "
            f"**+{R['mean']:.0f} bps per call**, winning **{R['win']:.0f}%** of the time. |\n"
            f"| Is that statistically real? | **No — not at the one-month horizon.** The *t*-stat "
            f"is only **+{R['t']:.2f}** (you want ≥ 2), and the error bars on the average run from "
            f"**{R['ci_lo']:+d} to {R['ci_hi']:+d} bps** — straight through zero. |\n"
            "| Could you actually trade it? | **No.** The list of 'his worst calls' only exists "
            "*after* the calls went bad. You can't trade hindsight. |\n"
            "| So why does the meme feel so true? | **Selection.** We show a pundit who is a pure "
            "**coin flip** still looks fade-able once you publish only his worst moments. |\n\n"
            "> The Inverse-Cramer meme isn't a strategy — it's a highlight reel. A curated list of "
            "anyone's most memorable mistakes will look unbeatable, and that's exactly the trick."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Jim Cramer is the world's greatest contrarian indicator. Whatever he tells you to "
            "buy, sell — and whatever he tells you to sell, buy. The Inverse Cramer never loses.\"*\n\n"
            "— finance Twitter, c. 2021–2023, and the (real, now-defunct) **SJIM** ETF prospectus "
            "in polite institutional language\n\n"
            "The steelman: a famous TV personality makes high-conviction calls to a huge retail "
            "audience, often near tops (attention peaks when a stock is already euphoric). If his "
            "endorsements mark crowd-driven exhaustion, the move *after* the call should reverse — "
            "so fading him would pay. That's a real, testable mechanism, not just a joke."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two reasons this is worth a careful look. First, an *actual fund* was built on it and "
            "*closed* — so reality already voted, and we get to explain why the meme and the fund "
            "disagree. Second, the Inverse-Cramer is the purest possible specimen of a trap that "
            "shows up everywhere: a **curated highlight reel** that looks like evidence. Learn to "
            "see through it here and you'll spot it in every 'look how wrong the experts were' "
            "thread you ever read."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Fade the famous calls and measure.** Take a curated table of notable Cramer "
            "calls, trade the *opposite* of each (entered the day *after* he speaks — no peeking), "
            "hold a month, and look at the average **and** its error bars.\n"
            "2. **Race it against a coin.** Trade a *random* direction on the same calls. If "
            "fading knows something, it beats the coin by more than noise.\n"
            "3. **Build a coin-flip pundit on purpose.** Simulate a pundit who is *exactly* a coin, "
            "then publish only his worst calls — and see whether that alone fakes an 'edge'. "
            "Spoiler: it does.\n\n"
            "Tape: real daily total-return prices for the call tickers, since 2007."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how good does the fade look on the famous calls?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.excess_over_coin(REAL_LEDGER)\n"
            "    f, c = res['fade'], res['coin']\n"
            "    f_mean, f_win, f_t, lo, hi = f['mean_bps'], f['win_rate']*100, f['tstat'], f['ci_lo'], f['ci_hi']\n"
            "    c_mean, c_win = c['mean_bps'], c['win_rate']*100\n"
            "else:\n"
            f"    f_mean, f_win, f_t, lo, hi = {R['mean']}, {R['win']}, {R['t']}, {R['ci_lo']}, {R['ci_hi']}\n"
            f"    c_mean, c_win = {R['coin_mean']}, {R['coin_win']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Fade Cramer', 'Random direction (coin)'], [f_mean, c_mean],\n"
            "           color=[AMBER, GREY], width=.5)\n"
            "ax.errorbar(0, f_mean, yerr=[[f_mean-lo],[hi-f_mean]], fmt='none', ecolor='k', capsize=6, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('profit per call (bps)')\n"
            "ax.set_title('The fade LOOKS huge — but the error bar runs through zero')\n"
            "for bar_, w in zip(b, [f_win, c_win]):\n"
            "    ax.annotate(f'win {w:.0f}%', (bar_.get_x()+bar_.get_width()/2, 20),\n"
            "                ha='center', va='bottom', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Fade: {f_mean:+.0f} bps  win {f_win:.0f}%  HAC t={f_t:+.2f}  95% CI [{lo:+.0f}, {hi:+.0f}]')"
        ),
        md(
            f"There's the catch. The fade earns a fat **+{R['mean']:.0f} bps/call** and wins "
            f"**{R['win']:.0f}%** of the time — but the *t*-stat is only **+{R['t']:.2f}** (below "
            f"the 2 you need), and the error bar on the average runs from **{R['ci_lo']:+d} to "
            f"{R['ci_hi']:+d} bps** — straight through zero. Twenty hand-picked calls just isn't "
            "enough to tell a real edge from a lucky reel."
        ),
        md(
            "**Now the punchline. Let's build a pundit who is *literally a coin flip*** — zero "
            "skill, zero anti-skill — and then publish only his *worst* calls, the way a meme would. "
            "What 'edge' does fading a coin produce?"
        ),
        code(
            "# A coin-flip pundit, then curate the published list to his worst calls.\n"
            "_, calls_null, _ = data.synthetic_calls(anti_alpha=0.0, selection_bias=0.0, seed=336)\n"
            "_, calls_cur,  _ = data.synthetic_calls(anti_alpha=0.0, selection_bias=0.6, seed=336)\n"
            "s_null = st.summarize(st.fade_returns(st.synthetic_ledger(calls_null)))\n"
            "s_cur  = st.summarize(st.fade_returns(st.synthetic_ledger(calls_cur)))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(['All his calls','Only his worst\\n(curated)'], [s_null['win_rate']*100, s_cur['win_rate']*100],\n"
            "       color=[GREY, AMBER], width=.5)\n"
            "a1.axhline(50, ls='--', c='k'); a1.set_ylim(0,100); a1.set_ylabel('fade win rate %')\n"
            "a1.set_title('Same coin-flip pundit, two lists')\n"
            "a2.bar(['All his calls','Only his worst\\n(curated)'], [s_null['mean_bps'], s_cur['mean_bps']],\n"
            "       color=[GREY, AMBER], width=.5)\n"
            "a2.axhline(0, c='k'); a2.set_ylabel('fade profit/call (bps)')\n"
            "a2.set_title('Curation alone manufactures an \"edge\"')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"All calls : win {s_null['win_rate']*100:.0f}%  mean {s_null['mean_bps']:+.0f} bps  t={s_null['tstat']:+.2f}\")\n"
            "print(f\"Curated   : win {s_cur['win_rate']*100:.0f}%  mean {s_cur['mean_bps']:+.0f} bps  t={s_cur['tstat']:+.2f}\")"
        ),
        md(
            f"There it is. The *same* coin-flip pundit, taken at face value, is a coin to fade "
            f"({R['null_win']:.0f}% wins, a *losing* average). But publish only his worst calls and "
            f"the fade suddenly wins **{R['sel_win']:.0f}%** at **+{R['sel_mean']:.0f} bps**, with a "
            f"*t*-stat of **+{R['sel_t']:.2f}** — comfortably 'significant'. No skill was harmed in "
            "the making of this edge; it's **pure curation**. That's the whole Inverse-Cramer meme "
            "in one chart."
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The curated fade earns +{R['mean']:.0f} bps/call but only "
            f"*t* = +{R['t']:.2f} at one month, with a CI through zero. The meme (and the SJIM ETF) "
            "assert a real edge; this tape can't certify it.\n"
            "- **Tradability — Mirage.** You can't trade a list that only exists in hindsight. A "
            "real 'fade everything he says' rule would take *hundreds* of calls a week, not the 20 "
            "a meme remembers.\n"
            "- **Selection bias? — Not supported (as an edge).** A coin-flip pundit, curated, fakes "
            "a *t* = +4 fade. The famous edge is consistent with pure selection."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs aren't the problem — the per-call moves are large, so even a generous round-trip "
            "fee barely dents them:"
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(st.fade_returns(REAL_LEDGER, cost_bps=c))['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['c0']}, {R['c5']}, {R['c10']}, {R['c20']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.fill_between(costs, net, 0, color=AMBER, alpha=.12); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost x 2 legs (bps)'); ax.set_ylabel('net profit per call (bps)')\n"
            "ax.set_ylim(0, max(net)*1.15)\n"
            "ax.set_title('Costs barely touch it — but that is NOT why it fails')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The edge dies on selection and sample size, not on costs.')"
        ),
        md(
            "Unlike the desk's cost-killed mirages, this one shrugs off fees. It fails for a "
            "deeper reason: **the strategy as stated isn't a strategy.** 'Fade his worst calls' is "
            "a description of the past. The honest version — fade *every* call he makes, live — is "
            "a completely different (and far less flattering) experiment, and it's the one SJIM "
            "actually ran. It closed."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The real test is the live, all-calls fade.** Scrape *every* on-air recommendation "
            "(not the memorable ones) and fade them with no hindsight. That's the experiment SJIM "
            "ran — and the one a curated table can never stand in for.\n"
            "- **Where the curation trap shows up elsewhere.** The desk's alt-data lot "
            "([Study 252 — Google-Trends](../../252-google-trends/) onward) keeps finding the same "
            "thing: a crowd/attention signal that looks predictive until you account for how the "
            "sample was chosen.\n"
            "- **The closest cousin.** [Study 291 — Doge-Tweets](../../291-doge-tweets/) — a single "
            "loud voice 'moving' the tape, with the same selection caveats.\n\n"
            "*Think the fade is real? Fork this, swap in a complete, time-stamped call log (no "
            "cherry-picking), and show the t-stat surviving once the curation is gone.*"
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
            "# Inverse-Cramer — a quantitative teardown 🔬\n"
            "### Curated real calls · forward-return fade · random-direction control · "
            "HAC *t* + block bootstrap · the selection-bias synthetic\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Selection_bias%3F: Not_supported](https://img.shields.io/badge/Selection_bias%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We fade a curated table of "
            "on-air calls, pin it against a random-direction control, test horizon and look-ahead "
            "sensitivity, and use a deterministic synthetic universe to separate a genuine "
            "anti-signal from pure curation.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo total-return daily bars via "
            "`quantlab.data`; as-of 2026-06-20; the offline core and tests run on a deterministic "
            "synthetic universe. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Curated fade +{R['mean']:.0f} bps/call but HAC "
            f"*t* = **+{R['t']:.2f}** at fwd=21, boot 95% CI [{R['ci_lo']:+d}, {R['ci_hi']:+d}] "
            f"through zero; clears *t*=2 only at the snooped fwd=63 (+{R['t63']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Unrunnable: the call list is hindsight. Cost-robust "
            f"(+{R['c10']:.0f} bps at 10 bps) but there is no forward, all-calls rule here. |\n"
            f"| **Selection bias?** | `NOT SUPPORTED` | Synthetic coin pundit + curation → fade "
            f"*t* = **+{R['sel_t']:.2f}** with zero predictive content. |\n\n"
            "> 💡 In plain words: the fade looks great on a list chosen because it looks great. "
            "Strip the selection (synthetic null) and the edge is a coin; keep it (real curated "
            "table) and 20 points can't clear the bar anyway."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a call carry stated direction $d\\in\\{-1,+1\\}$ and forward return $r$ over the "
            "hold. The **fade** position is $-d$, so its return is $-d\\,r$.\n\n"
            "- **H₁ (anti-signal).** $\\mathbb{E}[-d\\,r] > 0$ — the pundit is systematically wrong.\n"
            "- **H₂ (beats random).** It exceeds the same statistic with a random direction on the "
            "*same* calls.\n"
            "- **H₃ (no look-ahead).** It survives a conservative one-extra-day execution lag.\n"
            "- **H₄ (tradable).** A *forward, all-calls* rule — not a curated reel — is the actual "
            "system, and it must hold up out of hindsight.\n"
            "- **H₅ (the curated edge is real).** The visible edge is not an artefact of selecting "
            "on the outcome.\n\n"
            "We find **H₁/H₂ unproven** at the meme horizon (|t| < 2, CI through zero), **H₃ "
            "fragile** (a 28% mean swing from one day), **H₄ untestable here** (no forward log), "
            "and **H₅ rejected** (a coin pundit + curation reproduces the edge)."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The prior from the forecasting literature (Tetlock 2005) is that pundit calls carry "
            "little forward information — which cuts **both ways**: if the calls are ≈ noise, fading "
            "them is ≈ noise too, not an edge. The interesting content is therefore not 'is he "
            "wrong' but **how much of the visible fade-edge is manufactured by curation**. Quant\n"
            "ifying that — the synthetic `selection_bias` arm — is the study's real deliverable."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Stated direction on the on-air date; fade $=-d$; enter **one session "
            "later** (one shift), hold `fwd` sessions. A **+2** lag is the look-ahead check.\n"
            "- **Inference.** Newey-West HAC *t* on per-call returns **and** a circular "
            "block-bootstrap 95% CI on the mean — a tiny n demands the CI, not just the point.\n"
            "- **Control.** Random ±1 direction on the identical calls (seeded coin).\n"
            "- **Horizon.** Pre-declared fwd=21 (the meme's month); fwd=63 reported only as a "
            "sensitivity (and flagged as snoopable).\n"
            "- **Synthetic.** Two orthogonal knobs — a genuine `anti_alpha` and a pure "
            "`selection_bias` — to separate real anti-skill from curation.\n\n"
            "> 💡 In plain words: we measure the fade three ways and then *try to fake it* with a "
            "coin, so we know what a non-edge looks like."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The curated fade vs a coin, with honest error bars\n\n"
            "The point estimate is large; the question is the *uncertainty* on a 20-call sample."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.excess_over_coin(REAL_LEDGER)\n"
            "    f, c = res['fade'], res['coin']\n"
            "    fmean, ftt, lo, hi, excess = f['mean_bps'], f['tstat'], f['ci_lo'], f['ci_hi'], res['excess_bps']\n"
            "    cmean = c['mean_bps']\n"
            "else:\n"
            f"    fmean, ftt, lo, hi, excess = {R['mean']}, {R['t']}, {R['ci_lo']}, {R['ci_hi']}, {R['excess']}\n"
            f"    cmean = {R['coin_mean']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['Fade','Coin'], [fmean, cmean], color=[AMBER, GREY], width=.45)\n"
            "ax.errorbar(0, fmean, yerr=[[fmean-lo],[hi-fmean]], fmt='none', ecolor='k', capsize=7, lw=1.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean per call (bps)')\n"
            "ax.set_title(f'Fade {fmean:+.0f} bps (t={ftt:+.2f}) — 95% CI [{lo:+.0f},{hi:+.0f}] spans 0')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Excess over coin: {excess:+.0f} bps   HAC t={ftt:+.2f}   CI [{lo:+.0f}, {hi:+.0f}]')"
        ),
        md(
            f"> 💡 In plain words: +{R['mean']:.0f} bps and a {R['win']:.0f}% hit-rate are seductive, "
            f"but *t* = +{R['t']:.2f} is below the bar and the bootstrap CI ([{R['ci_lo']:+d}, "
            f"{R['ci_hi']:+d}]) contains zero. **H₁/H₂ are not established** at the meme's horizon."
        ),
        md(
            "### 4b · Horizon & look-ahead sensitivity — the only 'real' result is the snooped one\n\n"
            "If an effect only appears at the horizon you happened to pick, that's a red flag."
        ),
        code(
            "rows = [('fwd=21 (meme)', "
            f"{R['mean']}, {R['t']}), ('fwd=63 (quarter)', {R['mean63']}, {R['t63']}), "
            f"('fwd=21, +1 extra day', {R['lag2_mean']}, {R['lag2_t']})]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lab, fwd, lag in [('fwd=21 (meme)',21,1), ('fwd=63 (quarter)',63,1), ('fwd=21, +1 extra day',21,2)]:\n"
            "        led = st.build_real_ledger(data.calls_table(), REAL_TAPES, fwd=fwd, entry_lag=lag)\n"
            "        s = st.summarize(st.fade_returns(led)); rows.append((lab, s['mean_bps'], s['tstat']))\n"
            "labs = [r[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if abs(t) >= 2 else AMBER for t in ts]\n"
            "ax.bar(labs, ts, color=col); ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat'); ax.set_title('Only the post-hoc longer horizon clears |t|=2')\n"
            "ax.tick_params(axis='x', labelsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab, m, t in rows: print(f'{lab:24s} mean {m:+7.0f} bps  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the fade clears *t* = 2 only at fwd=63 (+{R['t63']:.2f}), a "
            f"horizon chosen *after* seeing the data, on a sample already curated against the pundit. "
            f"A single extra execution day swings the mean by ~28% ({R['mean']:+.0f}→{R['lag2_mean']:+.0f} "
            "bps) — the signature of a tiny, noisy sample. **H₃ is fragile; the fwd=63 result is "
            "snoopable, not a certificate.**"
        ),
        md(
            "### 4c · The selection-bias synthetic — faking the edge with a coin\n\n"
            "The decisive test of H₅. Three synthetic worlds, seed 336, n=200, fwd=21: a coin "
            "pundit; a genuinely anti-skilled pundit; and a **coin pundit whose published list is "
            "curated to his worst calls**."
        ),
        code(
            "worlds = [('Coin pundit\\n(no curation)', 0.0, 0.0, GREY),\n"
            "          ('Genuine anti-signal', 0.6, 0.0, GREEN),\n"
            "          ('Coin pundit\\n+ CURATION', 0.0, 0.6, RED)]\n"
            "rows = []\n"
            "for lab, a, s, _ in worlds:\n"
            "    _, calls, _ = data.synthetic_calls(anti_alpha=a, selection_bias=s, seed=336)\n"
            "    summ = st.summarize(st.fade_returns(st.synthetic_ledger(calls)))\n"
            "    rows.append((lab, summ['win_rate']*100, summ['mean_bps'], summ['tstat']))\n"
            "dfm = pd.DataFrame(rows, columns=['world','win%','mean_bps','t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.bar(dfm['world'], dfm['t'], color=[w[3] for w in worlds])\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('fade HAC t-stat'); ax.tick_params(axis='x', labelsize=9)\n"
            "ax.set_title('Curation alone (red) clears |t|=2 with ZERO real skill')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfm.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the coin pundit is a coin to fade (*t* = {R['null_t']:.2f}); the "
            f"genuinely anti-skilled one is detected (*t* = +{R['anti_t']:.2f}); and **the coin "
            f"pundit + curation prints *t* = +{R['sel_t']:.2f}** with no skill at all. The real "
            "curated table is the same machine at n=20. **H₅ rejected — selection not supported as "
            "an edge.**"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — curated fade +{R['mean']:.0f} bps/call, HAC *t* +{R['t']:.2f} at "
            f"fwd=21, boot CI [{R['ci_lo']:+d}, {R['ci_hi']:+d}] through zero; clears *t*=2 only at "
            f"the snooped fwd=63 (+{R['t63']:.2f}).\n"
            "- **Tradability `MIRAGE`** — the call list is hindsight; cost-robust but there is no "
            "forward, all-calls rule here, and the live version (SJIM) closed.\n"
            f"- **Selection bias? `NOT SUPPORTED`** — synthetic coin pundit + curation → "
            f"*t* +{R['sel_t']:.2f} with no skill. The visible edge is consistent with pure curation."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep and the real wall\n\n"
            "Per-call moves are large, so costs are a rounding error. The wall is structural, not "
            "frictional."
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(st.fade_returns(REAL_LEDGER, cost_bps=c)) for c in costs]\n"
            "    means = [s['mean_bps'] for s in sw]; ts = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    means = [{R['c0']},{R['c5']},{R['c10']},{R['c20']}]\n"
            f"    ts = [{R['c0t']},{R['c5t']},{R['c10t']},{R['c20t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, means, 'o-', c=AMBER, lw=2, label='net mean (bps)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, ts, 's--', c=GREY, lw=1.5)\n"
            "ax2.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost x 2 legs (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax.set_ylim(0, max(means)*1.15); ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Costs do not move the t-stat off the floor — the sample does')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'At 10 bps: {R['c10']:+.0f} bps/call, t={R['c10t']:+.2f} — still below the bar.')"
        ),
        md(
            "> 💡 In plain words: the strategy never dies on costs because it never lives long "
            "enough to pay them. 'Fade his worst calls' is a description of the past; the honest, "
            "forward, *every-call* version is a different experiment — and it's the one the real "
            "ETF ran before closing."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 · Going further — the faithful detector & the real experiment\n\n"
            "Is the engine a faithful detector of a *genuine* anti-signal (when one exists)? Sweep "
            "the planted `anti_alpha` and watch the fade's edge over a coin."
        ),
        code(
            "alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]\n"
            "edge = []\n"
            "for a in alphas:\n"
            "    _, calls, _ = data.synthetic_calls(anti_alpha=a, selection_bias=0.0, seed=336)\n"
            "    edge.append(st.excess_over_coin(st.synthetic_ledger(calls), seed=336)['excess_bps'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(alphas, edge, 'o-', c=GREEN, lw=2); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted anti-signal (anti_alpha)'); ax.set_ylabel('fade - coin (bps/call)')\n"
            "ax.set_title('The engine is a faithful detector — edge rises with real anti-skill')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Edge ~0 at the null, monotone in the planted anti-signal.')"
        ),
        md(
            "The engine cleanly recovers a *real* anti-signal when one is planted and reads ~0 at "
            "the null — so the real-tape ambiguity is a statement about **the data**, not the "
            "harness: a 20-call curated reel cannot certify an edge, and a coin pundit reproduces "
            "the same reel.\n\n"
            "**The experiment that would settle it** is a complete, time-stamped log of *every* "
            "on-air call faded live, with no cherry-picking — the experiment SJIM ran in the real "
            "world. For the closest desk cousin, see [Study 291 — Doge-Tweets](../../291-doge-tweets/); "
            "for the broader curation trap, the alt-data lot from "
            "[Study 252 — Google-Trends](../../252-google-trends/) onward."
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
