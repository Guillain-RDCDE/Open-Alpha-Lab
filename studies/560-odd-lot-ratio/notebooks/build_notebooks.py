"""Generate the two narrative notebooks for Study 560 (Odd-Lot-Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Everything runs offline
and deterministic — this is a **synthetic-only** study (no free real odd-lot ratio series exists),
so there is no real-tape banner: the ``fade_alpha = 0`` tape is the honest 'modern null' headline
and the ``fade_alpha > 0`` tape is the positive control. The dict ``R`` mirrors docs/results.md.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; modern null fade_alpha=0;
# 1,039-week synthetic tape 2005-01-14 -> 2024-12-06; tape fp 2659c7bf0fd0).
R = dict(
    fp_tape="2659c7bf0fd0", n=1039, split="2005-01-14", end="2024-12-06",
    slope=0.032, slope_t=0.50, corr=0.015, placebo_p=0.62,
    panic_ann=8.1, euphoria_ann=9.7, regime_spread=-1.6, regime_t=-0.20,
    gross_ann=-2.0, net_ann=-2.7, gross_sharpe=-0.11, net_sharpe=-0.15, turnover=25.9,
    cost_bps=2.0, borrow_bps=50.0,
    # robustness windows: (label, slope%/z, HAC t)
    win=[("2005-01 -> 2009-12", -0.105, -0.90), ("2010-01 -> 2014-12", 0.101, 0.87),
         ("2014-12 -> 2019-12", -0.151, -1.12), ("2019-12 -> 2024-12", 0.247, 2.20)],
    # synthetic control: (fade_alpha, mean HAC t over 25 seeds)
    ctrl=[(0.0000, 0.19), (0.0005, -0.57), (0.0010, -1.33), (0.0015, -2.10),
          (0.0020, -2.86), (0.0030, -4.39)],
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

from odd_lot_ratio import data, strategy as st

# SYNTHETIC-ONLY study: the modern post-decimalization null (fade_alpha = 0) is the headline tape.
# No real odd-lot ratio series is available on a free retail stack -> no real-tape banner anywhere.
NULL, _ = data.synthetic_series(fade_alpha=0.0)          # modern null (the honest default)
EDGE, _ = data.synthetic_series(fade_alpha=0.0020)       # the OLD dumb-money world (control)
print("synthetic modern-null tape:", len(NULL), "weeks, fp", data.fingerprint(NULL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Odd-Lot Ratio — do small retail trades still mark the wrong side? 🪙\n"
            "### The oldest 'dumb-money' fade signal, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Alive_post-decimalization%3F: Busted](https://img.shields.io/badge/Alive_post--decimalization%3F-Busted-8b949e?style=flat-square)\n\n"
            "Back when the stock market ran on paper tickets, a trade of **fewer than 100 shares** was "
            "called an *odd lot* — and it was the fingerprint of the little guy, the small investor who "
            "couldn't afford a round 100-share lot. Wall Street lore said the odd-lot crowd was "
            "*reliably wrong*: they bought at the top and sold at the bottom. So the **odd-lot theory** "
            "(Garfield Drew, 1950s) said: watch what the odd-lotters do, and do the **opposite**. When "
            "they're buying like crazy, sell. When they're panic-selling, buy. **Fade the dumb money.**\n\n"
            "Did it ever work? Maybe, once. Does it work now? We test the fade cleanly — and find it's "
            "**dead**, for a very concrete reason: today most 'odd lots' aren't retail at all.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool. **Synthetic-only**: a clean, "
            "long odd-lot ratio series is proprietary, so there is no real tape to fetch — the honest "
            "default is a 'modern' world where the fade is switched off. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does fading odd-lot buying pay today? | **No.** On the modern tape the 'panic' weeks "
            f"earned **+{R['panic_ann']}%/yr** and the 'euphoria' weeks **+{R['euphoria_ann']}%/yr** — "
            "the crowd's euphoria was, if anything, *slightly better*, the opposite of the fade. |\n"
            f"| Is that just noise? | **It's all noise** — a shuffle test puts the odds at *p* = "
            f"{R['placebo_p']}, and the signal's direction flips from window to window. |\n"
            f"| Could you trade it? | **No.** A weekly contrarian overlay *loses* "
            f"**{R['net_ann']}%/yr** after costs, trading itself over 25× a year. |\n"
            "| So why is it dead? | **Because 'odd lot' no longer means 'retail'.** Since "
            "decimalization (2001), computers chop one big order into thousands of tiny ones — most "
            "odd lots today are *machines*, not mom-and-pop. |\n\n"
            "> The fade was real folklore when odd lots really were the little guy. The premise died "
            "with penny ticks and algorithmic order-slicing — and the odd-lot data itself was "
            "proprietary and off the public tape for years."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The odd-lot crowd is chronically wrong — so fade them.\"* — the odd-lot theory "
            "(Garfield Drew, 1950s).\n\n"
            "The idea: small investors are emotional. They chase rallies (buying tops) and flee "
            "crashes (selling bottoms). If you could *see* what they were doing — via the odd-lot "
            "ratio, the share of small-lot volume that was buying vs selling — you'd have a contrarian "
            "crystal ball: heavy odd-lot buying = a top is near = sell."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's free money from other people's mistakes — the purest form of a sentiment "
            "edge. It's also a close cousin of the desk's survey-sentiment study "
            "([257 AAII-Sentiment](../../257-aaii-sentiment/)), which asks the *same* contrarian "
            "question of a bull-bear survey. The difference: AAII measures *stated opinion*; the "
            "odd-lot ratio measures *actual order flow*. Flow should be harder to fake — if the theory "
            "holds anywhere, it should hold here."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Watch the odd-lot ratio** each week — the share of small-lot volume that's buying "
            "(high = euphoric, low = panicky).\n"
            "2. **Fade it.** When last week's ratio was high, go short/underweight this week; when it "
            "was low, go long. (We use *last* week's ratio, so there's no peeking — one week's lag.)\n"
            "3. **Check who won.** Did the 'panic' weeks (fade said buy) beat the 'euphoria' weeks "
            "(fade said sell)? The theory says yes.\n\n"
            "Catch: **there is no free odd-lot ratio series.** It lives in proprietary market data, and "
            "odd lots weren't even on the public tape until 2013–14. So we build a *synthetic* tape "
            "with a knob — turn the fade edge **on** (the old world) or **off** (today) — and see what "
            "a clean test does with each."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**On the modern tape (fade switched off), did panic beat euphoria?**"
        ),
        code(
            "rs = st.regime_sort(NULL)\n"
            "panic, euph = rs['panic_ann']*100, rs['euphoria_ann']*100\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['PANIC weeks\\n(fade says BUY)','EUPHORIA weeks\\n(fade says SELL)'], [panic, euph],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('next-week return, annualised %')\n"
            "ax.set_title(f'The fade says PANIC wins. Here they roughly tie ({panic:.1f}% vs {euph:.1f}%)')\n"
            "fig.suptitle('synthetic modern-null tape (fade_alpha = 0)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'panic +{panic:.1f}%/yr  vs  euphoria +{euph:.1f}%/yr  ->  spread {rs[\"spread_ann\"]*100:+.1f}%/yr (t {rs[\"t\"]:+.2f})')"
        ),
        md(
            f"There's the tell: the fade predicts the **panic** weeks win. On the modern tape they "
            f"**tie** (+{R['panic_ann']}% vs +{R['euphoria_ann']}%) — a spread of {R['regime_spread']}%/yr "
            f"with a *t* of {R['regime_t']}, i.e. nothing. The odd-lot crowd's 'euphoria' weeks weren't "
            "worse; small trades no longer mark the wrong side."
        ),
        md(
            "**Is the signal just *gone*, or does its direction wobble?** Run the fade over four "
            "slices of the tape:"
        ),
        code(
            "rows = st.window_sweep(NULL, n_windows=4)\n"
            "labs = [w['window'][:7] + '..' + w['window'][-7:] for w in rows]\n"
            "ts = [w['t'] for w in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if t < 0 else RED for t in ts]  # negative t = fade direction (correct)\n"
            "ax.bar(range(len(ts)), ts, color=cols, width=.55)\n"
            "ax.set_xticks(range(len(ts))); ax.set_xticklabels(labs, rotation=15, fontsize=8)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('fade slope HAC t (negative = fade works)')\n"
            "ax.set_title('The direction flips window to window — and never clears -2 the right way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t by window:', [round(t,2) for t in ts])"
        ),
        md(
            "The direction **flips**: some windows lean faintly the fade way (negative), others the "
            "*anti*-fade way (positive) — and the only window that reaches the |*t*|=2 significance bar "
            "does so with the **wrong** sign. A signal whose direction depends on the calendar isn't a "
            "signal."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The fade is statistically zero (*t* +{R['slope_t']}, placebo "
            f"*p* {R['placebo_p']}), the panic/euphoria spread is a tie, and the direction flips across "
            "windows. Synthetic-only, so it can't rate above `WEAK`/`NONE` anyway.\n"
            "- **Tradability — Mirage.** The overlay loses money and churns 26× a year.\n"
            "- **Alive post-decimalization? — Busted.** The premise (odd lot = dumb retail money) died "
            "with penny ticks and algorithmic order-slicing."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — twice over. First, there's nothing to trade: the fade is the wrong sign and loses "
            "money. Second, even if it worked, **you couldn't get the data**: a clean, timely odd-lot "
            "ratio is proprietary market-data plumbing, not something on a free retail feed. The "
            "signal the theory rests on isn't just dead — for most people it was never visible."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The engine isn't broken — the world changed.** In "
            "[02_for_the_quants](02_for_the_quants.ipynb) we *plant* the old dumb-money fade and watch "
            "a clean test bank it cleanly (the slope-*t* sails past −2). Switch the fade off (today) "
            "and the test correctly reads zero.\n"
            "- **The survey cousin.** [257 AAII-Sentiment](../../257-aaii-sentiment/) runs the same "
            "contrarian idea on a bull-bear *survey* — directionally right but weak.\n\n"
            "*Think the odd-lot fade is alive on real, point-in-time tape data? Fork this, wire in a "
            "genuine consolidated-tape odd-lot ratio, and show the fade slope clearing t = −2 the "
            "right way after decimalization.*"
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
            "# The Odd-Lot Ratio — a quantitative teardown 🔬\n"
            "### contrarian fade signal · HAC slope · label-shuffle placebo · panic/euphoria sort · four-window sign-flip · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Alive_post-decimalization%3F: Busted](https://img.shields.io/badge/Alive_post--decimalization%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the textbook odd-lot "
            "fade (position = −z of the prior-week odd-lot ratio) and test whether fading the "
            "'dumb-money' crowd pays.\n\n"
            "> ⚠️ **Not investment advice. Synthetic-only study.** A clean, long, point-in-time odd-lot "
            "ratio series is proprietary (TAQ/consolidated-tape flags; odd lots weren't on the SIP "
            f"until 2013–14), so the real free data does not exist and this caps at `WEAK`/`NONE`. The "
            f"headline is the deterministic **modern-null** tape (`fade_alpha = 0`, fp `{R['fp_tape']}`, "
            "as-of 2026-06-30); the positive control plants the fade. Methods in "
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
            f"| **Signal** | `NONE` | Fade slope **+{R['slope']}%/z-unit** (HAC *t* **+{R['slope_t']}**, "
            f"placebo *p* {R['placebo_p']}) — *wrong* sign, not significant; regime spread "
            f"**{R['regime_spread']}%/yr** (*t* {R['regime_t']}); sign flips across windows. "
            "**Synthetic-only** → capped. |\n"
            f"| **Tradability** | `MIRAGE` | Weekly overlay gross **{R['gross_ann']}%/yr** → net "
            f"**{R['net_ann']}%/yr** (Sharpe {R['gross_sharpe']} → {R['net_sharpe']}) at "
            f"**{R['turnover']}×** turnover. |\n"
            "| **Alive post-decimalization?** | `BUSTED` | Order-slicing turned odd lots into machine "
            "child-orders; the retail premise is gone. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on the modern "
            "null tape the odd-lot fade is absent, sign-unstable, and loses money — and no real tape "
            "exists to lift it above `WEAK`."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $O_{t-1}$ be the standardised odd-lot ratio observed at week $t-1$'s close and $r_t$ "
            "the index's week-$t$ forward return.\n\n"
            "- **H₁ (the fade).** $\\text{slope of } r_t \\text{ on } O_{t-1} < 0$ at HAC *t* ≤ −2 "
            "(more odd-lot buying → weaker next week).\n"
            "- **H₂ (regime).** $\\mathbb{E}[r \\mid \\text{panic}] - \\mathbb{E}[r \\mid \\text{euphoria}] "
            "> 0$ (fading works: panic weeks out-earn euphoria weeks).\n"
            "- **H₃ (robustness).** The sign of H₁ is stable across scoring windows.\n"
            "- **H₄ (net).** The fade overlay beats zero after costs and the short-leg borrow.\n\n"
            "On the modern null tape we find **H₁ rejected** (slope *positive*, *t* ≈ +0.5), **H₂ "
            "rejected** (spread negative, *t* ≈ −0.2), **H₃ rejected** (sign flips), **H₄ rejected** "
            "(overlay loses)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Two structural breaks retired this once-real signal: (a) "
            "**decimalization (2001)** and algorithmic order-slicing, which shred one institutional "
            "parent order into thousands of *odd-lot child orders* — so an odd lot is no longer 'dumb "
            "retail money'; and (b) **tape reporting** — odd lots were excluded from the consolidated "
            "SIP tape until 2013–14 (O'Hara, Yao & Ye 2014 show that missing odd-lot flow was often "
            "*informed*). The `fade_alpha = 0` tape *is* the post-break world. This is the flow cousin "
            "of the survey study [257 AAII-Sentiment](../../257-aaii-sentiment/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\text{position}_t = -z(O_{t-1})$ — the textbook fade (short the euphoric "
            "crowd, long the panicked one), using only the *prior*-week ratio.\n"
            "- **Headline stat.** HAC (Newey–West, 4 weekly lags) slope of $r_t$ on $z(O_{t-1})$. The "
            "ratio is autocorrelated, so a naive OLS *t* overstates significance — the HAC *t* is the "
            "house bar.\n"
            "- **Regime sort.** Prior-ratio terciles: panic (lowest) vs euphoria (highest) next-week "
            "returns, two-sample *t*.\n"
            "- **Placebo.** Label-shuffle null (2000 perms) on the |slope|.\n"
            "- **Frictions.** One-way cost per weekly rebalance on the NAV traded + an annual borrow on "
            "the short-euphoria leg. Gross AND net.\n"
            "- **Positive control.** A deterministic synthetic world with a planted fade "
            "(`fade_alpha > 0`) and the null — averaged over 25 seeds (house rule).\n\n"
            "Execution lag: the ratio is observed at week $t-1$'s close (public); the position is "
            "entered there and held over week $t$. No future data enters the signal — one documented "
            "lag."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The fade slope — H₁\n\n"
            "HAC slope of next-week return on the standardised prior-week odd-lot ratio, with the "
            "label-shuffle placebo. A *negative* slope at *t* ≤ −2 is the fade."
        ),
        code(
            "fs = st.fade_slope(NULL); p = st.placebo_pvalue(NULL, n_perm=2000, seed=560)\n"
            "d = NULL.dropna(subset=['z_ratio_prior','forward_ret'])\n"
            "x = d['z_ratio_prior'].to_numpy(); y = d['forward_ret'].to_numpy()*100\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.6))\n"
            "ax.scatter(x, y, s=10, color=GREY, alpha=.5)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (fs['slope']*xs + y.mean()/100 - fs['slope']*x.mean())*100, c=RED, lw=2)\n"
            "ax.set_xlabel('standardised prior-week odd-lot ratio (higher = more odd-lot BUYING)')\n"
            "ax.set_ylabel('forward (next-week) return %')\n"
            "ax.set_title(f'Fade slope {fs[\"slope\"]*100:+.3f}%/z (HAC t {fs[\"t\"]:+.2f}, placebo p {p:.2f}) — NOT the fade')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {fs[\"slope\"]*100:+.3f}%/z | HAC t {fs[\"t\"]:+.2f} | corr {fs[\"corr\"]:+.3f} | placebo p {p:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** The slope is a tiny **+{R['slope']}%/z-unit** with "
            f"HAC *t* **+{R['slope_t']}** — the *opposite* sign to the fade and statistically zero — and "
            f"the placebo *p* = {R['placebo_p']} confirms even that is noise. Fading odd-lot buying does "
            "not pay on the modern tape."
        ),
        md(
            "### 4b · The regime sort — H₂ (panic vs euphoria)\n\n"
            "Next-week return of the panic tail (lowest prior ratio) vs the euphoria tail (highest)."
        ),
        code(
            "rs = st.regime_sort(NULL)\n"
            "panic, euph = rs['panic_ann']*100, rs['euphoria_ann']*100\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['PANIC (low ratio)','EUPHORIA (high ratio)'], [panic, euph], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('next-week return, annualised %')\n"
            "ax.set_title(f'panic - euphoria = {rs[\"spread_ann\"]*100:+.1f}%/yr (two-sample t {rs[\"t\"]:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'panic +{panic:.1f}%/yr | euphoria +{euph:.1f}%/yr | spread {rs[\"spread_ann\"]*100:+.1f}%/yr | t {rs[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected** — panic (+{R['panic_ann']}%/yr) *under*-earned "
            f"euphoria (+{R['euphoria_ann']}%/yr), spread {R['regime_spread']}%/yr, *t* {R['regime_t']}. "
            "The fade predicts the reverse; here it's a tie leaning the wrong way."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold?)\n\n"
            "The fade slope over four equal windows of the tape."
        ),
        code(
            "rows = st.window_sweep(NULL, n_windows=4)\n"
            "tab = pd.DataFrame([(w['window'], w['slope']*100, w['t'], w['n']) for w in rows],\n"
            "                   columns=['window','slope%/z','HAC_t','n']).set_index('window')\n"
            "ts = tab['HAC_t'].to_numpy()\n"
            "fig, ax = plt.subplots(figsize=(8.7, 4.3))\n"
            "cols = [GREEN if t < 0 else RED for t in ts]  # negative t = fade direction\n"
            "ax.bar(range(len(ts)), ts, color=cols, width=.55)\n"
            "ax.set_xticks(range(len(ts))); ax.set_xticklabels([w[:7] for w in tab.index], rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 (fade bar)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = +2 (anti-fade)')\n"
            "ax.set_ylabel('fade slope HAC t'); ax.set_title('Sign flips across windows; only crosses |2| the WRONG way')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** The HAC *t* runs **−0.90 / +0.87 / −1.12 / +2.20** "
            "across the four windows — the direction flips, and the only window that clears the "
            "significance bar does so with the *anti*-fade sign. That is the fingerprint of noise, not "
            "a decayed-but-real edge."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (slope +{R['slope']}%/z, *t* +{R['slope_t']}, wrong "
            f"sign, placebo *p* {R['placebo_p']}); H₂ rejected (spread {R['regime_spread']}%/yr); H₃ "
            "rejected (sign flips). Synthetic-only caps it here regardless.\n"
            f"- **Tradability `MIRAGE`** — overlay gross {R['gross_ann']}%/yr → net {R['net_ann']}%/yr, "
            f"Sharpe {R['net_sharpe']}, {R['turnover']}× turnover.\n"
            "- **Alive post-decimalization? `BUSTED`** — the retail premise died with order-slicing."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "A weekly contrarian overlay, scaled to unit gross exposure; one-way cost per rebalance + a "
            "borrow on the short-euphoria leg. Gross AND net."
        ),
        code(
            "ov = st.overlay_performance(NULL, cost_bps=2.0, borrow_ann_bps=50.0)\n"
            "g, n = ov['gross_ann']*100, ov['net_ann']*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [g, n], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('overlay return, annualised %')\n"
            "ax.set_title(f'Loses before costs ({g:+.1f}%/yr), worse after ({n:+.1f}%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}%/yr (Sharpe {ov[\"gross_sharpe\"]:+.2f}) -> net {n:+.2f}%/yr (Sharpe {ov[\"net_sharpe\"]:+.2f}); turnover {ov[\"ann_turnover\"]:.1f}x/yr')"
        ),
        md(
            "> 💡 In plain words: there's nothing to charge costs against — the fade overlay *loses* "
            f"({R['gross_ann']}%/yr gross) before frictions and churns {R['turnover']}× a year. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real 'dumb-money' "
            "fade (`fade_alpha > 0`) of increasing strength and watch the fade-slope HAC *t* go "
            "negative — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [a for a,_ in " + repr(R["ctrl"]) + "]\n"
            "ts = [st.synthetic_mean_t(data, fade_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.7, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (fade works)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted fade_alpha (larger = stronger dumb-money fade)')\n"
            "ax.set_ylabel('mean fade-slope HAC t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted fade — flat at the modern null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'fade_alpha {a:+.4f} -> mean HAC t {t:+.2f}')"
        ),
        md(
            "The HAC *t* is ≈ 0 at the modern null (`fade_alpha = 0`) and falls past −2 as the planted "
            "fade grows — so the engine is a faithful detector. The null result is therefore a "
            "statement about **the post-decimalization world**, where the fade is switched off, not a "
            "broken pipeline. And because the real odd-lot series is proprietary and unmeasurable on a "
            "free stack, this synthetic-only study is capped at `WEAK`/`NONE` by house rule. For the "
            "survey-sentiment cousin, see [257 AAII-Sentiment](../../257-aaii-sentiment/)."
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
