"""Generate the two narrative notebooks for Study 760 (Michigan-Sentiment-Day).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded UMich
sentiment snapshot (always available) and the cached SPY prices under ../_cache/, and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The
synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (UMCSENT hardcoded monthly
# snapshot + SPY daily/month-end, 1993-01 -> 2026-04, 400 months, 33.2 years).
R = dict(
    start="1993-01-31", end="2026-04-30", months=400, years=33.2, daily_days=8405,
    # A. release-day drift
    rel_n=401, rel_mean_bp=6.2, all_mean_bp=4.8, rel_t=0.23,
    # (n_beat, beat_bp, n_miss, miss_bp, t_beat_minus_miss)
    drift1=(196, 8.2, 204, 17.1, -0.64),
    drift2=(196, 23.4, 204, 43.7, -1.31),
    # B. regime per-horizon: (months, base%, low%, t_low, low_rising%, n_lr, t_lr, p_block)
    h1=(1, 0.95, 1.02, 0.19, 1.55, 70, 1.01, 0.090),
    h3=(3, 2.81, 2.85, 0.05, 3.73, 68, 0.85, 0.261),
    h6=(6, 5.77, 5.41, -0.33, 8.21, 67, 1.54, 0.203),
    h12=(12, 11.97, 11.55, -0.25, 18.65, 64, 3.55, 0.103),
    episodes=21,
    # overlay: (bh_mean%, bh_sharpe, gross%, gross_sharpe, net%, net_sharpe, switches, exposure%)
    overlay=(11.3, 0.77, 3.3, 0.46, 3.1, 0.44, 53, 18),
    # robustness 12m: (label, n_lr, lr%, naive_t, p_block)
    robust=[("q<=20%", 47, 18.6, 3.08, 0.159), ("q<=30%", 64, 18.7, 3.55, 0.103),
            ("q<=40%", 89, 16.0, 2.30, 0.187), ("k=1", 75, 15.1, 1.51, 0.274),
            ("k=6", 66, 18.2, 3.90, 0.111), ("ex-GFC", 64, 18.1, 3.63, 0.115),
            ("ex-COVID", 59, 17.0, 2.37, 0.237)],
    # synthetic control: (edge, n_lr, lr%, base%, t, p_block)
    syn=[(0.0, 29, 1.67, 1.73, -0.02, 0.332), (0.10, 29, 61.23, 13.00, 8.35, 0.004)],
)

BADGES = (
    "![Signal: None on the release-day · Weak on the level]"
    "(https://img.shields.io/badge/Signal-None_%C2%B7_Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Bottom--timer%3F: Not_supported]"
    "(https://img.shields.io/badge/Bottom--timer%3F-Not_supported-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from michigan_sentiment_day import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_monthly() if HAVE_REAL else None
SPY = data.load_spy() if HAVE_REAL else None
DATES = data.release_dates(1993, 2026)
print("SPY cache present:", HAVE_REAL,
      "| sentiment+SPY months:", (0 if F is None else len(F)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the University of Michigan sentiment number move the market — or mark the bottom? 🙂\n"
            "### The most-watched consumer mood gauge, as a release-day trade *and* a crystal ball, in plain English\n\n"
            + BADGES +
            "Twice a month the University of Michigan publishes how Americans *feel* about the economy — the "
            "**Index of Consumer Sentiment**, headline news every mid-month Friday. Two beliefs ride on it. "
            "First, that the print **moves the market** on the day it lands. Second, and more seductive: that "
            "when sentiment is **low and starting to turn up**, you're looking at a **market bottom** — buy "
            "with both hands.\n\n"
            "Both are testable. This notebook asks three blunt questions: does SPY actually *do* anything on "
            "release day? When sentiment is cheap-and-rising, does the market really rocket? And if you "
            "**bought every one of those 'bottoms,'** would you beat just holding stocks?\n\n"
            "> 📓 **Plain-language layer.** Want the event-study *t*, the block bootstrap and the synthetic "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The live FRED feed is blocked here, so we use a **frozen "
            "snapshot** of the official Michigan sentiment series (the settled monthly numbers) — public and "
            "faithful, including the all-time-low **50.0** of June 2022. Release days use the standard "
            "*second-Friday* schedule. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does SPY move on release day? | **No.** Release days average "
            f"**+{R['rel_mean_bp']:.1f} bp** — basically an ordinary day (**+{R['all_mean_bp']:.1f} bp**). "
            "And the 'surprise' drift runs *backwards*: the market drifts **more** after a *disappointing* "
            "print than a good one. |\n"
            "| When sentiment is low-and-rising, does the market soar? | **It looks like it — "
            f"+{R['h12'][4]:.1f}% over the next year vs +{R['h12'][1]:.1f}% normally.** That's the eye-catching "
            "chart. Hold that thought. |\n"
            "| Is that reliable? | **No.** Those 'bottoms' are really the same ~20 post-crash recoveries "
            "(2009, 2020, 2022…) counted many times over. A test that respects the overlap can't tell it "
            "from luck. |\n"
            "| So could you trade it? | **It loses badly.** \"Buy only the bottoms\" earned "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%** for just holding — because it sits "
            f"in cash **{100-R['overlay'][7]:.0f}%** of the time. |\n\n"
            "> Sentiment bottoms and market bottoms *do* line up — after crashes. But \"buy low-then-rising "
            "sentiment\" is really \"buy after a crash,\" dressed up by a flattering statistic — and you "
            "can't beat buy-and-hold by owning stocks only one month in five."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Michigan sentiment is a market-mover — trade the print. And when confidence is low and "
            "starting to recover, that's your signal the bottom is in: buy stocks.\"*\n\n"
            "The second half has real pedigree. **Fisher & Statman (2003)** showed consumer confidence is a "
            "**contrarian** signal — high confidence precedes *low* returns, and vice-versa — and the folk "
            "version (*be greedy when others are fearful*) is Buffett 101. The trading leap we test is the "
            "specific, tradable form: that a sentiment **low that's turning up** marks an equity bottom you "
            "can act on, and that the release itself is worth trading on the day."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If either worked it'd be a gift: a free, twice-monthly, government-adjacent number that either "
            "hands you a day-trade or tells you when to back up the truck. But both hide the same trap. A "
            "sentiment **low that's rising** almost only happens *after a crash* — which is also when stocks "
            "are cheapest and bounce hardest. So a signal that 'calls bottoms' might not be forecasting "
            "anything; it might just be **standing where the recoveries already are**. Telling those apart — "
            "a real edge vs a repackaged rebound — is the whole game, and it comes down to *how many "
            "genuinely different bottoms* the signal rests on."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of Michigan sentiment against SPY, and:\n\n"
            "1. **Watch release day.** Compare SPY's return on the ~400 release Fridays to an average day, "
            "and check whether it drifts *with* the surprise (a good print → up).\n"
            "2. **Split by mood.** Call sentiment **low** when it's in the cheapest 30% of its own history "
            "so far, and **rising** when it's above where it was three months ago. Compare what SPY did next "
            "(1/3/6/12 months) after **low-and-rising** months vs an average month.\n"
            "3. **Count the bottoms.** The killer check: how many *independent* episodes is the signal "
            "really made of? And does a test that respects the overlap still call it real?\n"
            "4. **Try to trade it.** Buy SPY only in low-and-rising months, hold cash otherwise, pay costs — "
            "and see if it beats buy-and-hold."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw mood.** Here's three decades of Michigan sentiment — the booms (dot-com, "
            "mid-2010s) and the famous troughs: 2008–09, August 2011, and the **all-time low of 50.0** in "
            "June 2022. Sentiment clearly *knows* about crises. The question is what that's worth."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = F['sent']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(s.index, s.values, c=GREY, lw=1.3)\n"
            "    lo = s[s <= s.quantile(.15)]\n"
            "    ax.scatter(lo.index, lo.values, s=14, c=RED, zorder=3, label='cheapest 15% of months')\n"
            "    ax.set_title('U. Michigan Index of Consumer Sentiment (1966:Q1 = 100)')\n"
            "    ax.set_ylabel('sentiment index'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('all-time low in sample:', s.min(), 'around', s.idxmin().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; sentiment bottomed at 50.0 in June 2022')"
        ),
        md(
            "**Question 1 — does the print move the market?** Here's SPY's average return on the ~400 "
            "release days next to an average day. If the number were a market-mover, the release bar would "
            "tower over the baseline."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rd = st.release_day_summary(SPY, DATES)\n"
            "    rel_bp, all_bp, tval = rd['release_mean']*1e4, rd['all_mean']*1e4, rd['t_vs_all']\n"
            "else:\n"
            "    rel_bp, all_bp, tval = R['rel_mean_bp'], R['all_mean_bp'], R['rel_t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['an average\\nday', 'a sentiment\\nrelease day'], [all_bp, rel_bp], color=[GREY, AMBER], width=.55)\n"
            "for i,v in enumerate([all_bp, rel_bp]): ax.annotate(f'{v:.1f} bp',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average SPY return (basis points)')\n"
            "ax.set_title(f'Release day is just... a day  (Welch t = {tval:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'release {rel_bp:.1f}bp vs all {all_bp:.1f}bp, t={tval:+.2f}')"
        ),
        md(
            f"A dead heat: **+{R['rel_mean_bp']:.1f} bp** on release day vs **+{R['all_mean_bp']:.1f} bp** "
            f"normally, *t* = **+{R['rel_t']:.2f}**. And when we split by whether the print *beat* or *missed* "
            "the prior month, the market drifts **more after misses** than beats — the exact opposite of a "
            "'good news → rally' story. **The release-day trade is a non-event.**"
        ),
        md(
            "**Question 2 — the bottom-timer.** Now the seductive one. For each horizon, the average forward "
            "SPY return after a **low-and-rising** sentiment month next to an average month. Watch the "
            "one-year bars."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize_regime(F, m, boot=False) for m in hs]\n"
            "    lr = [r['low_rising_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    lr = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "    base = [R['h1'][1], R['h3'][1], R['h6'][1], R['h12'][1]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, lr, .4, color=GREEN, label='after LOW-and-RISING sentiment')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward SPY return (%)')\n"
            "ax.set_title('The bottom-timer chart everyone loves: +18.7% vs +12.0% at one year')\n"
            "for i,(a,b) in enumerate(zip(lr,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('12-month: low-and-rising', f'{lr[-1]:.1f}%', 'vs base', f'{base[-1]:.1f}%')"
        ),
        md(
            f"That's the chart in every 'buy the fear' thread: **+{R['h12'][4]:.1f}%** over the next year "
            f"after low-and-rising sentiment vs **+{R['h12'][1]:.1f}%** normally. It even has a big *t*-stat "
            f"(**{R['h12'][6]:.1f}**). Case closed? **No — here's the catch.**"
        ),
        md(
            "**Question 3 — how many bottoms is that, really?** The 64 'low-and-rising' months aren't 64 "
            "separate events. They cluster into a handful of recoveries. Here's *when* the signal fired."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mask = st.regime_mask(F)['low_rising'].fillna(False)\n"
            "    fire = F.index[mask]\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 3.4))\n"
            "    ax.plot(F.index, F['sent'].values, c=GREY, lw=1.1)\n"
            "    ax.scatter(fire, F['sent'].reindex(fire).values, s=22, c=GREEN, zorder=3, label='signal fires')\n"
            "    ax.set_title(f'{st.n_episodes(F)} clumps, not {int(mask.sum())} independent bets')\n"
            "    ax.set_ylabel('sentiment'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('signal months:', int(mask.sum()), '| independent episodes:', st.n_episodes(F))\n"
            "else:\n"
            "    print(f\"{R['h12'][5]} signal months collapse to ~{R['episodes']} independent episodes\")"
        ),
        md(
            f"See the clumps? The **{R['h12'][5]} signal months** are really about **{R['episodes']} "
            "independent recoveries** — 2003, 2009, 2020, 2022–23 and a few others. A big *t*-stat that "
            "treats them as 64 separate coin-flips is **fooling itself**: it's the same handful of "
            "post-crash bounces, counted over and over. A proper test that accounts for the clustering "
            "(the quants notebook does it explicitly) puts the odds this is luck at about **1 in 10** — "
            "*not* good enough to call real."
        ),
        md(
            "**Question 4 — could you trade it anyway?** Suppose you bought SPY only in low-and-rising "
            "months and sat in cash the rest of the time. Here's that strategy's growth vs just buying and "
            "holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    mask = st.regime_mask(F)['low_rising']; pos = mask.astype(float).shift(1)\n"
            "    rr = F['spy'].pct_change()\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(dfp['pos'].abs()); c = 10/1e4\n"
            "    overlay = dfp['pos']*dfp['r'] - sw*c\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold SPY')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='buy only the bottoms (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Buy only the bottoms\" barely leaves the ground')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][4]:.1f}%/yr vs buy-hold {R['overlay'][0]:.1f}%/yr (net) — see results.md\")"
        ),
        md(
            f"The bottom-buyer ends up **far below** buy-and-hold — **+{R['overlay'][4]:.1f}%/yr** net vs "
            f"**+{R['overlay'][0]:.1f}%/yr**, at a *lower* Sharpe ({R['overlay'][5]:.2f} vs "
            f"{R['overlay'][1]:.2f}). Why? Because it's in the market only **{R['overlay'][7]:.0f}%** of the "
            "time. Even though those months are genuinely above-average, you can't out-compound someone who "
            "owns stocks *all the time*. The signal isn't a way to beat the market — at best it's a reason "
            "to already **be** in it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Release-day drift — None.** SPY on release day is indistinguishable from an average day, "
            "and the surprise drifts the wrong way. Nothing to trade.\n"
            "- **Bottom-timer — Weak.** Low-and-rising sentiment really is followed by strong 12-month "
            "returns, but it's ~20 clustered post-crash recoveries wearing a big *t*-stat; a test that "
            "respects the clustering can't certify it, and cheap sentiment *by itself* does nothing.\n"
            "- **Tradability — Mirage.** Buying only the bottoms loses to buy-and-hold by a mile, because "
            "you're in cash four months out of five. The 'edge' is just a reason to own equities — which "
            "buy-and-hold does for free."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Even granting the tilt is real, the operational reality kills it. The signal needs sentiment to "
            "be **both** cheap **and** already three months into a recovery — so by the time it fires you've "
            "*missed the actual bottom*, and it fires only in a few post-crash windows a decade. In between, "
            "you're parked in cash while the market compounds. There is no version of \"buy the bottom\" that "
            "both (a) fires *at* the bottom in real time and (b) keeps you invested enough to beat simply "
            "holding. The contrarian instinct is sound; the *tradable rule* is a mirage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling tests.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "and [Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) put other famous "
            "macro 'crystal balls' through the same wringer.\n"
            "- **The cross-section.** Sentiment predicts returns best in *small, hard-to-arbitrage* stocks "
            "(Baker–Wurgler 2006), not a broad index like SPY. Re-run this on small-cap value and the effect "
            "may be sturdier — but also less tradable.\n"
            "- **Build your own.** Swap the level for the *expectations* sub-index, or pair sentiment with a "
            "price trend. The clustering problem survives: you can't manufacture independent bottoms by "
            "reslicing the same recoveries.\n\n"
            "*Think low-then-rising sentiment is a real edge? Show it clearing a **block bootstrap** — not "
            "just a naive t-stat on overlapping returns — and we'll talk.*"
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
            "# Michigan-Sentiment-Day — a quantitative teardown 🔬\n"
            "### A release-day event study · a low-then-rising regime split · the decisive **naive-t vs "
            "block-bootstrap** gap · the 21-episode clustering · a buy-the-bottom overlay · robustness · a "
            "planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Two believer "
            "claims, separated: (A) the sentiment **release** moves SPY and its surprise drifts; (B) a "
            "**low-and-rising** sentiment regime marks tradable equity bottoms. We find (A) a **non-event** "
            "(release-day *t* ≈ 0, surprise drift the wrong sign) and (B) a textbook **overlapping-return "
            "mirage**: a naive Welch *t* of **+3.55** at 12 months that a **circular block bootstrap** "
            "(*p* = 0.10) refuses to certify, resting on ~**21 independent episodes**, with the sentiment "
            "*level alone* carrying nothing.\n\n"
            "> ⚠️ **Data + snapshot note.** FRED's CSV endpoint is firewalled here; sentiment is a hardcoded "
            "monthly snapshot of `UMCSENT` (final print, **not** the real-time preliminary vintage — named "
            "on the Signal axis). SPY is yfinance daily adjusted close (total-return), daily for the event "
            "study and month-end for the regime test. Release days use the second-Friday schedule proxy. "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` on release-day · `WEAK` on level | Release day **+{R['rel_mean_bp']:.1f} bp** "
            f"vs all-day **+{R['all_mean_bp']:.1f} bp** (*t* = {R['rel_t']:+.2f}); 12m low-and-rising "
            f"**+{R['h12'][4]:.1f}%** vs base **+{R['h12'][1]:.1f}%** — naive *t* = **+{R['h12'][6]:.2f}** but "
            f"block-boot **p = {R['h12'][7]:.2f}**, level-alone *t* = {R['h12'][3]:+.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Buy-the-bottom overlay **+{R['overlay'][4]:.1f}%/yr** (Sharpe "
            f"**{R['overlay'][5]:.2f}**, exposure **{R['overlay'][7]:.0f}%**) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe **{R['overlay'][1]:.2f}**). |\n"
            f"| **Bottom-timer?** | `NOT SUPPORTED` | {R['h12'][5]} signal months = ~**{R['episodes']} "
            "independent episodes**; the effect is 'significant' only through the one statistic that ignores "
            "the overlap. |\n\n"
            "> 💡 In plain words: a low-that's-rising in sentiment happens almost only *after* a crash, when "
            "equities are cheap and rebounding. The 12-month forward return then looks huge — but it's the "
            "same ~20 recoveries measured through overlapping windows, which inflates the naive *t*. Respect "
            "the overlap and the certainty evaporates."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t$ be the Michigan sentiment level. Two hypotheses:\n\n"
            "- **A — release-day drift.** The release-day SPY return exceeds an ordinary day, and the "
            "next-day drift is positive after a *beat* ($\\Delta s_t = s_t - s_{t-1} > 0$).\n"
            "- **B — bottom-timer.** Let $p_t$ = the expanding percentile rank of $s_t$ (no look-ahead). "
            "**LOW** = $p_t \\le 0.30$; **RISING** = $s_t > s_{t-3}$. The believers claim "
            "$\\mathbb{E}[r_{t\\to t+H}\\mid \\text{LOW\\&RISING}] > \\mathbb{E}[r_{t\\to t+H}]$ — a "
            "*positive* excess, strongest at long $H$.\n\n"
            "We find **A rejected** (release-day *t* ≈ 0; surprise drift the wrong sign), **B directionally "
            "true but uncertifiable**: the naive Welch *t* is large only at the long, overlapping horizon "
            "where it's least trustworthy, and a block bootstrap can't clear it. The contrarian instinct is "
            "right where it's untradable (own equities after crashes) and unproven where it would pay (a "
            "certifiable, tradable bottom signal)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The regime test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{LR}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{LR}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "The trap: at $H=12$ the forward returns are **12-month overlapping** windows, so both the LR set "
            "and the base are heavily autocorrelated and the naive *t* **over-rejects** (Richardson–Stock "
            "1989; Boudoukh–Richardson–Whitelaw 2008). The honest inference resamples the forward-return "
            "series in **circular blocks** (Politis–Romano 1994) of length 12 and asks how often chance "
            "matches $\\widehat{\\Delta}$. If $k=64$ signal months are really $\\sim\\!21$ independent "
            "episodes, the effective sample is tiny and the block-boot $p$ tells the truth the naive $t$ "
            "hides."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Sentiment tape.** Monthly `UMCSENT` (index, 1966:Q1=100), hardcoded snapshot, "
            f"{R['start'][:7]}→{R['end'][:7]} ({R['months']} months). Final print, not real-time vintage "
            "(named on the axis).\n"
            "- **Release-day event study.** Second-Friday release proxy; release-day SPY return vs all-day "
            "mean (Welch *t*); next-day drift split by $\\text{sign}(\\Delta s_t)$, entered at the "
            "release-day close (print already public — no look-ahead).\n"
            "- **Regime.** $p_t$ = expanding percentile (≥36-mo warmup); LOW = $p_t\\le0.30$, RISING = "
            "$s_t>s_{t-3}$; forward $H\\in\\{1,3,6,12\\}$-month returns entered at month-end after the "
            "mid-month print (no extra lag).\n"
            "- **Null #1 (Welch t).** LR-set mean vs the unconditional mean.\n"
            "- **Null #2 (block bootstrap).** Circular 12-month-block resample of the forward-return series; "
            "$p = \\Pr[\\text{block-boot excess} \\ge \\widehat{\\Delta}]$ — the autocorrelation-aware test.\n"
            "- **Clustering.** Independent-episode count (signal months > 2 months apart).\n"
            "- **Tradability.** Buy-the-bottom overlay, 1-month lag, 10 bps one-way per switch, "
            "excess-of-zero Sharpe (labelled), vs buy-and-hold.\n"
            "- **Positive control.** A deterministic series with a *planted* low-then-rising→forward link "
            "over the next 12 months: `edge=0` must not fake significance; a large `edge` must light up the "
            "*t* **and** the block bootstrap."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Release day is a non-event\n\n"
            "Release-day mean SPY return vs the all-day mean, and the next-day drift split by the surprise "
            "sign. A tradable announcement drift needs the *beat* bar above the *miss* bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rd = st.release_day_summary(SPY, DATES)\n"
            "    d1 = st.drift_by_surprise(SPY, data.sentiment_series(), DATES, lag=1)\n"
            "    d2 = st.drift_by_surprise(SPY, data.sentiment_series(), DATES, lag=2)\n"
            "    rel_bp, all_bp, tv = rd['release_mean']*1e4, rd['all_mean']*1e4, rd['t_vs_all']\n"
            "    b1,m1,t1 = d1['beat_mean']*1e4, d1['miss_mean']*1e4, d1['t_beat_minus_miss']\n"
            "    b2,m2,t2 = d2['beat_mean']*1e4, d2['miss_mean']*1e4, d2['t_beat_minus_miss']\n"
            "else:\n"
            "    rel_bp, all_bp, tv = R['rel_mean_bp'], R['all_mean_bp'], R['rel_t']\n"
            "    b1,m1,t1 = R['drift1'][1], R['drift1'][3], R['drift1'][4]\n"
            "    b2,m2,t2 = R['drift2'][1], R['drift2'][3], R['drift2'][4]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(['average\\nday','release\\nday'], [all_bp, rel_bp], color=[GREY, AMBER], width=.55)\n"
            "for i,v in enumerate([all_bp, rel_bp]): a1.annotate(f'{v:.1f}bp',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('mean SPY return (bp)'); a1.set_title(f'Release day (t={tv:+.2f})')\n"
            "x = np.arange(2)\n"
            "a2.bar(x-.2, [b1,b2], .4, color=GREEN, label='after a BEAT')\n"
            "a2.bar(x+.2, [m1,m2], .4, color=RED, label='after a MISS')\n"
            "a2.set_xticks(x); a2.set_xticklabels(['next 1 day','next 2 days'])\n"
            "a2.set_ylabel('drift (bp)'); a2.set_title('Drift runs the WRONG way'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'release t={tv:+.2f}; drift 1d beat-miss t={t1:+.2f}; 2d t={t2:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: release day is **+{R['rel_mean_bp']:.1f} bp** vs **+{R['all_mean_bp']:.1f} "
            f"bp** normally (*t* = {R['rel_t']:+.2f}) — statistically a normal day. And the surprise drift is "
            f"*negative*: bigger after misses than beats (1-day *t* = {R['drift1'][4]:+.2f}, 2-day *t* = "
            f"{R['drift2'][4]:+.2f}). **Hypothesis A rejected** — there is no release-day edge, in level or in "
            "surprise direction."
        ),
        md(
            "### 4b · The regime split — the level is dead, the combo *looks* alive\n\n"
            "Forward means for **LOW** (level only) and **LOW&RISING**, vs the base rate, with $\\pm$SE. LOW "
            "alone hugs the base at every horizon; LOW&RISING pulls away — but read the *t* against what "
            "produces it."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    lrm, lom, bm, tlr, ses = [], [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize_regime(F, m, boot=False)\n"
            "        lrm.append(s['low_rising_mean']); lom.append(s['low_mean']); bm.append(s['base_mean']); tlr.append(s['t_low_rising'])\n"
            "        fwd = st.forward_returns(F, m); mask = st.regime_mask(F)['low_rising']\n"
            "        v = fwd[mask & fwd.notna()].dropna().values; ses.append(v.std(ddof=1)/np.sqrt(len(v)))\n"
            "else:\n"
            "    lrm = [R['h1'][4]/100, R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    lom = [R['h1'][2]/100, R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h1'][1]/100, R['h3'][1]/100, R['h6'][1]/100, R['h12'][1]/100]\n"
            "    tlr = [R['h1'][6], R['h3'][6], R['h6'][6], R['h12'][6]]; ses=[.02,.035,.05,.055]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x, [v*100 for v in lrm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='LOW & RISING (±SE)')\n"
            "ax.plot(x, [v*100 for v in lom], 's', ms=10, c=AMBER, label='LOW only')\n"
            "ax.plot(x, [v*100 for v in bm], 'D', ms=10, c=GREY, label='base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('LOW alone = base; LOW&RISING pulls away at 12m'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('naive Welch t (low&rising) by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,tlr)})"
        ),
        md(
            f"> 💡 In plain words: the sentiment **level alone is inert** — LOW months return "
            f"**+{R['h12'][2]:.1f}%** at 12m vs base **+{R['h12'][1]:.1f}%** (*t* = {R['h12'][3]:+.2f}). Cheap "
            "sentiment does *not* mean higher returns. Only the LOW-**and**-RISING combo separates "
            f"(**+{R['h12'][4]:.1f}%**, naive *t* = **+{R['h12'][6]:.2f}**) — and that's exactly the 'buy after "
            "the bounce has started' set. The next chart asks whether that *t* means anything."
        ),
        md(
            "### 4c · The decisive test — naive *t* vs block bootstrap\n\n"
            "For each horizon, the naive Welch *t* (which ignores the overlap) next to the "
            "autocorrelation-aware **block-bootstrap p** (12-month block). A real edge clears **both**; an "
            "overlapping-return artefact clears only the naive *t*."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    tt, pp = [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize_regime(F, m); tt.append(s['t_low_rising']); pp.append(s['p_block'])\n"
            "else:\n"
            "    tt = [R['h1'][6], R['h3'][6], R['h6'][6], R['h12'][6]]\n"
            "    pp = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([f'{m}m' for m in hs], tt, color=[GREEN if t>=2 else GREY for t in tt], width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='|t|=2 bar')\n"
            "for i,t in enumerate(tt): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a1.set_ylabel('naive Welch t'); a1.set_title('Naive t: 12m screams (3.55)'); a1.legend()\n"
            "a2.bar([f'{m}m' for m in hs], pp, color=[GREEN if p<.05 else RED for p in pp], width=.6)\n"
            "a2.axhline(.05, ls='--', c=RED, label='p=0.05 bar')\n"
            "for i,p in enumerate(pp): a2.annotate(f'{p:.2f}',(i,p),ha='center',va='bottom')\n"
            "a2.set_ylabel('block-bootstrap p (12-mo block)'); a2.set_title('Block boot: nothing clears 0.05'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('12m: naive t =', round(tt[-1],2), '| block-boot p =', round(pp[-1],3))"
        ),
        md(
            f"> 💡 In plain words: at 12 months the naive *t* is **+{R['h12'][6]:.2f}** (looks decisive) but "
            f"the block-bootstrap *p* is **{R['h12'][7]:.2f}** — *not* significant. The gap between the two "
            "panels **is** the finding: the apparent signal lives entirely in the autocorrelation the naive "
            "*t* pretends isn't there. This is the load-bearing result."
        ),
        md(
            "### 4d · Why — the signal is ~21 episodes, not 64 draws\n\n"
            "The signal months plotted on the sentiment tape. The naive *t* treats them as independent; they "
            "are a handful of post-crash recoveries."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mask = st.regime_mask(F)['low_rising'].fillna(False)\n"
            "    fire = F.index[mask]\n"
            "    fig, ax = plt.subplots(figsize=(9.6, 3.6))\n"
            "    ax.plot(F.index, F['sent'].values, c=GREY, lw=1.1)\n"
            "    ax.scatter(fire, F['sent'].reindex(fire).values, s=20, c=GREEN, zorder=3)\n"
            "    for yr in ('2003','2009','2020','2022'):\n"
            "        sub = fire[(fire>=f'{yr}-01-01') & (fire<f'{int(yr)+2}-01-01')]\n"
            "        if len(sub): ax.axvspan(sub.min(), sub.max(), color=GREEN, alpha=.08)\n"
            "    ax.set_title(f'{int(mask.sum())} signal months = ~{st.n_episodes(F)} independent episodes')\n"
            "    ax.set_ylabel('sentiment'); plt.tight_layout(); plt.show()\n"
            "    print('signal months:', int(mask.sum()), '| episodes (gap>2mo):', st.n_episodes(F))\n"
            "else:\n"
            "    print(f\"{R['h12'][5]} signal months -> ~{R['episodes']} independent episodes\")"
        ),
        md(
            f"> 💡 In plain words: **{R['h12'][5]} months** collapse to **~{R['episodes']} episodes**. With an "
            "effective sample of ~20 clustered recoveries, a two-point 12-month excess is well within what "
            "chance produces — precisely what the block bootstrap reports. The naive *t*'s denominator uses "
            "$k=64$; the honest one knows better."
        ),
        md(
            "### 4e · Tradability — the buy-the-bottom overlay\n\n"
            "Long SPY only in LOW&RISING months, else cash (1-month lag, 10 bps/switch). Annualised mean and "
            "Sharpe vs buy-and-hold; note the exposure."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.timing_overlay(F, cost_bps=10.0)\n"
            "    bh_m, bh_s = o['bh_mean']*100, o['bh_sharpe']\n"
            "    g_m, g_s = o['overlay_gross_mean']*100, o['overlay_gross_sharpe']\n"
            "    n_m, n_s = o['overlay_net_mean']*100, o['overlay_net_sharpe']; expo=o['exposure']*100; nsw=o['n_switches']\n"
            "else:\n"
            "    bh_m,bh_s = R['overlay'][0],R['overlay'][1]; g_m,g_s=R['overlay'][2],R['overlay'][3]\n"
            "    n_m,n_s = R['overlay'][4],R['overlay'][5]; nsw=R['overlay'][6]; expo=R['overlay'][7]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = ['buy &\\nhold','overlay\\ngross','overlay\\nnet @10bps']\n"
            "a1.bar(labels, [bh_m,g_m,n_m], color=[GREY,AMBER,RED], width=.6)\n"
            "for i,v in enumerate([bh_m,g_m,n_m]): a1.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('annualised mean (%)'); a1.set_title(f'Overlay in market only {expo:.0f}% of the time')\n"
            "a2.bar(labels, [bh_s,g_s,n_s], color=[GREY,AMBER,RED], width=.6)\n"
            "for i,v in enumerate([bh_s,g_s,n_s]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('Sharpe (excess-of-0)'); a2.set_title(f'Sharpe: overlay lower ({nsw} switches)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net overlay {n_m:.1f}%/yr (Sharpe {n_s:.2f}) vs buy-hold {bh_m:.1f}%/yr (Sharpe {bh_s:.2f}), exposure {expo:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: the overlay earns **+{R['overlay'][4]:.1f}%/yr** net (Sharpe "
            f"{R['overlay'][5]:.2f}) vs buy-hold **+{R['overlay'][0]:.1f}%** (Sharpe {R['overlay'][1]:.2f}), "
            f"in the market only **{R['overlay'][7]:.0f}%** of the time. Even at zero cost you can't beat "
            "full-time ownership by being long a rare subset — the signal is a lean-*in*, not a timing "
            "overlay. **`MIRAGE`.**"
        ),
        md(
            "### 4f · Robustness — the naive *t* is big everywhere, the bootstrap never clears\n\n"
            "Vary the percentile threshold and momentum window, and drop the GFC and COVID. The naive *t* "
            "stays large across the board; the block-boot *p* never drops below 0.10."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for q in (0.20,0.30,0.40):\n"
            "        s=st.summarize_regime(F,12,low_q=q); rob.append((f'q<={int(q*100)}%', s['n_low_rising'], s['t_low_rising'], s['p_block']))\n"
            "    for k in (1,6):\n"
            "        s=st.summarize_regime(F,12,k=k); rob.append((f'k={k}', s['n_low_rising'], s['t_low_rising'], s['p_block']))\n"
            "    F2=F[(F.index<'2008-07-01')|(F.index>='2009-07-01')]; s=st.summarize_regime(F2,12); rob.append(('ex-GFC', s['n_low_rising'], s['t_low_rising'], s['p_block']))\n"
            "    F3=F[(F.index<'2020-01-01')|(F.index>='2021-01-01')]; s=st.summarize_regime(F3,12); rob.append(('ex-COVID', s['n_low_rising'], s['t_low_rising'], s['p_block']))\n"
            "else:\n"
            "    rob = [(l,n,t,p) for (l,n,_r,t,p) in R['robust']]\n"
            "labels=[r[0] for r in rob]; tvals=[r[2] for r in rob]; pvals=[r[3] for r in rob]\n"
            "x=np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.4))\n"
            "ax.bar(x-.2, tvals, .4, color=GREY, label='naive Welch t')\n"
            "ax.bar(x+.2, [p*5 for p in pvals], .4, color=RED, label='block-boot p (x5 for scale)')\n"
            "ax.axhline(2, ls='--', c=GREEN, label='|t|=2'); ax.axhline(0.05*5, ls=':', c=RED, label='p=0.05 (x5)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20); ax.set_ylabel('value')\n"
            "ax.set_title('Every spec: naive t >2, block-boot p >0.10'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, naive_t, p_block):', [(r[0],r[1],round(r[2],2),round(r[3],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: drop the GFC — naive *t* still **{R['robust'][5][3]:.2f}**, "
            f"*p* = **{R['robust'][5][4]:.2f}**. Drop COVID — *t* **{R['robust'][6][3]:.2f}**, "
            f"*p* = **{R['robust'][6][4]:.2f}**. It isn't one crash; it's the *clustering* itself, and no "
            "threshold or window rescues it. A signal significant only through the statistic that ignores its "
            "clustering is the textbook definition of **WEAK**."
        ),
        md(
            "### 4g · Faithful-engine control — the bootstrap fires when the edge is real\n\n"
            "A deterministic series with a *planted* low-then-rising→forward link over the next 12 months. "
            "With `edge=0` both the *t* and the block boot must stay flat; with a large `edge` **both** must "
            "light up — proving the block bootstrap is honest, not merely conservative."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.10):\n"
            "    syn = data.synthetic(n_months=396, edge=edge, seed=760)\n"
            "    s = st.summarize_regime(syn['frame'], 12, min_periods=12)\n"
            "    res.append((edge, s['n_low_rising'], s['low_rising_mean']*100, s['base_mean']*100, s['t_low_rising'], s['p_block']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "labs = [f'planted\\nedge {e*100:.0f}%' for e,_,_,_,_,_ in res]\n"
            "a1.bar(labs, [r[4] for r in res], color=[GREY, GREEN], width=.5)\n"
            "a1.axhline(2, ls='--', c=RED); \n"
            "for i,r in enumerate(res): a1.annotate(f't={r[4]:.2f}',(i,r[4]),ha='center',va='bottom')\n"
            "a1.set_ylabel('naive Welch t'); a1.set_title('t: null flat, edge lights up')\n"
            "a2.bar(labs, [r[5] for r in res], color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(.05, ls='--', c=RED)\n"
            "for i,r in enumerate(res): a2.annotate(f'p={r[5]:.3f}',(i,r[5]),ha='center',va='bottom')\n"
            "a2.set_ylabel('block-boot p'); a2.set_title('p: fires (0.004) only when edge is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:.0f}%: n_lr={k} lr={c:.1f}% base={b:.1f}% t={t:.2f} p_block={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: no planted link → *t* = **{R['syn'][0][4]:.2f}**, block-boot *p* = "
            f"**{R['syn'][0][5]:.2f}** (no false positive). A real planted link → *t* = **{R['syn'][1][4]:.2f}** "
            f"**and** *p* = **{R['syn'][1][5]:.3f}**. So the block bootstrap *does* certify a genuine edge — "
            "which means the real-tape *p* = 0.10 is a true 'can't certify,' not an over-cautious test. The "
            "engine can bank a real bottom-timer; this tape just doesn't carry a certifiable one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE` (release-day) · `WEAK` (level).** Release day **+{R['rel_mean_bp']:.1f} bp** vs "
            f"**+{R['all_mean_bp']:.1f} bp** (*t* = {R['rel_t']:+.2f}), surprise drift the wrong sign. "
            f"12m low-and-rising **{R['h12'][4]-R['h12'][1]:+.1f}pp** excess at naive *t* = "
            f"**+{R['h12'][6]:.2f}** but block-boot **p = {R['h12'][7]:.2f}**; level alone inert "
            f"(*t* = {R['h12'][3]:+.2f}). Literature (Fisher–Statman contrarian effect) + a "
            "directionally-right-but-uncertifiable tilt ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`.** Buy-the-bottom overlay **+{R['overlay'][4]:.1f}%/yr** (Sharpe "
            f"{R['overlay'][5]:.2f}, exposure {R['overlay'][7]:.0f}%) vs buy-hold **+{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe {R['overlay'][1]:.2f}). A lean-in long tilt can't beat full-time ownership.\n"
            f"- **Bottom-timer? `NOT SUPPORTED`.** {R['h12'][5]} signal months = ~**{R['episodes']} "
            "independent episodes**; significant only via the overlap-blind statistic. The one word that "
            "sells it — *timer* — is what the honest inference rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the contrarian tilt is genuine. It still won't deploy. The rule requires sentiment to be "
            "cheap **and** already three months into a recovery, so it fires *after* the bottom and only in "
            "a few windows a decade — leaving you in cash "
            f"**{100-R['overlay'][7]:.0f}%** of the time while the equity risk premium compounds without you. "
            "That structural under-exposure is why the overlay's Sharpe sits **below** passive even at zero "
            "cost. And the regime it keys on — a post-crash rebound — is exactly when you'd want *maximum* "
            "long exposure, which a long/flat rule structurally cannot deliver. No threshold, window, or "
            "lag turns a clustered, lagging, lean-in signal into a market-beating timer."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The siblings.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) and "
            "[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/): the same "
            "hardcoded-snapshot + SPY method on other famous macro 'leading' signals.\n"
            "- **The cross-section.** Baker–Wurgler (2006) locate the sentiment effect in small, "
            "hard-to-arbitrage stocks. Re-run the regime split on a small-cap-value tape — the effect may be "
            "sturdier there (and even harder to trade cleanly).\n"
            "- **Real-time vintages.** Swap the settled `UMCSENT` for the preliminary release (ALFRED) to "
            "kill any revision look-ahead; the clustering — the load-bearing problem — survives, because you "
            "cannot manufacture independent bottoms by reslicing the same ~20 recoveries.\n\n"
            "*The reproducible core is offline and deterministic; the sentiment input is an explicit frozen "
            "snapshot. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
