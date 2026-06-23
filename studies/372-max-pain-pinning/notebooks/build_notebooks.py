"""Generate the two narrative notebooks for Study 372 (Max-Pain Pinning).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached option-chain
max-pain snapshot under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic pinning control runs anywhere with no network.
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


# Frozen headline numbers — mirror of docs/results.md (yfinance option-chain max-pain snapshot
# of 20 liquid US underlyings as-of 2026-06-22, + the deterministic synthetic pinning control).
R = dict(
    asof="2026-06-22", n_names=20,
    median_gap=2.06, mean_gap=2.16, within_0p5=25, within_1=30, max_gap=7.48,
    # synthetic control: (pin, d_mp, d_anchor, diff, paired_t, win%, placebo_p, fade_g%, fade_n%, hit%)
    syn=[(0.00, 3.69, 2.12, -1.57, -11.49, 26, 0.467, 0.05, -0.00, 51),
         (0.05, 3.04, 2.07, -0.97, -7.29, 32, 0.000, 0.73, 0.68, 63),
         (0.15, 2.15, 2.38, 0.23, 1.82, 48, 0.000, 1.73, 1.68, 75),
         (0.40, 1.32, 3.08, 1.76, 14.33, 69, 0.000, 2.82, 2.77, 86)],
    # robustness: false-positive check (n_episodes, paired_t, placebo_p) at pin=0
    fp=[(200, -8.46, 0.66), (400, -11.49, 0.47), (800, -16.35, 0.39)],
    # robustness: pinning needs days (dte, paired_t, win%) at pin=0.15
    dte=[(3, -5.12, 36), (5, 1.82, 48), (10, 9.59, 60)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Pinned%3F: Busted](https://img.shields.io/badge/Pinned%3F-Busted-8b949e?style=flat-square)\n\n"
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

from max_pain_pinning import data, strategy as st

HAVE_REAL = data.have_real()
SNAP = data.load_real() if HAVE_REAL else None
print("real max-pain snapshot present:", HAVE_REAL,
      "| underlyings:", (0 if SNAP is None else len(SNAP)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"The stock will pin to max-pain at expiry\" — does it? 📌\n"
            "### The expiry-day legend that price gets dragged to the strike where options hurt most, in plain English\n\n"
            + BADGES +
            "Every options-expiration Friday, someone says it: *\"watch, it'll pin to max-pain.\"* "
            "**Max-pain** is the strike price where the most option open interest expires "
            "**worthless** — the price that inflicts maximum *pain* on option buyers (and saves option "
            "sellers the most money). The legend says dealers, hedging their books, quietly steer the "
            "stock toward that strike into the close.\n\n"
            "It's a great story with a real kernel — and a much bigger claim bolted on top. This "
            "notebook pulls the two apart: a *modest* tendency for some single stocks to cling to busy "
            "strikes is real; **\"the stock lands on the max-pain strike\"** as a law you can trade is "
            "not.\n\n"
            "> 📓 **Plain-language layer.** Want the paired *t*-test, the placebo and the planted-pin "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** A free option chain only shows the *next* expiries, so we can "
            "compute **today's** max-pain but never watch those contracts *expire*. So our real data is "
            "a **snapshot** (where price sits vs max-pain right now), and the *landing* claim is tested "
            "on a simulator where we know the truth. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is price sitting on the max-pain strike right now? | **No.** Across 20 liquid names today, "
            f"spot is a **median ~{R['median_gap']:.0f}%** away from max-pain — only **{R['within_1']}%** "
            "are even within 1%. |\n"
            "| Do the most-traded names pin tightest? | **The big indexes sit closest** (SPY/QQQ ~0.3% "
            "off) — but that's *arbitrage keeping them efficient*, not a pin, and single stocks scatter "
            "much wider. |\n"
            "| Can we test the 'lands on max-pain at expiry' claim on free data? | **Not directly** — no "
            "history of expiry-day closes is kept. So we **simulate** it: build worlds with and without "
            "a pin and check what the test says. |\n"
            "| What does the simulator say? | With **no** pin the close drifts to where it *started*, "
            "not to max-pain. A pin only shows up when we **deliberately plant a strong one** — then a "
            "fade-to-max-pain trade finally pays. |\n\n"
            "> Max-pain is a real number with a thin, real academic footnote. \"The stock pins to it\" "
            "is mostly a story — and a snapshot already catches price a couple percent away."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Add up all the call and put open interest at every strike. The **max-pain** strike is "
            "the one where, if the stock settled there, the **most** option value would expire "
            "worthless. Dealers who sold those options hedge in a way that nudges the stock toward that "
            "strike — so at expiry, the stock **pins** to max-pain.\"*\n\n"
            "The mechanism isn't crazy: a market-maker short a pile of options near a strike hedges by "
            "**buying dips and selling rips**, which is stabilising — it can pull a drifting stock "
            "toward the busy strike. Academics have even measured a *whiff* of it in single stocks. The "
            "question is whether that whiff is the **max-pain law** the folklore sells."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were a law, it'd be a money printer: a few days before expiry you'd see where "
            "max-pain is, bet the stock drifts there, and collect. That's exactly the trade retail "
            "max-pain sites imply. But two things have to be true for it to pay: the stock must "
            "**actually land** near max-pain (not just *be near a strike* — everything is near *some* "
            "strike), and the pull must be **bigger than the noise and the costs** of trading single-name "
            "options into expiry. A vivid story that's really just \"price is near a strike, and strikes "
            "are everywhere\" is **not** an edge — it's geometry."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"**Part 1 — look at today.** Pull the live option chain for **{R['n_names']}** liquid US "
            "names, compute each one's max-pain strike, and measure the gap to spot. If price were "
            "pinned *now*, those gaps would huddle near zero.\n\n"
            "**Part 2 — simulate the landing.** Free data won't tell us how past expiries closed, so we "
            "build a **pinning simulator**: many fake expiries, each with a real max-pain strike and a "
            "**dial** for how hard the stock is dragged toward it. With the dial at **0** there's no "
            "pin (the honest null); turn it up and a pin appears. Then we check whether our test "
            "correctly says \"no pin\" at 0 and \"pin!\" when it's cranked — and whether the famous "
            "fade-to-max-pain trade makes money in either world."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, today's snapshot.** For each name, how far (in %) is spot from its max-pain "
            "strike? A real pin would cluster these bars near zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = SNAP.sort_values('gap_pct')\n"
            "    names = s['ticker'].values; gaps = s['gap_pct'].values\n"
            "else:\n"
            "    names = ['GOOGL','MSFT','XOM','AMZN','META','PFE','WMT','KO','SPY','QQQ','DIA','NVDA','AAPL','IWM','TSLA','HD','PG','JPM','BAC','AMD']\n"
            "    gaps  = [-5.0,-2.8,-2.5,-2.3,-2.2,-2.0,-1.5,-0.2,-0.3,-0.4,0.4,0.4,0.5,1.7,1.9,2.1,2.6,2.9,4.1,7.5]\n"
            "cols = [GREEN if abs(g)<=0.5 else (AMBER if abs(g)<=2 else RED) for g in gaps]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 5.2))\n"
            "ax.barh(range(len(names)), gaps, color=cols)\n"
            "ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)\n"
            "ax.axvline(0, c='k', lw=.8); ax.axvspan(-0.5, 0.5, color=GREEN, alpha=.12)\n"
            "ax.set_xlabel('spot vs max-pain strike (%)  —  0 = perfectly pinned')\n"
            "ax.set_title('If price pinned to max-pain, these bars would hug zero — most don\\'t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('median |gap|:', f\"{np.median(np.abs(gaps)):.1f}%\", '| within 1% of max-pain:', f\"{(np.abs(gaps)<=1).mean()*100:.0f}%\")"
        ),
        md(
            f"There's the first crack. The median name sits **~{R['median_gap']:.0f}%** from max-pain, "
            f"and only **{R['within_1']}%** are within 1%. The big liquid indexes (SPY, QQQ) *are* "
            "closest — but that's arbitrage keeping them efficient, the opposite of a mysterious pull. "
            "Single stocks scatter widely. Price is **not** glued to max-pain today."
        ),
        md(
            "**But a snapshot can't see the *landing*.** Pinning is about where price *ends up* at the "
            "close, and free data doesn't keep that history. So we build worlds where we *know* the "
            "truth. Here's the honest test: with the pin-dial at **zero**, does the stock drift to "
            "max-pain — or to where it started?"
        ),
        code(
            "pins = [r[0] for r in R['syn']]\n"
            "if True:\n"
            "    wins, ts = [], []\n"
            "    for p in pins:\n"
            "        ep = data.synthetic_episodes(n_episodes=400, pin=p, seed=372)\n"
            "        sm = st.summarize(ep); wins.append(sm['win_rate']*100); ts.append(sm['t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if w < 50 else GREEN for w in wins]\n"
            "ax.bar([f'pin={p:.2f}' for p in pins], wins, color=cols, width=.55)\n"
            "ax.axhline(50, ls='--', c=GREY, label='coin-flip (no pin)')\n"
            "for i,w in enumerate(wins): ax.annotate(f'{w:.0f}%',(i,w),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 90); ax.set_ylabel('% of expiries that land nearer max-pain than the start')\n"
            "ax.set_title('No pin (pin=0) → price lands nearer its START, not max-pain'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('win-rate by pin dial:', {f'{p:.2f}': f'{w:.0f}%' for p,w in zip(pins,wins)})"
        ),
        md(
            f"Read it carefully. With **no pin** (`pin=0.00`) the close lands nearer max-pain only "
            f"**{R['syn'][0][5]:.0f}%** of the time — *below* the 50% coin-flip, because without a pull "
            "the stock simply wanders away from where it began. The pin only beats a coin flip once we "
            f"**crank the dial hard** (`pin=0.40` → **{R['syn'][3][5]:.0f}%**). The test isn't fooled "
            "by geometry: it correctly says *\"no pin\"* when there's no pin."
        ),
        md(
            "**And the trade?** The believer's bet is *fade toward max-pain* — go long if max-pain is "
            "above spot, short if below, and pocket the drift. Here's what it earns across the dial."
        ),
        code(
            "fade_g = []\n"
            "for p in pins:\n"
            "    ep = data.synthetic_episodes(n_episodes=400, pin=p, seed=372)\n"
            "    fade_g.append(st.fade_trade(ep)['gross_mean']*100)\n"
            "cols = [GREY if g < 0.3 else GREEN for g in fade_g]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar([f'pin={p:.2f}' for p in pins], fade_g, color=cols, width=.55)\n"
            "for i,g in enumerate(fade_g): ax.annotate(f'{g:.2f}%',(i,g),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg gross return of the fade-to-max-pain trade (%)')\n"
            "ax.set_title('The trade only pays when a STRONG pin is planted'); \n"
            "plt.tight_layout(); plt.show()\n"
            "print('fade-trade gross by pin:', {f'{p:.2f}': f'{g:.2f}%' for p,g in zip(pins,fade_g)})"
        ),
        md(
            f"With no pin the trade earns essentially **zero** (`{R['syn'][0][7]:.2f}%`); it only turns "
            f"into real money (**+{R['syn'][3][7]:.1f}%**) when we *hand-build* a strong pull. In the "
            "real world there's no dial we can crank — and the snapshot already showed price scattered "
            "~2% off max-pain. The trade lives only in the world where we cheat."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Academics do find a *small* single-stock tendency to cling to busy "
            "strikes — so it's not nothing. But our free tape can't test the landing claim, and the "
            f"snapshot shows spot a **median ~{R['median_gap']:.0f}%** from max-pain. Real as a "
            "footnote, weak as a law.\n"
            "- **Tradability — Mirage.** The fade-to-max-pain trade only pays in a simulator with a "
            "**planted** pin; in reality there's no pull to capture, and single-name expiry trading eats "
            "spreads and borrow. No edge.\n"
            "- **\"Pinned\"? — Busted.** As a general law it's busted: absent in liquid indexes, "
            "contradicted by today's ~2% gaps, and only detectable when we build the pin ourselves."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest operational read\n\n"
            "Forget significance and ask the trader's question: even granting a *real* pin, what would "
            "you have to overcome? Here's the pull a pin must reach before it beats a coin flip — and "
            "how long it needs to act."
        ),
        code(
            "dte = [r[0] for r in R['dte']]\n"
            "wins = []\n"
            "for d in dte:\n"
            "    ep = data.synthetic_episodes(n_episodes=400, pin=0.15, seed=372, days_to_expiry=d)\n"
            "    wins.append(st.summarize(ep)['win_rate']*100)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot(dte, wins, 'o-', c=AMBER, lw=2, ms=8)\n"
            "ax.axhline(50, ls='--', c=GREY, label='coin-flip')\n"
            "for x,w in zip(dte,wins): ax.annotate(f'{w:.0f}%',(x,w),ha='center',va='bottom')\n"
            "ax.set_xlabel('days the pin has to act before expiry'); ax.set_ylabel('% landing nearer max-pain')\n"
            "ax.set_title('Even a moderate pin needs DAYS to overcome where price started'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('a moderate (pin=0.15) drag:', {f'{d}d': f'{w:.0f}%' for d,w in zip(dte,wins)})"
        ),
        md(
            "Notice the operational trap: a **moderate** pin doesn't even beat a coin flip until it's "
            "had **~a week** to work — and over that week the stock can wander anywhere. To trade it "
            "you'd need to *know* the pin is strong *and* enter early *and* survive the noise *and* pay "
            "single-name expiry spreads. The romance — \"buy the morning of expiry, collect the pin\" — "
            "is exactly the version that doesn't hold."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The calendar cousin.** [Study 195 — Monthly OpEx](../195-monthly-opex/) asks whether "
            "the expiration *week itself* carries a return — same expiry plumbing, different hook.\n"
            "- **The mechanism.** [Study 14 — Gamma Gospel](../14-gamma-gospel/) tests the dealer-gamma "
            "story the pinning legend leans on — does aggregate dealer hedging actually steer the tape?\n"
            "- **Build a real test.** Get a feed that stores expiry-day closes *and* historical open "
            "interest, match each expiry to its max-pain, and run the same paired test on the **real** "
            "tape — that's the experiment the free data can't give us.\n\n"
            "*Think max-pain pins for real? Capture historical expiry closes vs that day's max-pain, "
            "show the close landing **on** max-pain more than the spot-anchor — with a paired t past 2 — "
            "and we'll re-grade it.*"
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
            "# Max-Pain Pinning — a quantitative teardown 🔬\n"
            "### Max-pain from the option chain · a snapshot gap distribution · a planted-pin control with a "
            "paired *t* + label-shuffle placebo + spot-anchor baseline · a cost-charged fade-trade\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things the pinning legend fuses: a **real-but-modest** single-stock clustering "
            "toward high-OI strikes from the **geometry** that makes *any* close 'near a strike'. The "
            "decisive object is a **positive control**: a deterministic pinning simulator with a tunable "
            "drag, where a pin must (a) **not** appear when absent and (b) **appear** when planted. The "
            "real tape is, by construction, only a **snapshot** — yfinance keeps no expiry-day history.\n\n"
            "> ⚠️ **Data + snapshot note.** A live option chain serves only **upcoming** expiries: we can "
            "compute today's max-pain but never observe the contracts *expire*. The landing claim is "
            "therefore tested on the synthetic control, not the tape. Real data: yfinance option chains "
            "for 20 liquid US underlyings, one as-of 2026-06-22. Offline core + control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Real tape can't test the landing claim (snapshot only); spot sits a "
            f"**median {R['median_gap']:.1f}%** from max-pain ({R['within_1']}% within 1%). The pin is "
            "significant **only** in the synthetic control where we plant it (paired *t* "
            f"{R['syn'][3][4]:.1f} at pin=0.40). Literature + synthetic-only ⇒ **WEAK**, never REAL. |\n"
            f"| **Tradability** | `MIRAGE` | Fade-to-max-pain pays only with a planted pin "
            f"(**+{R['syn'][3][7]:.1f}%** gross, {R['syn'][3][9]:.0f}% hit at pin=0.40); **+{R['syn'][0][7]:.2f}%** "
            "at pin=0. No real-tape pull to capture; single-name expiry spreads + borrow finish it. |\n"
            f"| **Pinned?** | `BUSTED` | With pin=0 the paired *t* is **{R['syn'][0][4]:.1f}** and the "
            f"label-shuffle placebo *p* = **{R['syn'][0][6]:.2f}** — a shuffled max-pain pins as well as the "
            "true one. Real-as-a-footnote, busted-as-a-law. |\n\n"
            "> 💡 In plain words: the legend's grain of truth (modest single-stock clustering) does not "
            "scale into a tradable max-pain law. The only world where the test fires is the one where we "
            "hand-build the pin."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the max-pain strike be "
            "$K^\\star = \\arg\\min_s \\big[\\sum_K \\max(s-K,0)\\,\\mathrm{OI}^{c}_K + "
            "\\sum_K \\max(K-s,0)\\,\\mathrm{OI}^{p}_K\\big]$ — the settlement minimising total option "
            "payout. Let $C$ be the expiry close and $S_0$ the spot a few sessions out.\n\n"
            "- **H₁ (pinning).** $\\mathbb{E}\\,|C - K^\\star| < \\mathbb{E}\\,|C - A|$ where $A$ is the "
            "**spot-anchor** (the strike nearest $S_0$) — i.e. the close is pulled *toward* max-pain and "
            "*away from* where it started.\n"
            "- **H₂ (deployable).** A direction-correct fade $\\mathrm{sign}(K^\\star - S_0)\\cdot(C/S_0-1)$ "
            "is positive net of costs.\n"
            "- **H₃ (max-pain is special).** The attraction is to *the* max-pain strike, not to 'a "
            "central strike in general'.\n\n"
            "On the real tape H₁ is **untestable** (no expiry-day panel) and the snapshot is **against** "
            "a current pin; in the control H₁/H₂/H₃ all **fail at pin=0** and all **hold only at large "
            "planted pin**. The legend is true exactly where the literature put it (a faint single-stock "
            "clustering) and unproven exactly where it would pay (a max-pain landing law)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the baseline is the whole game\n\n"
            "Strikes form a **grid**, so every close is 'near a strike' and max-pain usually sits "
            "centrally — a naive 'is $C$ near $K^\\star$?' test fires with **zero** pinning. The honest "
            "estimand is the *paired* contrast against the **spot-anchor**:\n\n"
            "$$\\widehat{\\Delta} = \\overline{|C - A|} - \\overline{|C - K^\\star|},\\qquad "
            "t = \\frac{\\widehat{\\Delta}}{s_{\\Delta}/\\sqrt{n}}.$$\n\n"
            "Under **no** pinning, $C$ is a random walk *from* $S_0$, so it lands nearer $A$ than "
            "$K^\\star$ (whenever they differ) and $\\widehat{\\Delta} < 0$. Pinning must drag $C$ "
            "*off* $A$ and *onto* $K^\\star$ to make $\\widehat{\\Delta} > 0$. We pair this with a "
            "**label-shuffle placebo** — permute which $K^\\star$ goes with which $(C, S_0)$ — to confirm "
            "the pull is to *the* max-pain, not to centrality. That contrast, not a raw 'near a strike' "
            "count, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real snapshot.** yfinance live chains for {R['n_names']} liquid underlyings (as-of "
            f"{R['asof']}); compute $K^\\star$ per name and the gap $|S_0 - K^\\star|/S_0$. **Snapshot, "
            "not a panel** (named on the axis): the landing claim is not testable here.\n"
            "- **Synthetic episodes.** 400 episodes/grid: an OI hump → a known $K^\\star$, a spot a few "
            "days out, an expiry close evolved as a random walk **plus** a drag "
            "$\\text{drift}_t = \\text{pin}\\cdot(K^\\star - P_t)/P_t$. `pin` = daily fraction of the gap "
            "closed; **pin=0 ⇒ pure random walk** (the null).\n"
            "- **Closeness.** Per episode, $|C-K^\\star|$ vs $|C-A|$ in strike-spacings; paired Welch *t* "
            "on the difference.\n"
            "- **Placebo.** Permute $K^\\star$ across episodes; $p = \\Pr[\\text{shuffled mean dist} \\le "
            "\\text{true mean dist}]$.\n"
            "- **Costs.** Direction-correct fade with a **1-day entry lag** baked into the spot→close "
            "path and a **5-bps** one-way round-trip (shorts pay the same slippage).\n"
            "- **Positive control.** pin=0 must give *t* < 2 (no false pin) **and** placebo *p* ≈ 0.5; a "
            "large pin must give *t* ≫ 2."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The real snapshot — gaps, not pins\n\n"
            "The distribution of $|S_0 - K^\\star|/S_0$ across the 20 names. A current pin would pile the "
            "mass at zero; instead the median sits near 2%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ag = SNAP['gap_pct'].abs().values; g = st.snapshot_gap_stats(SNAP)\n"
            "    med = g['median_abs_gap']; w1 = g['frac_within_1pct']*100\n"
            "else:\n"
            "    ag = np.array([0.3,0.4,0.4,0.5,0.2,1.7,1.9,2.0,2.1,2.2,2.3,2.5,2.6,2.8,2.9,4.1,5.0,7.5,0.3,1.5])\n"
            "    med = R['median_gap']; w1 = R['within_1']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(ag, bins=np.arange(0, 8.5, 0.5), color=GREY, alpha=.85, edgecolor='white')\n"
            "ax.axvline(med, c=RED, lw=2.5, label=f'median |gap| = {med:.1f}%')\n"
            "ax.axvspan(0, 0.5, color=GREEN, alpha=.18, label='\"pinned\" zone (≤0.5%)')\n"
            "ax.set_xlabel('|spot − max-pain| / spot   (%)'); ax.set_ylabel('# of underlyings')\n"
            "ax.set_title('Where price sits vs max-pain TODAY — not on it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'median |gap| {med:.1f}% | within 1% of max-pain: {w1:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: if dealers were pinning price to max-pain *now*, this histogram would "
            f"spike at 0. It doesn't — median **~{R['median_gap']:.0f}%**, only **{R['within_1']}%** "
            "within 1%. The snapshot can't test the *landing*, but it cleanly rejects a *current* pin in "
            "the cross-section. (The tightest names — SPY/QQQ — are the most arbitraged, not the most "
            "'pinned'.)"
        ),
        md(
            "### 4b · The control — pin=0 must NOT fire, and the placebo proves it\n\n"
            "The paired *t* across the pin dial, with the *t*=2 bar. At pin=0 the *t* is **negative** "
            "(close lands nearer its start); only a strong planted pin clears the bar."
        ),
        code(
            "pins = [r[0] for r in R['syn']]\n"
            "ts, ps = [], []\n"
            "for p in pins:\n"
            "    ep = data.synthetic_episodes(n_episodes=400, pin=p, seed=372)\n"
            "    sm = st.summarize(ep); ts.append(sm['t']); ps.append(sm['p_placebo'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if t < 2 else GREEN for t in ts]\n"
            "ax.bar([f'pin\\n{p:.2f}' for p in pins], ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:.1f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('paired t  (dist→anchor − dist→max-pain)')\n"
            "ax.set_title('No false pin at pin=0; only a STRONG planted pin clears t=2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('paired t by pin:', {f'{p:.2f}': round(t,2) for p,t in zip(pins,ts)})\n"
            "print('placebo p by pin:', {f'{p:.2f}': round(pp,3) for p,pp in zip(pins,ps)})"
        ),
        md(
            f"> 💡 In plain words: at **pin=0** the paired *t* is **{R['syn'][0][4]:.1f}** (deeply "
            "negative — the close lands nearer where it *started*), and the label-shuffle placebo "
            f"*p* = **{R['syn'][0][6]:.2f}**: a *shuffled* max-pain pins exactly as well as the real one. "
            "The harness cannot invent a pin. Only at a **huge** planted drag (pin=0.40) does *t* reach "
            f"**{R['syn'][3][4]:.1f}**. H₁/H₃ both fail unless the pin is built by hand."
        ),
        md(
            "### 4c · Tradability — the fade only pays where the pin is planted\n\n"
            "The direction-correct fade-to-max-pain return (gross vs net of 5 bps) across the dial. At "
            "pin=0 it's a wash; it only becomes a 'strategy' in the cranked world."
        ),
        code(
            "g = [r[7] for r in R['syn']]; nname = 'fade'\n"
            "if HAVE_REAL or True:\n"
            "    gg, nn = [], []\n"
            "    for p in pins:\n"
            "        ep = data.synthetic_episodes(n_episodes=400, pin=p, seed=372)\n"
            "        f = st.fade_trade(ep); gg.append(f['gross_mean']*100); nn.append(f['net_mean']*100)\n"
            "x = np.arange(len(pins))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, gg, .4, color=GREEN, label='gross')\n"
            "ax.bar(x+.2, nn, .4, color=GREY, label='net @ 5bps')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'pin={p:.2f}' for p in pins]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg fade-trade return (%)')\n"
            "ax.set_title('Fade-to-max-pain: ~0 with no pin, pays only with a planted one'); ax.legend()\n"
            "for i,(a,b) in enumerate(zip(gg,nn)):\n"
            "    ax.annotate(f'{a:.2f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('fade gross/net by pin:', {f'{p:.2f}': (round(a,2),round(b,2)) for p,a,b in zip(pins,gg,nn)})"
        ),
        md(
            f"> 💡 In plain words: at pin=0 the fade earns **{R['syn'][0][7]:.2f}%** gross / "
            f"**{R['syn'][0][8]:.2f}%** net — a coin flip. Costs barely matter (a 5-bps round-trip on a "
            "few-day hold is tiny) — the binding constraint is that **there's nothing to fade** unless a "
            "strong pin exists, which the real snapshot says it doesn't. Tradability is a mirage."
        ),
        md(
            "### 4d · False-positive stress — more data never fakes a pin\n\n"
            "The deepest check: at pin=0, does piling on episodes ever *flip* the test into a false "
            "positive? No — the *t* gets **more** negative, the absence of a pin detected ever more "
            "confidently."
        ),
        code(
            "ns = [r[0] for r in R['fp']]\n"
            "tt = []\n"
            "for n in ns:\n"
            "    ep = data.synthetic_episodes(n_episodes=n, pin=0.0, seed=372)\n"
            "    tt.append(st.summarize(ep)['t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot(ns, tt, 'o-', c=RED, lw=2, ms=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axhline(2, ls='--', c=GREY, label='t=2 (would be a false pin)')\n"
            "for x,t in zip(ns,tt): ax.annotate(f't={t:.1f}',(x,t),ha='center',va='top')\n"
            "ax.set_xlabel('# episodes (pin=0)'); ax.set_ylabel('paired t')\n"
            "ax.set_title('At pin=0, more data → MORE confident \"no pin\" (never a false positive)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pin=0 paired t by sample size:', {n: round(t,2) for n,t in zip(ns,tt)})"
        ),
        md(
            "> 💡 In plain words: a broken test would drift toward 'significant' as $n$ grows (that's how "
            "spurious findings are manufactured). Ours does the opposite — the no-pin world is rejected "
            "as a pin **more** strongly with more data, because the truth (no pin) is being measured ever "
            "more precisely. The engine is honest; the verdict is the absence of a real-tape pin, not a "
            "weak measurement of one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the literature (Ni–Pearson–Poteshman 2005; Avellaneda–Lipkin 2003) "
            "documents a *modest* single-stock clustering toward high-OI strikes, so the effect isn't "
            "nothing. But the real tape can't test the landing claim (snapshot only), the snapshot shows "
            f"spot a **median {R['median_gap']:.1f}%** from max-pain, and significance appears **only** in "
            "the planted-pin control. Literature + synthetic-only support ⇒ WEAK, never REAL.\n"
            f"- **Tradability `MIRAGE`** — fade-to-max-pain earns **+{R['syn'][0][7]:.2f}%** at pin=0 and "
            f"only **+{R['syn'][3][7]:.1f}%** with a strong planted pin. No real-tape pull, and single-name "
            "expiry spreads/borrow would finish a live version. No NAV-scale edge.\n"
            f"- **Pinned? `BUSTED`** — pin=0 gives paired *t* **{R['syn'][0][4]:.1f}** and placebo "
            f"*p* **{R['syn'][0][6]:.2f}** (shuffled max-pain pins as well as the true one); price sits "
            "~2% off max-pain in the cross-section. A real-as-a-footnote tendency sold as a law."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — pinning needs time it rarely gets\n\n"
            "Even granting a *real* moderate pin, the operational killer is **time**: the drag must "
            "overcome where price started, and that takes days the trader doesn't reliably have. Here, "
            "the win-rate of a fixed moderate pin (pin=0.15) as a function of how long it can act."
        ),
        code(
            "dte = [r[0] for r in R['dte']]\n"
            "wins, ts2 = [], []\n"
            "for d in dte:\n"
            "    ep = data.synthetic_episodes(n_episodes=400, pin=0.15, seed=372, days_to_expiry=d)\n"
            "    sm = st.summarize(ep); wins.append(sm['win_rate']*100); ts2.append(sm['t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(dte, wins, 'o-', c=AMBER, lw=2, ms=9, label='win-rate')\n"
            "ax.axhline(50, ls='--', c=GREY, label='coin-flip')\n"
            "for x,w,t in zip(dte,wins,ts2): ax.annotate(f'{w:.0f}%\\nt={t:.1f}',(x,w),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xlabel('days the pin has to act'); ax.set_ylabel('% landing nearer max-pain')\n"
            "ax.set_title('A moderate pin only beats a coin flip after ~a week'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pin=0.15 win-rate by days-to-expiry:', {f'{d}d': f'{w:.0f}%' for d,w in zip(dte,wins)})"
        ),
        md(
            f"> 💡 In plain words: at 3 days a moderate pin loses to a coin flip "
            f"(**{R['dte'][0][2]:.0f}%**, *t*={R['dte'][0][1]:.1f}); it only edges ahead by ~10 days "
            f"(**{R['dte'][2][2]:.0f}%**). The folk version — *enter the morning of expiry and collect "
            "the pin* — is the worst case: too little time, too much noise, real spreads. There is no "
            "entry, sizing, or cost regime that turns a faint, slow, single-name clustering into a "
            "NAV-scale strategy. **The story's drama is exactly its undoing.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The expiry-week cousin.** [Study 195 — Monthly OpEx](../195-monthly-opex/): does the "
            "expiration week carry a return effect distinct from this strike-level pinning question?\n"
            "- **The mechanism under the hood.** [Study 14 — Gamma Gospel](../14-gamma-gospel/): the "
            "dealer-gamma hedging that the pinning story invokes — does it actually move the tape?\n"
            "- **The test the free tape can't run.** With a feed that stores historical expiry-day "
            "closes *and* per-strike open interest, match each expiry to its max-pain and run this exact "
            "paired test on the **real** panel (single names vs indexes, monthly vs weekly). The "
            "literature predicts a faint single-stock effect — quantify it with a HAC/clustered *t* and "
            "a multiple-testing correction across names.\n\n"
            "*The reproducible core is offline and deterministic; the real tape is an explicit snapshot. "
            "Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
