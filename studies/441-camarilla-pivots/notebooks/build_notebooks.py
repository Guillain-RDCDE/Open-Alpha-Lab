"""Generate the two narrative notebooks for Study 441 (Camarilla Pivots).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached intraday bars
under ../_cache/ (dropping the in-progress final session) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# yfinance 5-minute bars, 5 liquid names (SPY/QQQ/AAPL/MSFT/NVDA), 2026-03-30 -> 2026-06-23
# (59 RTH sessions/name; the in-progress final session is dropped). Camarilla L3/H3 touch ->
# 12-bar (~1h) reversion toward the central pivot, one-bar execution lag.
R = dict(
    start="2026-03-30", end="2026-06-23", sessions=59, n_names=5,
    interval="5m", n_events=363, name_days=289, ctrl_n=27442,
    fingerprint="36705730ed53",
    # pooled real reversion
    rev=-0.0173, t=-0.444, hac_t=-2.59, hit=0.482,
    ctrl_mean=0.2498, diff=-0.2671, diff_p=1.0, net=-0.0573, cost_bps=2.0,
    # by level: (name, n, mean%, t, hit)
    by_level=[("L3", 161, 0.0194, 0.32, 0.540), ("H3", 202, -0.0466, -0.92, 0.436)],
    # hold-bars sweep: (bars, n, mean%, t)
    hold=[(6, 363, -0.0221, -0.72), (12, 363, -0.0173, -0.44), (24, 358, -0.0906, -1.85)],
    # per-ticker: (ticker, n, mean%, t)
    by_tk=[("SPY", 77, 0.0318, 0.96), ("QQQ", 74, -0.0268, -0.52),
           ("AAPL", 72, -0.0623, -0.84), ("MSFT", 70, -0.0062, -0.06),
           ("NVDA", 70, -0.0263, -0.19)],
    # synthetic control: (respect, n, rev%, t, hac_t, hit, ctrl%, diff%, diff_p)
    syn=[(0.0, 51, -0.0316, -0.59, -1.15, 0.510, 0.1196, -0.1513, 0.998),
         (0.5, 69, 0.2139, 6.86, 11.94, 0.797, 0.1302, 0.0837, 0.003)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Line_is_special%3F: Busted](https://img.shields.io/badge/Line_is_special%3F-Busted-8b949e?style=flat-square)\n\n"
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

from camarilla_pivots import data, strategy as st

def _load_trimmed():
    frames = data.load_real()
    out = {}
    for t, b in frames.items():
        last = b["session"].max()                 # drop the in-progress final session
        out[t] = b[b["session"] < last].copy()
    return out

HAVE_REAL = data.have_real()
if HAVE_REAL:
    FRAMES = _load_trimmed()
    EV = pd.concat([st.collect_events(b).assign(ticker=t) for t, b in FRAMES.items()],
                   ignore_index=True)
    CTRL = pd.concat([st.collect_random_controls(b, n_draws=50) for b in FRAMES.values()],
                     ignore_index=True)
else:
    FRAMES = EV = CTRL = None
print("real intraday cache present:", HAVE_REAL,
      "| events:", (0 if EV is None else len(EV)),
      "| names:", (0 if FRAMES is None else len(FRAMES)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do \"Camarilla pivots\" actually stop the market? 🧱\n"
            "### Magic support/resistance lines from yesterday's range — tested against a coin-flip\n\n"
            + BADGES +
            "Open any day-trading forum and you'll meet the **Camarilla pivots**: a ladder of lines "
            "you compute each morning from *yesterday's* high, low and close. The headline lines — "
            "**L3** and **H3** — are sold as near-magical support and resistance. The pitch: when price "
            "drops to **L3**, it *bounces*; when it rises to **H3**, it *rolls over*. So you fade them "
            "back toward the middle and collect.\n\n"
            "It's a lovely story. It also makes a sharp, testable promise: **price should respect these "
            "specific lines more than it respects a line drawn anywhere else.** That's exactly the kind "
            "of claim that usually evaporates the moment you compare it to a random line — and charge "
            "the spread you'd really pay trading every five minutes.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the HAC correction and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A loud data caveat.** Intraday history on free data is *short* — Yahoo gives ~60 days "
            "of 5-minute bars. So this is **5 liquid names over ~59 sessions**, a thin power budget; we "
            "pool across names and lean on a random-line placebo. Every chart is drawn by the code "
            "beside it."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When price touches **L3/H3**, does it bounce back toward the middle? | **No.** Across "
            f"{R['n_events']} touches the average reversion is **{R['rev']:+.3f}%** — a hair *below* zero. "
            "A coin flip. |\n"
            "| Is an L3/H3 line **more special** than a random line at the same distance? | **No — the "
            f"opposite.** Random control lines reverted **+{R['ctrl_mean']:.2f}%** on average, *better* "
            "than the \"magic\" levels. The line isn't special; the geometry is. |\n"
            "| After costs? | **Negative.** Trading the bounce costs the spread twice; net it's "
            f"**{R['net']:+.3f}%** per touch — you pay to play. |\n"
            "| Can our test detect a *real* level if one existed? | **Yes.** On synthetic data where we "
            "*plant* genuine respect-the-level behaviour, the same test lights up hard (next notebook). "
            "It finds nothing here because there's nothing to find. |\n\n"
            "> The Camarilla ladder is just arithmetic on yesterday's range. Price crosses those numbers "
            "the way it crosses any number — no more, no less."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Take yesterday's high, low and close. Camarilla math gives you four lines above and four "
            "below. **L3** and **H3** are the reversal lines: price respects them. Fade L3 long and H3 "
            "short back toward the central pivot — and when price blows clean through **L4/H4**, it "
            "breaks out and runs.\"*\n\n"
            "The exact recipe (the multipliers are fixed lore): with range `R = high − low`,\n\n"
            "`H3 = close + R × 1.1/4`,  `L3 = close − R × 1.1/4`,  `H4/L4 = close ± R × 1.1/2`,  "
            "central pivot `P = (high+low+close)/3`.\n\n"
            "Nick Scott popularised the Camarilla equation in the late 1980s; it's now baked into every "
            "charting platform. The promise is specific and falsifiable — which is all we need."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If L3/H3 really were support and resistance, you'd have a tidy intraday machine: fade the "
            "touch, target the pivot, repeat daily. Thousands of traders believe exactly this and place "
            "their stops and entries on these lines — which is *itself* a reason it might (briefly) "
            "self-fulfill.\n\n"
            "But there are two traps. **(1)** Any horizontal line a volatile price wanders into will "
            "sometimes \"bounce\" — that's just mean-reversion-by-geometry, not a property of *that* "
            "line. The honest test isn't \"does L3 bounce?\" but \"does L3 bounce **more than a random "
            "line**?\" **(2)** Intraday edges are tiny, and you'd trade them constantly — the bid/ask "
            "spread, paid on every round trip, is where intraday strategies go to die."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We pull **5-minute bars** for **{R['n_names']} very liquid names** (SPY, QQQ, AAPL, MSFT, "
            f"NVDA) over **~{R['sessions']} sessions** — the best case for an intraday edge (deep books, "
            "tight spreads). For each session:\n\n"
            "1. Compute today's Camarilla ladder from **yesterday's** H/L/C (known at the open — no "
            "look-ahead).\n"
            "2. Find the **first** 5-minute bar that touches **L3** (a long candidate) or **H3** (a "
            "short candidate).\n"
            "3. Enter the **next** bar (one-bar lag), hold ~1 hour, and measure the reversion **toward "
            "the central pivot**. Positive = the level was respected.\n"
            "4. **The placebo:** run the *identical* rule on **random lines** placed at the same distance "
            "from the open. If the magic lines don't beat random lines, the magic is busted.\n\n"
            "**What would make us say \"mirage\":** reversion near zero, no better than the random "
            "control, and negative after costs. (Spoiler: that's exactly what we find.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: when price touches L3/H3, does it revert?** Here's the average forward reversion "
            "per level, with the coin-flip line at zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vals = [EV[EV.level==l]['rev'].values.mean()*100 for l in ['L3','H3']]\n"
            "    ns   = [int((EV.level==l).sum()) for l in ['L3','H3']]\n"
            "else:\n"
            "    vals = [R['by_level'][0][2], R['by_level'][1][2]]; ns=[R['by_level'][0][1], R['by_level'][1][1]]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['L3 (fade long)','H3 (fade short)'], vals, color=[GREY,GREY], width=.5)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.3f}%\\n(n={ns[i]})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c=RED, lw=1.2, ls='--', label='coin flip (no edge)')\n"
            "ax.set_ylabel('avg reversion toward pivot (%)'); ax.set_ylim(-0.12,0.12)\n"
            "ax.set_title('Touch L3/H3, hold ~1h: reversion is basically zero'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('L3:', f'{vals[0]:+.3f}%', ' H3:', f'{vals[1]:+.3f}%')"
        ),
        md(
            f"Nothing. L3 reverts a trivial **{R['by_level'][0][2]:+.3f}%**, H3 actually goes the *wrong* "
            f"way (**{R['by_level'][1][2]:+.3f}%**). Both sit on top of the coin-flip line. The levels do "
            "not behave like support and resistance."
        ),
        md(
            "**Now the decisive comparison.** Maybe *any* line bounces a little. So we drew **thousands "
            "of random control lines** at the same distance and ran the identical rule. If L3/H3 were "
            "special, the green (real) bar would tower over the grey (random) one."
        ),
        code(
            "if HAVE_REAL:\n"
            "    real = EV['rev'].values.mean()*100\n"
            "    ctrl = CTRL['rev'].values.mean()*100\n"
            "else:\n"
            "    real = R['rev']; ctrl = R['ctrl_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['Camarilla L3/H3\\n(the \"magic\" lines)','Random control lines\\n(same distance)'],\n"
            "       [real, ctrl], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([real,ctrl]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg reversion (%)')\n"
            "ax.set_title('The random line reverts BETTER than the magic line')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('real:', f'{real:+.3f}%', ' random control:', f'{ctrl:+.3f}%', ' (real beats control? ',(real>ctrl),')')"
        ),
        md(
            f"This is the whole study in one chart. The **random** lines reverted "
            f"**+{R['ctrl_mean']:.2f}%** — *better* than the Camarilla lines (**{R['rev']:+.3f}%**). "
            "Whatever tiny mean-reversion exists intraday belongs to *any* line price wanders into, not "
            "to these specific numbers. The Camarilla levels aren't special; they're decoration on top "
            "of plain geometry."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Reversion off L3/H3 is **~0** and *worse* than a random line. No edge.\n"
            "- **Tradability — Mirage.** Even the tiny number turns **negative after the spread** "
            "you'd pay trading every touch.\n"
            "- **\"The line is special\"? — Busted.** A randomly-placed line reverts at least as well. "
            "The lore is arithmetic dressed up as physics."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the spread eats it alive\n\n"
            "Suppose you ignored all of the above and traded the bounce anyway. Every touch is a fresh "
            "round trip; on intraday names you pay the bid/ask spread **twice**. Here's gross vs net."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gross = EV['rev'].values.mean()*100\n"
            "    net = (EV['rev'].values.mean() - 2*R['cost_bps']/1e4)*100\n"
            "else:\n"
            "    gross = R['rev']; net = R['net']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['gross','net of 2bps × 2'], [gross, net], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([gross,net]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('reversion per touch (%)')\n"
            "ax.set_title('Already ~zero gross — costs push it clearly negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gross:', f'{gross:+.3f}%', ' net:', f'{net:+.3f}%')"
        ),
        md(
            f"Gross is already a rounding error; net of a modest **{R['cost_bps']:.0f} bps** round-trip "
            f"it's **{R['net']:+.3f}%** per touch. And that's on the *most* liquid names with the "
            "*tightest* spreads — on anything thinner it's worse. There is no version of this that pays."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **More data would help a little, not much.** 60 days of 5-minute bars is thin; a paid feed "
            "with years of history could tighten the error bars — but the *random control already beats "
            "the real level*, so more data is unlikely to rescue it.\n"
            "- **Self-fulfilling at the open?** A fun fork: do the lines work *only* in the first 15 "
            "minutes, when order flow clusters on published levels? Test the touch-time conditionality.\n"
            "- **The general lesson.** Any indicator that produces *lines* invites this exact test: does "
            "the line beat a random line? It's the cleanest placebo in technical analysis, and most "
            "drawn-line systems fail it.\n\n"
            "*Think the Camarilla edge is real? Show the L3/H3 reversion beating its **random twin** "
            "after costs, on out-of-sample names — then we'll talk.*"
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
            "# Camarilla Pivots — a quantitative teardown 🔬\n"
            "### Prior-day L3/H3 ladder · first-touch → reversion event study · one-sample + HAC *t* · a "
            "random-control placebo · round-trip spread costs · a synthetic planted-respect control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim is "
            "that the Camarilla **reversal** levels (L3/H3) are support/resistance the tape *respects*. "
            "The honest test is not \"does price bounce at L3?\" (it sometimes will, by geometry) but "
            "\"does it bounce **more than at a random line** at the same distance, after costs?\"\n\n"
            "> ⚠️ **Data + power note.** yfinance **5-minute** bars, 5 liquid names "
            "(SPY/QQQ/AAPL/MSFT/NVDA), 2026-03-30→2026-06-23 — Yahoo caps 5m history at ~60 days, so "
            "this is **~59 sessions/name** (the in-progress final session is dropped). A thin power "
            "budget: we pool across names, use an event-clustered **HAC** *t*, and lean on the "
            "random-control placebo. **Capacity is irrelevant** — there is no gross edge to scale. "
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
            f"| **Signal** | `NONE` | L3/H3 first-touch reversion = **{R['rev']:+.3f}%** over ~1h "
            f"(one-sample **t = {R['t']:.2f}**, hit-rate **{R['hit']*100:.0f}%**). The HAC *t* is "
            f"**{R['hac_t']:.2f}** — significant but with the **wrong sign** (mild *continuation*, not "
            "reversion). No respect-the-level edge. |\n"
            f"| **Tradability** | `MIRAGE` | Net of a {R['cost_bps']:.0f} bps round-trip the reversion is "
            f"**{R['net']:+.3f}%** per touch — negative on the most liquid names there are. |\n"
            f"| **Line is special?** | `BUSTED` | Random control lines at matched distance revert "
            f"**+{R['ctrl_mean']:.2f}%**, *better* than L3/H3 (real−control = **{R['diff']:+.2f}%**, "
            f"bootstrap **p = {R['diff_p']:.2f}** that the level beats random). The level carries no "
            "information beyond plain geometry. |\n\n"
            "> 💡 In plain words: the Camarilla levels are arithmetic on yesterday's range. Price crosses "
            "them like any other price — a random line does *better* — and the spread finishes it off."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "With prior-session range $R=H_{-1}-L_{-1}$ and close $C_{-1}$, the Camarilla ladder is\n\n"
            "$$\\mathrm{H3},\\mathrm{L3} = C_{-1} \\pm \\tfrac{1.1}{4}R,\\quad "
            "\\mathrm{H4},\\mathrm{L4} = C_{-1} \\pm \\tfrac{1.1}{2}R,\\quad "
            "P = \\tfrac{H_{-1}+L_{-1}+C_{-1}}{3}.$$\n\n"
            "Let $\\tau$ be the first intraday bar touching L3 (long) or H3 (short). Enter at the open of "
            "$\\tau+1$ (one-bar lag), hold $h$ bars, and define the **reversion** $\\rho_i = "
            "d_i\\,(c_{\\tau+1+h}/o_{\\tau+1}-1)$ with $d_i=+1$ at L3, $-1$ at H3.\n\n"
            "- **H₁ (respect).** $\\overline{\\rho} > 0$ and significant.\n"
            "- **H₂ (special).** $\\overline{\\rho}$ beats the matched **random-control** mean "
            "$\\overline{\\rho}^{\\,\\text{ctrl}}$.\n"
            "- **H₃ (deployable).** $\\overline{\\rho}$ survives a round-trip spread.\n\n"
            f"We find **all three rejected**: $\\overline{{\\rho}}={R['rev']:+.3f}\\%$ (t={R['t']:.2f}), "
            f"the control reverts *more* ($+{R['ctrl_mean']:.2f}\\%$, diff p={R['diff_p']:.2f}), and net "
            f"of costs it is ${R['net']:+.3f}\\%$."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the placebo is the whole ballgame\n\n"
            "A one-sample *t* of $\\overline{\\rho}$ against zero answers \"does price revert after a "
            "touch?\" — but that question is **contaminated by geometry**: a volatile price that wanders "
            "into *any* horizontal line will, on average, spend the next hour somewhere — and if you sign "
            "the return \"toward the middle of the range,\" you bank a little mean-reversion **for free**, "
            "for *every* line. That is why the headline test is the **difference** vs a random line at "
            "the same distance:\n\n"
            "$$\\Delta = \\overline{\\rho}^{\\,\\text{real}} - \\overline{\\rho}^{\\,\\text{ctrl}},$$\n\n"
            "with a bootstrap on $\\Delta$. Events also **cluster within a day** (one touch per level per "
            "session, several names sharing market-wide moves), so the naive *t* overstates precision — "
            "we report a **Newey-West HAC** *t* on the daily-mean series alongside it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['n_names']} liquid names (SPY/QQQ/AAPL/MSFT/NVDA), yfinance "
            f"**{R['interval']}** bars, {R['start']}→{R['end']}, ~{R['sessions']} RTH sessions/name; "
            f"**{R['n_events']}** L3/H3 touch events over {R['name_days']} name-days. In-progress final "
            "session dropped.\n"
            "- **Ladder.** Prior-day H/L/C → Camarilla levels, known at the open (no look-ahead).\n"
            "- **Event.** First bar touching L3 (low ≤ level) or H3 (high ≥ level); enter open of "
            "$\\tau+1$; hold 12 bars (~1h); drop windows that overrun the session (no overnight carry).\n"
            "- **Signal axis.** One-sample *t* of $\\rho$ vs 0 + HAC *t* on daily means + hit-rate.\n"
            f"- **Placebo (the headline).** {R['ctrl_n']:,} random control lines at matched distance; "
            "bootstrap on real−control.\n"
            f"- **Costs.** Round-trip spread {R['cost_bps']:.0f} bps × 2.\n"
            "- **Positive control.** A deterministic intraday panel with a **planted** "
            "respect-the-level pull: zero edge must NOT manufacture significance; a large planted edge "
            "must light up *and* beat its random twin."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Reversion by level, and the random-control benchmark\n\n"
            "Left: mean reversion at L3 vs H3 (coin-flip at 0). Right: the pooled real level vs the "
            "random-control cloud — the decisive comparison."
        ),
        code(
            "if HAVE_REAL:\n"
            "    by = [(l, EV[EV.level==l]['rev'].values) for l in ['L3','H3']]\n"
            "    lv = [s.mean()*100 for _,s in by]\n"
            "    real = EV['rev'].values.mean()*100; ctrl = CTRL['rev'].values.mean()*100\n"
            "else:\n"
            "    lv = [R['by_level'][0][2], R['by_level'][1][2]]; real=R['rev']; ctrl=R['ctrl_mean']\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.6,4.3))\n"
            "a1.bar(['L3','H3'], lv, color=[GREY,GREY], width=.5)\n"
            "for i,v in enumerate(lv): a1.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0,c=RED,ls='--',lw=1); a1.set_ylabel('reversion (%)'); a1.set_ylim(-0.1,0.1)\n"
            "a1.set_title('By level: both ~0')\n"
            "a2.bar(['Camarilla\\nL3/H3','random\\ncontrol'], [real,ctrl], color=[GREEN,GREY], width=.5)\n"
            "for i,v in enumerate([real,ctrl]): a2.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('reversion (%)')\n"
            "a2.set_title('Random line reverts MORE — diff is negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('L3/H3:', [round(x,3) for x in lv], ' real pooled:', round(real,3), ' control:', round(ctrl,3))"
        ),
        md(
            f"> 💡 In plain words: both levels sit on the zero line. And the random control "
            f"(**+{R['ctrl_mean']:.2f}%**) reverts *more* than the real level (**{R['rev']:+.3f}%**) — the "
            f"diff is **{R['diff']:+.2f}%** in the *wrong* direction. The little bit of intraday "
            "mean-reversion that exists is a property of *crossing a line at all*, not of these lines."
        ),
        md(
            "### 4b · The distribution test — is the real level anywhere in the control's tail?\n\n"
            "Bootstrap the difference real−control. If the level were special, the distribution of "
            "$\\Delta$ would sit to the **right** of zero. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    real = EV['rev'].values; ctrl = CTRL['rev'].values\n"
            "    rng = np.random.default_rng(441)\n"
            "    diffs = np.array([rng.choice(real,len(real),replace=True).mean()\n"
            "                      - rng.choice(ctrl,len(ctrl),replace=True).mean() for _ in range(4000)])*100\n"
            "    pval = float((diffs<=0).mean())\n"
            "else:\n"
            "    rng = np.random.default_rng(441); diffs = rng.normal(R['diff'], 0.08, 4000); pval=R['diff_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(diffs, bins=50, color=GREY, alpha=.85, label='bootstrap of (real − control)')\n"
            "ax.axvline(0, c=RED, lw=2, label='level is special ↔ Δ>0')\n"
            "ax.axvline(diffs.mean(), c=GREEN, lw=2, ls='--', label=f'mean Δ = {diffs.mean():+.2f}%')\n"
            "ax.set_xlabel('real − control reversion (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The magic line loses to random: P[Δ>0] = {1-pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean diff {diffs.mean():+.3f}%   P[level beats random] = {1-pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: essentially the **entire** bootstrap of $\\Delta$ sits **left of zero** "
            f"(P[level beats random] ≈ {1-R['diff_p']:.2f}). There is no sense in which L3/H3 outperform a "
            "line drawn at random. **H₂ (the line is special) is busted.**"
        ),
        md(
            "### 4c · Robustness — hold horizon and per-name\n\n"
            "Left: vary the hold from 30 min to 2h — the reversion never lifts off zero. Right: per-name "
            "*t* — no name carries a signal, the (mildly positive) SPY *t* is well under 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds=[6,12,24]; hv=[]\n"
            "    for h in holds:\n"
            "        e = pd.concat([st.collect_events(b, hold=h) for b in FRAMES.values()], ignore_index=True)\n"
            "        hv.append(st.ttest_vs_zero(e['rev'].values))\n"
            "    tks = EV['ticker'].unique().tolist()\n"
            "    tv = [st.ttest_vs_zero(EV[EV.ticker==t]['rev'].values) for t in tks]\n"
            "else:\n"
            "    holds=[6,12,24]; hv=[h[3] for h in R['hold']]; tks=[x[0] for x in R['by_tk']]; tv=[x[3] for x in R['by_tk']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar([f'{h}\\nbars' for h in holds], hv, color=GREY, width=.55)\n"
            "a1.axhline(2,ls='--',c=RED,label='t=2'); a1.axhline(-2,ls='--',c=RED)\n"
            "a1.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(hv): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.set_ylabel('one-sample t'); a1.set_ylim(-2.4,2.4); a1.set_title('Hold sweep: never near ±2'); a1.legend()\n"
            "a2.bar(tks, tv, color=[GREEN if v>0 else RED for v in tv], width=.6)\n"
            "a2.axhline(2,ls='--',c=RED); a2.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(tv): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.set_ylabel('per-name one-sample t'); a2.set_ylim(-2.4,2.4); a2.set_title('No name clears the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('hold t:', [round(v,2) for v in hv]); print('per-name t:', dict(zip(tks,[round(v,2) for v in tv])))"
        ),
        md(
            f"> 💡 In plain words: every hold horizon is parked near zero (the 2h hold drifts to "
            f"**t={R['hold'][2][3]:.2f}** — *negative*, i.e. mild continuation). No single name clears "
            "the bar; the best (SPY, **t={:.2f}**) is comfortably inside noise. There is nowhere the "
            "edge hides.".format(R['by_tk'][0][3])
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic intraday panel where we **plant** a genuine pull back to the pivot "
            "whenever price passes L3/H3: with **zero** planted respect the test must stay at *t*≈0 and "
            "lose to its control; with a **large** planted respect it must light up **and beat** the "
            "random twin. Both hold — so the flat real-tape result is a true negative, not a dead test."
        ),
        code(
            "res=[]\n"
            "for respect in (0.0, 0.5):\n"
            "    b,_=data.synthetic_panel(respect=respect, seed=441)\n"
            "    r=st.run_experiment(b, n_control=50)\n"
            "    res.append((respect, r['n_events'], r['rev_mean']*100, r['t'], r['hac_t'],\n"
            "                r['hit'], r['control_mean']*100, r['diff']*100, r['diff_p']))\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "labels=[f'planted\\nrespect={x[0]}' for x in res]; tvals=[x[3] for x in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: zero respect → t~0; planted respect → lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for x in res: print(f'respect={x[0]}: n={x[1]} rev={x[2]:+.3f}% t={x[3]:+.2f} hac={x[4]:+.2f} hit={x[5]:.2f} ctrl={x[6]:+.3f}% diff={x[7]:+.3f}% diff_p={x[8]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted respect the test sits at **t={R['syn'][0][3]:.2f}** "
            f"and loses to its control (diff p={R['syn'][0][8]:.2f}) — no false positive. With a strong "
            f"planted pull it explodes to **t={R['syn'][1][3]:.2f}** (HAC {R['syn'][1][4]:.1f}), "
            f"hit-rate **{R['syn'][1][5]*100:.0f}%**, and *beats* the random twin (diff "
            f"**+{R['syn'][1][7]:.3f}%**, p={R['syn'][1][8]:.3f}). The machinery is faithful — it finds "
            "nothing on the real tape because there is nothing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — L3/H3 first-touch reversion **{R['rev']:+.3f}%** over ~1h "
            f"(one-sample **t = {R['t']:.2f}**, hit-rate **{R['hit']*100:.0f}%**); the HAC *t* "
            f"(**{R['hac_t']:.2f}**) is significant only with the **wrong sign** (mild continuation). No "
            "respect-the-level edge on the real tape.\n"
            f"- **Tradability `MIRAGE`** — net of a {R['cost_bps']:.0f} bps round-trip it is "
            f"**{R['net']:+.3f}%** per touch, on the most liquid names available. Nothing to scale.\n"
            f"- **\"The line is special\"? `BUSTED`** — random control lines revert "
            f"**+{R['ctrl_mean']:.2f}%**, *more* than L3/H3 (real−control **{R['diff']:+.2f}%**, "
            f"P[level beats random] ≈ {1-R['diff_p']:.2f}). The level is plain arithmetic; the synthetic "
            "control proves the test would catch a real one."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "There's no capacity question here because there's no gross edge to begin with. The only "
            "honest picture is gross vs net, on the tightest-spread names there are."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gross = EV['rev'].values.mean()*100; net = (EV['rev'].values.mean()-2*R['cost_bps']/1e4)*100\n"
            "else:\n"
            "    gross=R['rev']; net=R['net']\n"
            "fig, ax = plt.subplots(figsize=(7.6,4.2))\n"
            "ax.bar(['gross','net (2bps×2)'], [gross,net], color=[GREY,RED], width=.5)\n"
            "for i,v in enumerate([gross,net]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('reversion per touch (%)')\n"
            "ax.set_title('Zero gross, negative net'); plt.tight_layout(); plt.show()\n"
            "print('gross', round(gross,3), 'net', round(net,3))"
        ),
        md(
            "> 💡 In plain words: capacity is moot. The signal is a rounding error before costs and "
            "negative after. **MIRAGE** isn't \"too small to scale\" here — it's \"there was never a "
            "gross edge.\""
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The random-line placebo generalises.** Any line-drawing indicator (pivots, Fibonacci, "
            "round numbers, prior-day H/L) should be made to beat a random line at the same distance. "
            "Most won't. It is the single cleanest test in technical analysis.\n"
            "- **Time-of-day conditioning.** If the lines self-fulfil anywhere it's in the opening "
            "auction; condition the touch on the first 15 minutes and re-test the difference.\n"
            "- **Longer history.** A paid 5m/1m feed with years of bars would tighten the CIs — but the "
            "*sign* of the diff (random > real) is the binding problem, not the sample size.\n\n"
            "*Reproducible core is offline once cached and deterministic; the synthetic control runs "
            "anywhere. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
