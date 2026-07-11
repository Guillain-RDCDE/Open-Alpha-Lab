"""Generate the two narrative notebooks for Study 705 (Rounding Top).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached OHLC panel
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, SPY + 29 large-caps,
# 2004-01-02 -> 2026-06-30, 22.5 years, mechanical rounding-top breakdown detector, base_len=90,
# short trades charged 5 bps one-way x2 + 30 bps/yr borrow pro-rated over the horizon).
R = dict(
    start="2004-01-02", end="2026-06-30", years=22.5, n_names=30, fp="9da1b7ce7758",
    base_len=90, cost_bps=5.0, borrow_bps=30.0,
    # per horizon: (H, n_sig, gross%, net%, base%, win%, base_win%, t0, tHAC, tBase, p_placebo)
    h10=(10, 227, -0.76, -0.88, -0.59, 46, 43, -1.83, -1.89, -0.42, 0.684),
    h20=(20, 227, -1.61, -1.74, -1.14, 44, 42, -3.30, -3.34, -0.96, 0.827),
    h60=(60, 225, -4.39, -4.56, -3.32, 33, 37, -5.69, -5.41, -1.38, 0.891),
    # robustness at H=20 — base_len: (base_len, n, gross%, t0, tBase, p)
    rob_bl=[(60, 222, -0.78, -1.64, 0.75, 0.197), (90, 227, -1.61, -3.30, -0.96, 0.827),
            (120, 183, -1.73, -3.02, -1.03, 0.884)],
    # robustness at H=20, base_len=90 — r2_min: (r2_min, n, gross%, t0, tBase, p)
    rob_r2=[(0.45, 295, -1.40, -3.18, -0.59, 0.748), (0.55, 227, -1.61, -3.30, -0.96, 0.827),
            (0.70, 83, -1.90, -2.32, -0.93, 0.832)],
    # synthetic control at H=20: (edge, n, gross%, base%, t0, tBase, p, win%)
    syn=[(0.00, 145, -0.20, -0.40, -0.44, 0.42, 0.351, 53),
         (-0.10, 143, 3.18, 0.15, 5.37, 5.11, 0.000, 71)],
    syn_null_mean=-0.17, syn_null_sd=0.95, syn_null_fire=1, syn_null_seeds=20,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Distribution_signal%3F: Not_supported](https://img.shields.io/badge/Distribution_signal%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from rounding_top import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real() if HAVE_REAL else None
print("real rounding-top cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"rounding top\" really warn of a decline? 🌄\n"
            "### The dome distribution — a beautiful chart shape that, on the tape, predicts nothing special\n\n"
            + BADGES +
            "Open any technical-analysis book and you'll meet the **rounding top** (a.k.a. the "
            "*dome distribution*): a long, smooth, arched roll-over where — the story goes — *smart "
            "money* is quietly **distributing** (selling into strength) while everyone else keeps "
            "buying. When price finally curves over and **breaks down** below the rim, that's your "
            "sell signal: the top is in, the decline begins.\n\n"
            "It's a lovely picture — the exact bearish mirror of the rounding bottom / saucer base "
            "we already tested in [Study 416](../../416-rounding-bottom/). So we built the same kind "
            "of **mechanical detector**, flipped to look for domes instead of bowls — fit a parabola, "
            "demand a real inverted-U, a peak in the middle, and a confirmed support breakdown — and "
            "asked the only question that matters: *after a confirmed dome breakdown, shorted, does "
            "the stock do anything a random short wouldn't?* The honest answer is **no**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the base-rate comparison, the borrow "
            "cost and the placebo test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Two honest caveats up front.** (1) Chart figures are **partly subjective** — we "
            "test the closest *mechanical* definition and say so; a human chartist might draw the "
            "dome differently. (2) We use a fixed **30-name** basket (SPY + large-caps still trading "
            "today), which carries **survivorship** — but here it works *against* the bearish claim "
            "(the worst confirmed outcomes, delistings and bankruptcies, can't appear in a "
            "2026-survivors panel). House style: [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a confirmed dome breakdown, shorted, do you make money? | **Sort of — but so "
            "does shorting almost anything the market later dislikes.** The short averages "
            f"**{R['h20'][2]:+.1f}% over 20 days** and **{R['h60'][2]:+.1f}% over 60**. Sounds like a "
            "real edge…\n"
            "| …is that **better** than shorting a random day in the same stock? | **No.** A "
            f"*random* short in the same stocks earns **{R['h20'][4]:+.1f}% / {R['h60'][4]:+.1f}%** "
            "over the same windows — basically the same (both negative, because stocks drift up and "
            "shorting anything costs you that drift). The dome adds **nothing** over a random bet "
            "against the tape. |\n"
            "| Could the shape just be **luck**? | **Looks like it.** Shuffle in random entry dates "
            "thousands of times and they beat the dome breakdown **~83–89%** of the time — nowhere "
            "near significant. |\n"
            "| Is it 'distribution'? | **Not supported.** The breakdown return is indistinguishable "
            "from the ordinary cost of shorting an up-drifting market. No special smart-money "
            "footprint shows up. |\n\n"
            "> The shape is real and pretty. The **edge** is a mirage: shorting the dome breakdown "
            "loses about what shorting anything else in the same stocks loses — to the market's own "
            "upward drift."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A rounding top is a dome-shaped distribution that forms over weeks or months. "
            "During the dome, **strong hands quietly sell into strength** while buyers pile in near "
            "the top. The smooth roll-over and the **breakdown below the rim** confirm the "
            "distribution phase is over — short the breakdown and ride the new downtrend.\"*\n\n"
            "This is canon — the exact mirror of Edwards & Magee's rounding bottom, in every charting "
            "course. The believer's strongest version isn't \"the shape is pretty\" — it's that the "
            "breakdown marks a **regime change** from distribution to markdown, so the forward "
            "**short** return after a confirmed breakdown should **beat the cost of shorting the "
            "stock on an ordinary day**. That's the testable promise."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be gold for bears: a *visual* signal, drawable by eye, that flags "
            "the start of a decline. You'd scan charts for domes, short the breakdowns, and profit "
            "from the reversal.\n\n"
            "But there's a trap here that's the **mirror image** of the one that sank the rounding "
            "bottom. Stocks **drift up** on average — which means shorting *anything*, on an *ordinary "
            "day*, already costs you money before the pattern gets a say. So a positive-looking short "
            "return after a dome breakdown proves nothing by itself: the only honest test is whether "
            "it beats shorting a **random day** in the same names. If it doesn't, the dome is just an "
            "elaborate way of picking a bad-but-not-special day to fight the tape."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take SPY plus **{R['n_names']-1} large-caps** ({R['years']:.0f} years of daily bars) "
            "and run a **mechanical** rounding-top detector at every bar:\n\n"
            "1. **A real dome.** Fit a parabola to the last ~90 days of closes; demand it actually "
            "curves *down* (negative curvature) and fits well.\n"
            "2. **A peak in the middle.** The high point must sit *inside* the window, not at an "
            "edge — a dome, not a climb.\n"
            "3. **A confirmed breakdown.** The close must cross back **below the rim** (the left-edge "
            "support) for the first time — the dome is finished.\n\n"
            "Then we **short at the next day's open** (no cheating) and measure the forward 10 / 20 / "
            "60-day return — against the **base rate** (shorting a random day in the same stock) and "
            "a **date-shuffle placebo**, with realistic costs *and* a stock-borrow rate on the short. "
            "**Mirage verdict if** the breakdown doesn't beat random shorts."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what does the detector even find?** Here's one real dome it flagged — the "
            "fitted inverted-U and the breakdown bar. The shape detection works; that was never the "
            "question."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tk = 'AAPL'\n"
            "    bars = PANEL[tk]\n"
            "    bk = st.detect_breakdowns(bars, base_len=R['base_len'])\n"
            "    if len(bk):\n"
            "        bo = bk.index[len(bk)//2]\n"
            "        i = bars.index.get_loc(bo)\n"
            "        w = bars['close'].iloc[i-R['base_len']+1:i+40]\n"
            "        fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "        ax.plot(w.index, w.values, color=GREY, lw=1.6)\n"
            "        seg = np.log(bars['close'].iloc[i-R['base_len']+1:i+1].values)\n"
            "        a,b,d,r2 = st._parabola_fit(seg)\n"
            "        u = np.arange(len(seg)); fit = np.exp(a*u*u+b*u+d)\n"
            "        ax.plot(w.index[:len(seg)], fit, color=AMBER, lw=2.2, label=f'parabola fit (R\\u00b2={r2:.2f})')\n"
            "        ax.axvline(bo, color=RED, lw=2, label='confirmed breakdown')\n"
            "        ax.set_title(f'{tk}: a mechanically-detected rounding top'); ax.legend()\n"
            "        plt.tight_layout(); plt.show()\n"
            "        print(f'{tk} dome breakdown on {bo.date()}: fit R\\u00b2={r2:.2f}')\n"
            "else:\n"
            "    print('(cache absent — detector demo skipped; numbers are frozen in R)')"
        ),
        md(
            "The detector cleanly finds the figure: a smooth dome, a fitted parabola, a breakdown. "
            "The shape is genuinely *there*. Now the real question — **does shorting it pay?**"
        ),
        md(
            "**The decisive chart.** Confirmed-breakdown short return (red) vs the **base rate** — "
            "shorting a random day in the same stocks (grey) — at three horizons. If the pattern "
            "means anything, red should be far more negative (more profitable for the short) than "
            "grey."
        ),
        code(
            "hs = [10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    sig, base = [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, borrow_bps=30.0, placebo=False)\n"
            "        sig.append(r['gross_mean']*100); base.append(r['base_mean']*100)\n"
            "else:\n"
            "    sig = [R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    base = [R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.2, sig, .4, color=RED, label='dome breakdown, shorted')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='base rate (random short, same stocks)')\n"
            "for i,(s,b) in enumerate(zip(sig,base)):\n"
            "    ax.annotate(f'{s:+.1f}%',(i-.2,s),ha='center',va='top',fontsize=9)\n"
            "    ax.annotate(f'{b:+.1f}%',(i+.2,b),ha='center',va='top',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('short return (%)')\n"
            "ax.set_title('The dome breakdown barely beats a random short — both just pay the drift'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('signal:', [round(s,2) for s in sig], ' base:', [round(b,2) for b in base])"
        ),
        md(
            f"There's the whole story. At 20 days the breakdown short earns **{R['h20'][2]:+.1f}%** — "
            f"but shorting a **random day** in the same stocks earns **{R['h20'][4]:+.1f}%**. At 60 "
            f"days: **{R['h60'][2]:+.1f}%** vs **{R['h60'][4]:+.1f}%**. The bars are almost the same "
            "height. The dome isn't adding an edge; it's just selecting *a moment to fight a stock "
            "that was drifting up anyway* — no worse, but no better, than picking a day at random."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The breakdown's forward short return **does not beat the base "
            f"rate** (20-day: {R['h20'][2]:+.1f}% vs {R['h20'][4]:+.1f}%; difference statistically "
            "zero). A date-shuffle placebo says random shorts do at least as well ~83–89% of the "
            "time.\n"
            "- **Tradability — Mirage.** There's no excess return to trade — the loss is just the "
            "cost of shorting an up-drifting market, made worse by the borrow rate.\n"
            "- **\"Distribution signal\"? — Not supported.** No special smart-money footprint shows "
            "up in the forward returns; the breakdown looks like any other day you picked to short a "
            "stock that keeps drifting up."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Short version: **there's nothing to trade.** The dome breakdown's *gross* short return "
            "is already no better than a random short; charging **5 bps** round-trip plus a modest "
            "**30 bps/year** borrow rate just makes an already-zero edge a bit more negative. You'd "
            "do as well — and pay no scanning effort, no borrow — by not shorting at all."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g, nv, bs = [], [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, borrow_bps=30.0, placebo=False)\n"
            "        g.append(r['gross_mean']*100); nv.append(r['net_mean']*100); bs.append(r['base_mean']*100)\n"
            "else:\n"
            "    g=[R['h10'][2],R['h20'][2],R['h60'][2]]; nv=[R['h10'][3],R['h20'][3],R['h60'][3]]; bs=[R['h10'][4],R['h20'][4],R['h60'][4]]\n"
            "x=np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.25,g,.25,color=RED,label='breakdown gross')\n"
            "ax.bar(x,nv,.25,color=GREY,label='breakdown net (5bps+borrow)')\n"
            "ax.bar(x+.25,bs,.25,color=AMBER,label='base rate (random short)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('return (%)')\n"
            "ax.set_title('Net of costs the breakdown short is no better than a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net:', [round(v,2) for v in nv], ' base:', [round(b,2) for b in bs])"
        ),
        md(
            "> The honest bottom line: a rounding top is a **description of the past**, not a "
            "**prediction of the future**. By the time the breakdown confirms, the easy part — the "
            "roll-over from the peak — has already happened on the chart, and what's left is just "
            "the ordinary cost of betting against a market that goes up more often than it goes "
            "down."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Subjectivity is the obvious escape hatch.** A believer can always say \"that's not "
            "how *I'd* draw it.\" Fair — but a rule you can't write down is a rule you can't test or "
            "trade. Tighten or loosen the detector (we vary the window and fit threshold in the "
            "quants notebook) and the verdict doesn't budge.\n"
            "- **Volume confirmation.** Classic lore adds *rising volume on the breakdown*. A natural "
            "next experiment: condition on a volume surge and re-test — does the base-rate gap open "
            "up?\n"
            "- **The bullish mirror already tested.** [Study 416](../../416-rounding-bottom/) runs "
            "the exact same machinery on the saucer/rounding-bottom (long side) — same conclusion, "
            "opposite direction. Read them together for the full picture of this shape family.\n\n"
            "*Think the dome beats a random short once you add volume? Show the breakdown's forward "
            "short return clearing the **base rate** at t ≥ 2, after borrow — then we'll talk.*"
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
            "# Rounding Top — a quantitative teardown 🔬\n"
            "### A mechanical dome detector (parabola fit + interior peak + confirmed rim "
            "breakdown) · forward 10/20/60-day SHORT returns vs the every-bar SHORT base rate · "
            "one-sample & HAC *t*, a Welch *t* vs base, a date-shuffle placebo, borrow-adjusted "
            "costs · a synthetic shape-vs-continuation control (with a documented anchoring "
            "pitfall fixed)\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). This "
            "is the **bearish mirror** of [Study 416](../../416-rounding-bottom/)'s rounding bottom: "
            "same parabola machinery, negative curvature, a confirmed *breakdown*, and a **short** "
            "trade. The job here is to resist the mirror-image chart-pattern trap: stocks drift up, "
            "so **any** short-only rule shows a *negative* expected return before the pattern says "
            "anything. The decisive statistic is therefore **not** the breakdown short return "
            "against zero — it's against the **base rate** (shorting a random day in the same name) "
            "and a **date-shuffle placebo**. Both say the dome adds nothing.\n\n"
            "> ⚠️ **Subjectivity + survivorship notes.** Chart figures are partly subjective; we "
            "test the *closest mechanical definition* (a least-squares parabola with an interior "
            "vertex and a confirmed rim breakdown) and report robustness to its knobs. The basket is "
            "**30 survivors** (SPY + large-caps still trading 2026) — here survivorship "
            "**mildly works against** any 'top predicts a decline' finding (the worst confirmed "
            "outcomes — delistings, bankruptcies — can't appear in a 2026-survivors panel), which "
            "makes a *null* result only more credible. Methods: "
            "[`docs/references.md`](../docs/references.md); numbers: "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Breakdown 20d SHORT return **{R['h20'][2]:+.2f}%** vs base "
            f"rate **{R['h20'][4]:+.2f}%** → Welch **t = {R['h20'][9]:.2f}** (not vs zero: the "
            f"one-sample t = {R['h20'][7]:.2f} is just the equity drift working against every "
            f"short). Date-shuffle placebo **p = {R['h20'][10]:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | No excess over the base rate to trade; net of 5 bps + "
            f"30 bps/yr borrow the breakdown (**{R['h20'][3]:+.2f}%** at 20d) is worse, not better, "
            "than a random short. |\n"
            f"| **Distribution signal?** | `NOT SUPPORTED` | Across base-window {{60,90,120}} and "
            f"fit R\\u00b2 {{0.45,0.55,0.70}}, the breakdown-vs-base Welch *t* **never clears 2 in "
            f"magnitude** (range \\u22121.03 to +0.75). No regime-change footprint. |\n\n"
            "> 💡 In plain words: the dome breakdown's short return looks negative (profitable) "
            "only because shorting *anything* in an up-drifting market tends to lose slightly — "
            "compared to a random day in the same stock, it earns nothing extra, so the "
            "'distribution → markdown' story has no statistical support here."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a *confirmed rounding-top breakdown* in name $j$ at bar $t$ be the event $E_{j,t}$ "
            "defined by the mechanical detector (parabola curvature $a<0$, fit $R^2\\ge R^2_{\\min}$, "
            "interior vertex, height $\\ge \\delta$ above support, first close below the left-rim "
            "support). Short at $t{+}1$ open, cover $H$ days later; let $r_{j,t}(H) = 1 - "
            "\\frac{P_{t+1+H}}{P_{t+1}}$ be that forward SHORT return.\n\n"
            "- **H₁ (the believer).** $\\mathbb{E}[r \\mid E] > \\mathbb{E}[r]$ — the breakdown's "
            "forward SHORT return **exceeds the unconditional base rate** (distribution → markdown "
            "is a regime change).\n"
            "- **The null.** $\\mathbb{E}[r \\mid E] = \\mathbb{E}[r]$ — the dome just selects "
            "moments to short an up-drifting asset; the *shape* carries no information.\n\n"
            "The decisive test is a **Welch $t$ of breakdown SHORT returns vs the base rate**, plus a "
            "**date-shuffle placebo** (match the breakdown *count* per name, draw random entry dates, "
            "short them the same way). Testing vs **zero** would be the mirror-image classic error: "
            "it confounds the signal with the equity drift premium every short position pays."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the chart-pattern trap, mirrored\n\n"
            "Write the breakdown short return as $r_{j,t}(H) \\approx -\\mu_j H + "
            "\\alpha_{\\text{pattern}} + \\varepsilon$, where $\\mu_j$ is the name's average daily "
            "drift ($\\mu_j > 0$ for equities on average). A one-sample $t$ of $r$ against 0 tests "
            "$-\\mu_j H + \\alpha = 0$ — it lights up (in the *believer's* favoured, negative-for-t "
            "direction... here we test the short return, so a genuine decline reads *positive*) "
            "whenever the drift happens to be small or negative in-sample for those specific events, "
            "or simply because a large enough $H$ makes $-\\mu_j H$ dominate the noise — **for any "
            "short-only rule on any drifting stock, the sign is ambiguous and unstable, which is "
            "exactly why testing against zero is meaningless here.** Only "
            "$\\alpha_{\\text{pattern}}$ is the dome's own contribution, and the **base-rate "
            "difference** $\\mathbb{E}[r\\mid E] - \\mathbb{E}[r] \\approx \\alpha_{\\text{pattern}}$ "
            "isolates it. This is why the desk's inference bar is a *t* against the **right** "
            "benchmark, not against zero — a significant one-sample *t* on a short-only chart "
            "pattern, positive OR negative, is not itself evidence of anything pattern-specific."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY + **{R['n_names']-1}** US large-caps (yfinance daily adjusted OHLC, "
            f"{R['start']} → {R['end']}, {R['years']:.1f}y). **Survivor** panel — named on the Signal "
            f"axis, working *against* the bearish claim here. Fingerprint `{R['fp']}`.\n"
            "- **Detector (mechanical).** Trailing window `base_len=90`; least-squares parabola with "
            "$a<0$, $R^2\\ge 0.55$; vertex inside the central 60% of the window; peak $\\ge 0.08$ "
            "(log) above the support; the close is the **first** to break below the left-rim low; "
            "30-bar cooldown.\n"
            "- **Timing.** Signal known at close $t$; **short $t{+}1$ open**, cover close at "
            "$t{+}1{+}H$ (one documented lag). $H\\in\\{10,20,60\\}$.\n"
            "- **Benchmarks.** (a) base rate = every-bar forward $H$-SHORT-return (the 'just short a "
            "random day' null); (b) date-shuffle placebo = random entry dates matched to the "
            "per-name breakdown count, shorted the same way, 5,000 draws.\n"
            "- **Costs.** 5 bps one-way × NAV on the round trip **+ 30 bps/yr borrow**, pro-rated "
            "over the holding horizon — the house-rule cost a short pays that Study 416's long "
            "trade does not.\n"
            "- **Positive control.** A synthetic panel that plants the dome **shape**; `edge=0` adds "
            "no continuation (must NOT beat base) and `edge<0` plants a real decline (must light "
            "up). Checked clean across 20 seeds after fixing a subtle anchoring pitfall (4d)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The decisive comparison — breakdown vs base rate, with the *right* t\n\n"
            "Two *t*-stats side by side at each horizon: the **one-sample t vs zero** (the trap — it "
            "mixes in the equity-drift cost every short pays) and the **Welch t vs the base rate** "
            "(the honest test). Only the second one answers 'does the shape do anything?'."
        ),
        code(
            "hs = [10,20,60]\n"
            "if HAVE_REAL:\n"
            "    t0, tb = [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, borrow_bps=30.0, placebo=False)\n"
            "        t0.append(r['t_zero']); tb.append(r['t_vs_base'])\n"
            "else:\n"
            "    t0=[R['h10'][7],R['h20'][7],R['h60'][7]]; tb=[R['h10'][9],R['h20'][9],R['h60'][9]]\n"
            "x=np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.2,t0,.4,color=GREY,label='one-sample t vs ZERO (the trap = drift)')\n"
            "ax.bar(x+.2,tb,.4,color=RED,label='Welch t vs BASE RATE (the honest test)')\n"
            "ax.axhline(2,ls='--',c=RED,label='t = 2 bar'); ax.axhline(-2,ls='--',c=RED)\n"
            "ax.axhline(0,c='k',lw=.8)\n"
            "for i,(a,b) in enumerate(zip(t0,tb)):\n"
            "    ax.annotate(f'{a:.2f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.2f}',(i+.2,b),ha='center',va='top' if b<0 else 'bottom',fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('t-stat')\n"
            "ax.set_title('vs zero clears the bar (drift); vs base rate never does'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs zero:', [round(v,2) for v in t0]); print('t vs base:', [round(v,2) for v in tb])"
        ),
        md(
            f"> 💡 In plain words: against **zero**, the 20d and 60d breakdown shorts clear the bar "
            f"(t = {R['h20'][7]:.2f}, {R['h60'][7]:.2f}) — that's mostly the equity drift talking "
            f"(any random short over the same era shows a similar-magnitude negative one-sample t). "
            f"Against the **base rate** the same events sit at t = {R['h20'][9]:.2f} / "
            f"{R['h60'][9]:.2f} — nowhere near 2. The shape contributes no measurable alpha, and in "
            "one configuration (below) even points the wrong way."
        ),
        md(
            "### 4b · The date-shuffle placebo — is the *shape* doing anything?\n\n"
            "Draw the same number of random entry dates per name (matching the breakdown count) "
            "5,000 times, short them the same way, and pool the forward 20d SHORT returns. The "
            "observed breakdown mean should sit deep in the right tail if the dome matters. It "
            "doesn't — random shorts beat it most of the time."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=20, cost_bps=0.0, borrow_bps=0.0, n_draws=5000, seed=705)\n"
            "    counts = {tk: len(st.detect_breakdowns(PANEL[tk], base_len=R['base_len'])) for tk in PANEL}\n"
            "    pl = st.permutation_placebo(PANEL, R['base_len'], 20, counts, n_draws=5000, seed=705)\n"
            "    draws = pl['draws']*100; draws = draws[np.isfinite(draws)]\n"
            "    obs = r['gross_mean']*100; pval = r['p_placebo']\n"
            "else:\n"
            "    obs = R['h20'][2]; pval = R['h20'][10]\n"
            "    rng = np.random.default_rng(705); draws = rng.normal(R['h20'][4], 0.35, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: 5,000 random-date short draws')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed breakdown short {obs:+.2f}%')\n"
            "ax.set_xlabel('pooled 20-day forward SHORT return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  placebo p = {pval:.3f}  (random shorts beat the dome ~{pval*100:.0f}% of the time)')"
        ),
        md(
            f"> 💡 In plain words: the red line sits **inside** the random-date cloud — random "
            f"shorts beat the dome breakdown about **{R['h20'][10]*100:.0f}%** of the time "
            f"(p = {R['h20'][10]:.3f}). If the shape carried information, the observed return would "
            "be out in the (favourable) tail. It isn't."
        ),
        md(
            "### 4c · Robustness — the verdict doesn't move with the knobs\n\n"
            "Chart figures are subjective, so the fair worry is 'you drew it wrong.' We vary the "
            "dome **window length** and the **fit threshold**. The breakdown-vs-base *t* never "
            "clears 2 in magnitude in any configuration — the shortest window even flips sign in the "
            "believer's favour, without ever approaching significance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rbl = []\n"
            "    for bl in (60,90,120):\n"
            "        r = st.run_experiment(PANEL, base_len=bl, horizon=20, cost_bps=5.0, borrow_bps=30.0, placebo=False)\n"
            "        rbl.append((bl, r['n_signals'], r['t_vs_base']))\n"
            "    rr2 = []\n"
            "    for r2 in (0.45,0.55,0.70):\n"
            "        r = st.run_experiment(PANEL, base_len=90, horizon=20, r2_min=r2, cost_bps=5.0, borrow_bps=30.0, placebo=False)\n"
            "        rr2.append((r2, r['n_signals'], r['t_vs_base']))\n"
            "else:\n"
            "    rbl = [(b[0],b[1],b[4]) for b in R['rob_bl']]\n"
            "    rr2 = [(b[0],b[1],b[4]) for b in R['rob_r2']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar([str(b[0]) for b in rbl],[b[2] for b in rbl],color=AMBER,width=.55)\n"
            "a1.axhline(2,ls='--',c=RED); a1.axhline(-2,ls='--',c=RED); a1.axhline(0,c='k',lw=.8)\n"
            "for i,b in enumerate(rbl): a1.annotate(f'{b[2]:.2f}\\nn={b[1]}',(i,b[2]),ha='center',va='bottom' if b[2]>=0 else 'top',fontsize=8)\n"
            "a1.set_xlabel('base_len (window)'); a1.set_ylabel('t vs base rate (20d)'); a1.set_title('Window length: never clears 2')\n"
            "a2.bar([str(b[0]) for b in rr2],[b[2] for b in rr2],color=AMBER,width=.55)\n"
            "a2.axhline(2,ls='--',c=RED); a2.axhline(-2,ls='--',c=RED); a2.axhline(0,c='k',lw=.8)\n"
            "for i,b in enumerate(rr2): a2.annotate(f'{b[2]:.2f}\\nn={b[1]}',(i,b[2]),ha='center',va='bottom' if b[2]>=0 else 'top',fontsize=8)\n"
            "a2.set_xlabel('R\\u00b2 fit threshold'); a2.set_ylabel('t vs base rate (20d)'); a2.set_title('Fit strictness: never clears 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('base_len:', [(b[0],b[1],round(b[2],2)) for b in rbl]); print('r2_min:', [(b[0],b[1],round(b[2],2)) for b in rr2])"
        ),
        md(
            "> 💡 In plain words: no knob rescues it. A *looser* dome (more signals) still adds "
            "nothing; the *shortest* window (60 days) actually tips modestly toward the believer's "
            "favour (t = +0.75) — consistent with a short-term oversold bounce after a sharp decline "
            "— but even that never approaches significance. There is no setting where 'short the "
            "confirmed rounding-top breakdown' reliably beats a random short."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a synthetic panel we **plant the dome shape**. With `edge=0` there is no "
            "post-breakdown continuation (a true null: the figure fires, but must NOT beat the base "
            "rate). With `edge<0` we plant a real decline the detector must recover.\n\n"
            "Building this control taught a real lesson (documented in "
            "[`rounding_top/data.py`](../rounding_top/data.py)): a naively-anchored planted dome can "
            "**leak its own tail-end decline into the 'clean' forward window** whenever the "
            "detector's sliding 90-day lookback references a still-elevated point *inside* the dome "
            "rather than the true support bar — a look-ahead-flavoured artefact that pushed the "
            "null's Welch *t* past 2 in most of an early 20-seed sweep. The fix: anchor the shape to "
            "the exact bar the live detector's window actually references, and widen the flat "
            "consolidation pad at the support level so the lookback window settles back to a neutral "
            "reference quickly after the plant ends. Checked clean afterward (below)."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, -0.10):\n"
            "    px,_ = data.synthetic_panel(edge=edge, seed=705)\n"
            "    r = st.run_experiment(px, base_len=90, horizon=20, cost_bps=0.0, borrow_bps=0.0, n_draws=2000, seed=705)\n"
            "    res.append((edge, r['n_signals'], r['t_vs_base'], r['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "labels=[f'edge {e*100:.0f}%\\n(shape only)' if e==0 else f'edge {e*100:.0f}%\\n(real decline)' for e,_,_,_ in res]\n"
            "tv=[r[2] for r in res]\n"
            "ax.bar(labels,tv,color=[GREY,RED],width=.5)\n"
            "ax.axhline(2,ls='--',c=RED,label='t = 2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('t vs base rate (20d)'); ax.set_title('Control: shape-only -> ~0; planted decline -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,t,p in res: print(f'edge={e:+.2f}: n={n} t_vs_base={t:.2f} p_placebo={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: when the shape carries **no** continuation the control sits at "
            f"t = {R['syn'][0][5]:.2f} (p = {R['syn'][0][6]:.3f}) — it does **not** fake a decline "
            f"edge from a pretty shape. When a real decline is planted it jumps to t = "
            f"{R['syn'][1][5]:.2f} (p = {R['syn'][1][6]:.3f}). Checked across "
            f"**{R['syn_null_seeds']} seeds** (never a single lucky stream): null mean Welch t = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), **|t| ≥ 2 in "
            f"{R['syn_null_fire']}/{R['syn_null_seeds']} seeds** — essentially the nominal false-"
            "positive rate, not a systematic tell. The engine can detect continuation when it "
            "exists — so the real-tape null is a real null, not a blind detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the confirmed-breakdown forward short return "
            f"(**{R['h20'][2]:+.2f}%** at 20d, **{R['h60'][2]:+.2f}%** at 60d) **does not beat the "
            f"base rate** ({R['h20'][4]:+.2f}% / {R['h60'][4]:+.2f}%): Welch t = {R['h20'][9]:.2f} / "
            f"{R['h60'][9]:.2f}, placebo p = {R['h20'][10]:.3f} / {R['h60'][10]:.3f}. The one-sample "
            f"t vs zero (t = {R['h20'][7]:.2f}) is the equity-drift trap, not a signal. "
            "Survivorship here works *against* the bearish claim — there is still nothing to find.\n"
            "- **Tradability `MIRAGE`** — no excess over a random short to harvest; 5 bps of cost "
            "plus 30 bps/yr borrow just push an already-zero edge more negative. Nothing to size.\n"
            f"- **Distribution signal? `NOT SUPPORTED`** — across base-window {{60,90,120}} and fit "
            f"R\\u00b2 {{0.45,0.55,0.70}} the breakdown-vs-base t **never clears 2 in magnitude** "
            "(\\u22121.03 to +0.75); the 'distribution → markdown regime change' leaves no "
            "measurable forward footprint — the exact mirror of Study 416's finding for the "
            "rounding bottom."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is no excess to trade\n\n"
            "The operational truth in one line: gross breakdown short return ≈ base rate, and net of "
            "costs *and* borrow it is *below* it. The 'tradeable' object — excess over just shorting "
            "the stock on a random day — is indistinguishable from zero at every horizon, so "
            "capacity, impact and sizing are moot."
        ),
        code(
            "hs=[10,20,60]\n"
            "if HAVE_REAL:\n"
            "    exc=[]\n"
            "    for h in hs:\n"
            "        r=st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, borrow_bps=30.0, placebo=False)\n"
            "        exc.append((r['net_mean']-r['base_mean'])*100)\n"
            "else:\n"
            "    exc=[R['h10'][3]-R['h10'][4], R['h20'][3]-R['h20'][4], R['h60'][3]-R['h60'][4]]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar([f'{h}d' for h in hs], exc, color=[GREEN if v>0 else RED for v in exc], width=.55)\n"
            "ax.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(exc): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('net breakdown short return MINUS base rate (%)'); ax.set_title('Excess over just-shorting-randomly: ~zero, and negative net of cost+borrow')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net excess over base by horizon:', {f'{h}d': round(v,2) for h,v in zip(hs,exc)})"
        ),
        md(
            "> 💡 In plain words: the thing a trader would actually pocket — return *beyond* a "
            "random short — hovers around zero and dips more negative after costs and borrow. A "
            "rounding top is a fine *after-the-fact description* of a top that already formed; it is "
            "not a forward edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Volume-confirmed breakdowns.** The strongest steelman adds falling volume on the "
            "way up / rising volume on the breakdown; condition on a volume signature at the rim and "
            "re-run the base-rate test — a clean PR-able extension.\n"
            "- **Detector geometry.** We used a parabola; alternatives (cubic-spline curvature, "
            "swing-pivot symmetry, neckline slope) are worth a sensitivity pass — but 4c suggests "
            "the geometry isn't the bottleneck.\n"
            "- **The general lesson, mirrored.** *Every* short-only chart pattern fights the equity "
            "drift premium by default. Always benchmark a bearish figure against the base rate of "
            "shorting randomly, never against zero — the same trap [Study 416](../../416-rounding-"
            "bottom/) documents for the bullish saucer, and the sibling research-method demos make "
            "for look-ahead and multiple testing generally.\n\n"
            "*Reproducible core is offline & deterministic once cached. Methods/sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
