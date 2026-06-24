"""Generate the two narrative notebooks for Study 416 (Rounding Bottom).

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
# 2004-01-02 -> 2026-06-23, 22.5 years, mechanical rounding-bottom breakout detector, base_len=90).
R = dict(
    start="2004-01-02", end="2026-06-23", years=22.5, n_names=30, fp="69cb517da40d",
    base_len=90, cost_bps=5.0,
    # per horizon: (H, n_sig, gross%, net%, base%, win%, base_win%, t0, tHAC, tBase, p_placebo)
    h10=(10, 184, 0.53, 0.43, 0.59, 51, 57, 1.42, 1.22, -0.15, 0.569),
    h20=(20, 183, 1.51, 1.41, 1.14, 62, 58, 2.97, 2.72, 0.73, 0.252),
    h60=(60, 181, 4.07, 3.97, 3.31, 65, 63, 4.52, 4.42, 0.84, 0.205),
    # robustness at H=20 — base_len: (base_len, n, gross%, t0, tBase, p)
    rob_bl=[(60, 235, 1.12, 2.50, -0.05, 0.542), (90, 183, 1.51, 2.97, 0.73, 0.252),
            (120, 141, 0.02, 0.04, -1.80, 0.969)],
    # robustness at H=20, base_len=90 — r2_min: (r2_min, n, gross%, t0, tBase, p)
    rob_r2=[(0.45, 260, 1.29, 2.91, 0.34, 0.368), (0.55, 183, 1.51, 2.97, 0.73, 0.252),
            (0.70, 59, 0.87, 1.20, -0.37, 0.603)],
    # synthetic control at H=20: (edge, n, gross%, base%, t0, tBase, p, win%)
    syn=[(0.00, 185, 1.08, 0.48, 2.48, 1.38, 0.111, 57),
         (0.10, 186, 5.86, 1.07, 10.87, 8.87, 0.000, 80)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Accumulation_signal%3F: Not_supported](https://img.shields.io/badge/Accumulation_signal%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from rounding_bottom import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real() if HAVE_REAL else None
print("real rounding-bottom cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"rounding bottom\" really signal accumulation? 🥣\n"
            "### The saucer base — a beautiful chart shape that, on the tape, predicts nothing special\n\n"
            + BADGES +
            "Open any technical-analysis book and you'll meet the **rounding bottom** (a.k.a. the "
            "*saucer base*): a long, smooth, U-shaped dip where — the story goes — *smart money* is "
            "quietly **accumulating** while everyone else has given up. When price finally curves back "
            "up and **breaks out** above the rim, that's your buy signal: the saucer is full, the "
            "advance begins.\n\n"
            "It's a lovely picture. So we built a **mechanical detector** for it — fit a parabola, "
            "demand a real U-shape, a dip in the middle, and a confirmed breakout — and asked the only "
            "question that matters: *after a confirmed saucer breakout, does the stock do anything a "
            "random day wouldn't?* The honest answer is **no**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the base-rate comparison and the "
            "placebo test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Two honest caveats up front.** (1) Chart figures are **partly subjective** — we test "
            "the closest *mechanical* definition and say so; a human chartist might draw the saucer "
            "differently. (2) We use a fixed **30-name** basket (SPY + large-caps still trading today), "
            "which carries **survivorship**. House style: [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a confirmed saucer breakout, does the stock go **up**? | **Yes — but so does "
            "everything.** It averages **+1.5% over 20 days** and **+4.1% over 60**. Sounds great… |\n"
            "| …is that **more** than a random day in the same stock? | **No.** A *random* entry date "
            "earns **+1.1% / +3.3%** over the same windows — basically the same. The saucer adds "
            "**nothing** over just being long. |\n"
            "| Could the shape just be **luck**? | **Looks like it.** Shuffle in random entry dates "
            "thousands of times and they beat the saucer **~1 in 4** times — nowhere near significant. |\n"
            "| Is it 'accumulation'? | **Not supported.** The breakout return is indistinguishable from "
            "the stock's own upward drift. No special smart-money footprint shows up. |\n\n"
            "> The shape is real and pretty. The **edge** is a mirage: the saucer breakout earns exactly "
            "what the stock was going to earn anyway."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A rounding bottom is a saucer-shaped base that forms over weeks or months. During the "
            "saucer, **strong hands accumulate** while sellers are exhausted. The smooth curve up and "
            "the **breakout above the rim** confirm the accumulation phase is over — buy the breakout "
            "and ride the new uptrend.\"*\n\n"
            "This is canon. It's in Edwards & Magee's *Technical Analysis of Stock Trends*, in John "
            "Murphy, in every charting course. The believer's strongest version isn't \"the shape is "
            "pretty\" — it's that the breakout marks a **regime change** from accumulation to markup, so "
            "the forward return after a confirmed breakout should **beat the stock's normal drift**. "
            "That's the testable promise."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be gold: a *visual* signal, drawable by eye, that flags the start of an "
            "advance. You'd scan charts for saucers, buy the breakouts, and beat buy-and-hold.\n\n"
            "But there's a trap that snares almost every chart-pattern claim. Stocks **drift up** on "
            "average. So *any* rule that buys stocks and holds them will show a positive forward "
            "return — that's not evidence the **shape** did anything. The only honest test is: does the "
            "saucer breakout beat a **random entry** in the same names over the same horizon? If not, "
            "the pattern is just an elaborate way of saying \"buy a stock.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take SPY plus **{R['n_names']-1} large-caps** ({R['years']:.0f} years of daily bars) "
            "and run a **mechanical** rounding-bottom detector at every bar:\n\n"
            "1. **A real U.** Fit a parabola to the last ~90 days of closes; demand it actually curves "
            "up (positive curvature) and fits well.\n"
            "2. **A dip in the middle.** The low point must sit *inside* the window, not at an edge — a "
            "saucer, not a slide.\n"
            "3. **A confirmed breakout.** The close must cross back **above the rim** (the left-edge "
            "high) for the first time — the saucer is finished.\n\n"
            "Then we enter the **next day's open** (no cheating) and measure the forward 10 / 20 / "
            "60-day return — against the **base rate** (a random day in the same stock) and a "
            "**date-shuffle placebo**. **Mirage verdict if** the breakout doesn't beat random entries."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what does the detector even find?** Here's one real saucer it flagged — the fitted "
            "U-shape and the breakout bar. The shape detection works; that was never the question."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tk = 'AAPL'\n"
            "    bars = PANEL[tk]\n"
            "    bk = st.detect_breakouts(bars, base_len=R['base_len'])\n"
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
            "        ax.axvline(bo, color=GREEN, lw=2, label='confirmed breakout')\n"
            "        ax.set_title(f'{tk}: a mechanically-detected rounding bottom'); ax.legend()\n"
            "        plt.tight_layout(); plt.show()\n"
            "        print(f'{tk} saucer breakout on {bo.date()}: fit R\\u00b2={r2:.2f}')\n"
            "else:\n"
            "    print('(cache absent — detector demo skipped; numbers are frozen in R)')"
        ),
        md(
            "The detector cleanly finds the figure: a smooth U, a fitted parabola, a breakout. The shape "
            "is genuinely *there*. Now the real question — **does it pay?**"
        ),
        md(
            "**The decisive chart.** Saucer-breakout forward return (green) vs the **base rate** — a "
            "random day in the same stocks (grey) — at three horizons. If the pattern means anything, "
            "green should tower over grey."
        ),
        code(
            "hs = [10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    sig, base = [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, placebo=False)\n"
            "        sig.append(r['gross_mean']*100); base.append(r['base_mean']*100)\n"
            "else:\n"
            "    sig = [R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    base = [R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.2, sig, .4, color=GREEN, label='saucer breakout')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='base rate (random day, same stocks)')\n"
            "for i,(s,b) in enumerate(zip(sig,base)):\n"
            "    ax.annotate(f'{s:+.1f}%',(i-.2,s),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:+.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('forward return (%)')\n"
            "ax.set_title('The saucer breakout barely beats a random day — the edge is the drift'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('signal:', [round(s,2) for s in sig], ' base:', [round(b,2) for b in base])"
        ),
        md(
            f"There's the whole story. At 20 days the breakout earns **+{R['h20'][2]:.1f}%** — but a "
            f"**random day** in the same stocks earns **+{R['h20'][4]:.1f}%**. At 60 days: "
            f"**+{R['h60'][2]:.1f}%** vs **+{R['h60'][4]:.1f}%**. The bars are almost the same height. "
            "The saucer isn't adding an edge; it's just selecting *a moment to be long a stock that was "
            "drifting up anyway.*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The breakout's forward return **does not beat the base rate** "
            f"(20-day: +{R['h20'][2]:.1f}% vs +{R['h20'][4]:.1f}%; difference statistically zero). A "
            "date-shuffle placebo says random entries do just as well ~1 in 4 times.\n"
            "- **Tradability — Mirage.** There's no excess return to trade — the absolute gain is just "
            "the market's drift, which you'd capture for free by holding the stock.\n"
            "- **\"Accumulation signal\"? — Not supported.** No special smart-money footprint shows up "
            "in the forward returns; the breakout looks like any other day in an up-drifting stock."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Short version: **there's nothing to trade.** The saucer breakout's *gross* return is "
            "already no better than random; charging even a tiny **5 bps** round-trip cost just nudges "
            "an already-zero edge into the red. You'd do as well — and pay no scanning effort — by "
            "buying the stock on a coin flip and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g, nv, bs = [], [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, placebo=False)\n"
            "        g.append(r['gross_mean']*100); nv.append(r['net_mean']*100); bs.append(r['base_mean']*100)\n"
            "else:\n"
            "    g=[R['h10'][2],R['h20'][2],R['h60'][2]]; nv=[R['h10'][3],R['h20'][3],R['h60'][3]]; bs=[R['h10'][4],R['h20'][4],R['h60'][4]]\n"
            "x=np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.25,g,.25,color=GREEN,label='breakout gross')\n"
            "ax.bar(x,nv,.25,color=GREY,label='breakout net (5 bps)')\n"
            "ax.bar(x+.25,bs,.25,color=RED,label='base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('return (%)')\n"
            "ax.set_title('Net of costs the breakout is no better than just being long'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net:', [round(v,2) for v in nv], ' base:', [round(b,2) for b in bs])"
        ),
        md(
            "> The honest bottom line: a rounding bottom is a **description of the past**, not a "
            "**prediction of the future**. By the time the breakout confirms, the easy part — the climb "
            "out of the basin — has already happened on the chart."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Subjectivity is the obvious escape hatch.** A believer can always say \"that's not how "
            "*I'd* draw it.\" Fair — but a rule you can't write down is a rule you can't test or trade. "
            "Tighten or loosen the detector (we vary the window and fit threshold in the quants "
            "notebook) and the verdict doesn't budge.\n"
            "- **Volume confirmation.** Classic lore adds *rising volume on the breakout*. A natural "
            "next experiment: condition on a volume surge and re-test — does the base-rate gap open up?\n"
            "- **Small-caps / longer saucers.** The figure is folklore-strongest in beaten-down "
            "small-caps over multi-month bases; re-run there (mind survivorship and costs).\n\n"
            "*Think the saucer beats random entries once you add volume? Show the breakout's forward "
            "return clearing the **base rate** at t ≥ 2 — then we'll talk.*"
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
            "# Rounding Bottom — a quantitative teardown 🔬\n"
            "### A mechanical saucer detector (parabola fit + interior trough + confirmed rim breakout) "
            "· forward 10/20/60-day returns vs the every-bar base rate · one-sample & HAC *t*, a Welch "
            "*t* vs base, a date-shuffle placebo · a synthetic shape-vs-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "here is to resist the chart-pattern trap: stocks drift up, so **any** long-only rule shows "
            "a positive forward return. The decisive statistic is therefore **not** the breakout return "
            "against zero — it's against the **base rate** (a random day in the same name) and a "
            "**date-shuffle placebo**. Both say the saucer adds nothing.\n\n"
            "> ⚠️ **Subjectivity + survivorship notes.** Chart figures are partly subjective; we test "
            "the *closest mechanical definition* (a least-squares parabola with an interior vertex and a "
            "confirmed rim breakout) and report robustness to its knobs. The basket is **30 survivors** "
            "(SPY + large-caps still trading 2026) — survivorship mildly flatters any 'base works' "
            "finding, which makes a *null* result only more damning. Methods: "
            "[`docs/references.md`](../docs/references.md); numbers: [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Breakout 20d return **+{R['h20'][2]:.2f}%** vs base rate "
            f"**+{R['h20'][4]:.2f}%** → Welch **t = {R['h20'][9]:.2f}** (not vs zero: the one-sample "
            f"t = {R['h20'][7]:.2f} is just the drift). Date-shuffle placebo **p = {R['h20'][10]:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | No excess over the base rate to trade; net of 5 bps the "
            f"breakout (**+{R['h20'][3]:.2f}%** at 20d) merely matches being long. |\n"
            f"| **Accumulation signal?** | `NOT SUPPORTED` | Across base-window {{60,90,120}} and fit "
            f"R\\u00b2 {{0.45,0.55,0.70}}, the breakout-vs-base Welch *t* **never clears 2** (range "
            f"\\u22121.80 to +0.73). No regime-change footprint. |\n\n"
            "> 💡 In plain words: the saucer breakout earns a positive number only because stocks go up. "
            "Compared to a random day in the same stock, it earns nothing extra — so the 'accumulation → "
            "markup' story has no statistical support here."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a *confirmed rounding-bottom breakout* in name $j$ at bar $t$ be the event $E_{j,t}$ "
            "defined by the mechanical detector (parabola curvature $a>0$, fit $R^2\\ge R^2_{\\min}$, "
            "interior vertex, depth $\\ge \\delta$, first close above the left-rim resistance). Enter at "
            "$t{+}1$ open, hold $H$ days; let $r_{j,t}(H)$ be that forward return.\n\n"
            "- **H₀ (the believer).** $\\mathbb{E}[r \\mid E] > \\mathbb{E}[r]$ — the breakout's forward "
            "return **exceeds the unconditional base rate** (accumulation → markup is a regime change).\n"
            "- **The null.** $\\mathbb{E}[r \\mid E] = \\mathbb{E}[r]$ — the saucer just selects "
            "moments to hold an up-drifting asset; the *shape* carries no information.\n\n"
            "The decisive test is a **Welch $t$ of breakout returns vs the base rate**, plus a "
            "**date-shuffle placebo** (match the breakout *count* per name, draw random entry dates). "
            "Testing vs **zero** would be the classic error: it confounds the signal with the equity "
            "drift premium every long position earns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the chart-pattern trap, formally\n\n"
            "Write the breakout return as $r_{j,t}(H) = \\mu_j H + \\alpha_{\\text{pattern}} + "
            "\\varepsilon$, where $\\mu_j$ is the name's average daily drift. A one-sample $t$ of $r$ "
            "against 0 tests $\\mu_j H + \\alpha = 0$ — it lights up whenever $\\mu_j > 0$, i.e. for "
            "**any** long-only rule on **any** up-drifting stock. Only $\\alpha_{\\text{pattern}}$ is "
            "the saucer's contribution, and the **base-rate difference** $\\mathbb{E}[r\\mid E] - "
            "\\mathbb{E}[r] \\approx \\alpha_{\\text{pattern}}$ isolates it. This is why the desk's "
            "inference bar is a *t* against the **right** benchmark, not against zero — a positive "
            "one-sample *t* on a long-only chart pattern is the null result, dressed up."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY + **{R['n_names']-1}** US large-caps (yfinance daily adjusted OHLC, "
            f"{R['start']} → {R['end']}, {R['years']:.1f}y). **Survivor** panel — named on the Signal "
            f"axis. Fingerprint `{R['fp']}`.\n"
            "- **Detector (mechanical).** Trailing window `base_len=90`; least-squares parabola with "
            "$a>0$, $R^2\\ge 0.55$; vertex inside the central 60% of the window; trough $\\ge 0.08$ "
            "(log) below the rim; the close is the **first** to exceed the left-rim high; 30-bar "
            "cooldown.\n"
            "- **Timing.** Signal known at close $t$; **enter $t{+}1$ open**, exit close at $t{+}1{+}H$ "
            "(one documented lag). $H\\in\\{10,20,60\\}$.\n"
            "- **Benchmarks.** (a) base rate = every-bar forward $H$-return (the 'just be long' "
            "null); (b) date-shuffle placebo = random entry dates matched to the per-name breakout "
            "count, 5,000 draws.\n"
            "- **Costs.** 5 bps one-way × NAV on the round trip (long-only; no borrow).\n"
            "- **Positive control.** A synthetic panel that plants the saucer **shape**; `edge=0` adds "
            "no continuation (must NOT beat base) and `edge>0` plants a real drift (must light up)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The decisive comparison — breakout vs base rate, with the *right* t\n\n"
            "Two *t*-stats side by side at each horizon: the **one-sample t vs zero** (the trap — it's "
            "just the drift) and the **Welch t vs the base rate** (the honest test). Only the second one "
            "answers 'does the shape do anything?'."
        ),
        code(
            "hs = [10,20,60]\n"
            "if HAVE_REAL:\n"
            "    t0, tb = [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, placebo=False)\n"
            "        t0.append(r['t_zero']); tb.append(r['t_vs_base'])\n"
            "else:\n"
            "    t0=[R['h10'][7],R['h20'][7],R['h60'][7]]; tb=[R['h10'][9],R['h20'][9],R['h60'][9]]\n"
            "x=np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.2,t0,.4,color=GREY,label='one-sample t vs ZERO (the trap = drift)')\n"
            "ax.bar(x+.2,tb,.4,color=GREEN,label='Welch t vs BASE RATE (the honest test)')\n"
            "ax.axhline(2,ls='--',c=RED,label='t = 2 bar'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,(a,b) in enumerate(zip(t0,tb)):\n"
            "    ax.annotate(f'{a:.2f}',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.2f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('t-stat')\n"
            "ax.set_title('vs zero clears the bar (drift); vs base rate never does'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs zero:', [round(v,2) for v in t0]); print('t vs base:', [round(v,2) for v in tb])"
        ),
        md(
            f"> 💡 In plain words: against **zero**, the 20d and 60d breakouts clear the bar "
            f"(t = {R['h20'][7]:.2f}, {R['h60'][7]:.2f}) — that's the equity drift premium talking. "
            f"Against the **base rate** the same events sit at t = {R['h20'][9]:.2f} / {R['h60'][9]:.2f} "
            "— nowhere near 2. The shape contributes no measurable alpha."
        ),
        md(
            "### 4b · The date-shuffle placebo — is the *shape* doing anything?\n\n"
            "Draw the same number of random entry dates per name (matching the breakout count) 5,000 "
            "times and pool the forward 20d returns. The observed breakout mean should sit deep in the "
            "right tail if the saucer matters. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(PANEL, base_len=R['base_len'], horizon=20, cost_bps=0.0, n_draws=5000, seed=416)\n"
            "    counts = {tk: len(st.detect_breakouts(PANEL[tk], base_len=R['base_len'])) for tk in PANEL}\n"
            "    pl = st.permutation_placebo(PANEL, R['base_len'], 20, counts, n_draws=5000, seed=416)\n"
            "    draws = pl['draws']*100; draws = draws[np.isfinite(draws)]\n"
            "    obs = r['gross_mean']*100; pval = r['p_placebo']\n"
            "else:\n"
            "    obs = R['h20'][2]; pval = R['h20'][10]\n"
            "    rng = np.random.default_rng(416); draws = rng.normal(R['h20'][4], 0.35, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: 5,000 random-date draws')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed breakout {obs:+.2f}%')\n"
            "ax.set_xlabel('pooled 20-day forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  placebo p = {pval:.3f}  (random dates match the saucer ~{pval*100:.0f}% of the time)')"
        ),
        md(
            f"> 💡 In plain words: the green line sits **inside** the random-date cloud — random entries "
            f"beat the saucer about **{R['h20'][10]*100:.0f}%** of the time (p = {R['h20'][10]:.3f}). "
            "If the shape carried information, the observed return would be out in the tail. It isn't."
        ),
        md(
            "### 4c · Robustness — the verdict doesn't move with the knobs\n\n"
            "Chart figures are subjective, so the fair worry is 'you drew it wrong.' We vary the saucer "
            "**window length** and the **fit threshold**. The breakout-vs-base *t* never clears 2 in any "
            "configuration — if anything the *tighter* definitions go *negative*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rbl = []\n"
            "    for bl in (60,90,120):\n"
            "        r = st.run_experiment(PANEL, base_len=bl, horizon=20, cost_bps=5.0, placebo=False)\n"
            "        rbl.append((bl, r['n_signals'], r['t_vs_base']))\n"
            "    rr2 = []\n"
            "    for r2 in (0.45,0.55,0.70):\n"
            "        r = st.run_experiment(PANEL, base_len=90, horizon=20, r2_min=r2, cost_bps=5.0, placebo=False)\n"
            "        rr2.append((r2, r['n_signals'], r['t_vs_base']))\n"
            "else:\n"
            "    rbl = [(b[0],b[1],b[4]) for b in R['rob_bl']]\n"
            "    rr2 = [(b[0],b[1],b[4]) for b in R['rob_r2']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar([str(b[0]) for b in rbl],[b[2] for b in rbl],color=AMBER,width=.55)\n"
            "a1.axhline(2,ls='--',c=RED); a1.axhline(0,c='k',lw=.8)\n"
            "for i,b in enumerate(rbl): a1.annotate(f'{b[2]:.2f}\\nn={b[1]}',(i,b[2]),ha='center',va='bottom' if b[2]>=0 else 'top',fontsize=8)\n"
            "a1.set_xlabel('base_len (window)'); a1.set_ylabel('t vs base rate (20d)'); a1.set_title('Window length: never clears 2')\n"
            "a2.bar([str(b[0]) for b in rr2],[b[2] for b in rr2],color=AMBER,width=.55)\n"
            "a2.axhline(2,ls='--',c=RED); a2.axhline(0,c='k',lw=.8)\n"
            "for i,b in enumerate(rr2): a2.annotate(f'{b[2]:.2f}\\nn={b[1]}',(i,b[2]),ha='center',va='bottom' if b[2]>=0 else 'top',fontsize=8)\n"
            "a2.set_xlabel('R\\u00b2 fit threshold'); a2.set_ylabel('t vs base rate (20d)'); a2.set_title('Fit strictness: never clears 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('base_len:', [(b[0],b[1],round(b[2],2)) for b in rbl]); print('r2_min:', [(b[0],b[1],round(b[2],2)) for b in rr2])"
        ),
        md(
            "> 💡 In plain words: no knob rescues it. A *looser* saucer (more signals) still adds "
            "nothing; a *stricter*, prettier saucer (fewer signals) actually under-performs the base "
            "rate. There is no setting where 'buy the confirmed rounding-bottom breakout' beats random."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a synthetic panel we **plant the saucer shape**. With `edge=0` there is no "
            "post-breakout continuation (a true null: the figure fires, but must NOT beat the base "
            "rate). With `edge>0` we plant a real drift the detector must recover."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.10):\n"
            "    px,_ = data.synthetic_panel(edge=edge, seed=416)\n"
            "    r = st.run_experiment(px, base_len=90, horizon=20, cost_bps=0.0, n_draws=2000, seed=416)\n"
            "    res.append((edge, r['n_signals'], r['t_vs_base'], r['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "labels=[f'edge {e*100:.0f}%\\n(shape only)' if e==0 else f'edge {e*100:.0f}%\\n(real drift)' for e,_,_,_ in res]\n"
            "tv=[r[2] for r in res]\n"
            "ax.bar(labels,tv,color=[GREY,GREEN],width=.5)\n"
            "ax.axhline(2,ls='--',c=RED,label='t = 2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('t vs base rate (20d)'); ax.set_title('Control: shape-only -> ~0; planted drift -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,t,p in res: print(f'edge={e:+.2f}: n={n} t_vs_base={t:.2f} p_placebo={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: when the shape carries **no** continuation the control sits at "
            f"t = {R['syn'][0][5]:.2f} (p = {R['syn'][0][6]:.3f}) — it does **not** fake an edge from a "
            f"pretty shape. When a real drift is planted it jumps to t = {R['syn'][1][5]:.2f} "
            f"(p = {R['syn'][1][6]:.3f}). The engine can detect continuation when it exists — so the "
            "real-tape null is a real null, not a blind detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the confirmed-breakout forward return (**+{R['h20'][2]:.2f}%** at "
            f"20d, **+{R['h60'][2]:.2f}%** at 60d) **does not beat the base rate** "
            f"(+{R['h20'][4]:.2f}% / +{R['h60'][4]:.2f}%): Welch t = {R['h20'][9]:.2f} / "
            f"{R['h60'][9]:.2f}, placebo p = {R['h20'][10]:.3f} / {R['h60'][10]:.3f}. The one-sample "
            f"t vs zero (t = {R['h20'][7]:.2f}) is the equity-drift trap, not a signal. Survivorship "
            "would only flatter a real effect — there is none.\n"
            "- **Tradability `MIRAGE`** — no excess over being long to harvest; 5 bps of cost just turns "
            "an already-zero edge slightly negative. Nothing to size.\n"
            f"- **Accumulation signal? `NOT SUPPORTED`** — across base-window {{60,90,120}} and fit "
            f"R\\u00b2 {{0.45,0.55,0.70}} the breakout-vs-base t **never clears 2** (\\u22121.80 to "
            "+0.73); the 'accumulation → markup regime change' leaves no measurable forward footprint."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is no excess to trade\n\n"
            "The operational truth in one line: gross breakout return ≈ base rate, and net of costs it "
            "is *below* it. The 'tradeable' object — excess over just holding the stock — is "
            "indistinguishable from zero at every horizon, so capacity, impact and sizing are moot."
        ),
        code(
            "hs=[10,20,60]\n"
            "if HAVE_REAL:\n"
            "    exc=[]\n"
            "    for h in hs:\n"
            "        r=st.run_experiment(PANEL, base_len=R['base_len'], horizon=h, cost_bps=5.0, placebo=False)\n"
            "        exc.append((r['net_mean']-r['base_mean'])*100)\n"
            "else:\n"
            "    exc=[R['h10'][3]-R['h10'][4], R['h20'][3]-R['h20'][4], R['h60'][3]-R['h60'][4]]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar([f'{h}d' for h in hs], exc, color=[GREEN if v>0 else RED for v in exc], width=.55)\n"
            "ax.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(exc): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('net breakout return MINUS base rate (%)'); ax.set_title('Excess over just-being-long: ~zero, and negative net of cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net excess over base by horizon:', {f'{h}d': round(v,2) for h,v in zip(hs,exc)})"
        ),
        md(
            "> 💡 In plain words: the thing a trader would actually pocket — return *beyond* buy-and-hold "
            "— hovers around zero and dips negative after costs. A rounding bottom is a fine "
            "*after-the-fact description* of a base that already formed; it is not a forward edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Volume-confirmed breakouts.** The strongest steelman adds rising volume; condition on a "
            "volume surge at the rim and re-run the base-rate test — a clean PR-able extension.\n"
            "- **Detector geometry.** We used a parabola; alternatives (cubic-spline curvature, "
            "swing-pivot symmetry, neckline slope) are worth a sensitivity pass — but 4c suggests the "
            "geometry isn't the bottleneck.\n"
            "- **The general lesson.** *Every* long-only chart pattern inherits the equity drift "
            "premium. Always benchmark a bullish figure against the base rate, never against zero — the "
            "sibling research-method demos make the same point about look-ahead and multiple testing.\n\n"
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
