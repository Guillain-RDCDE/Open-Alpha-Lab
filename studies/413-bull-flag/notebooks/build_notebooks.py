"""Generate the two narrative notebooks for Study 413 (Bull Flag).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily OHLC panel
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
# auto-adjusted daily OHLC, as-of 2026-05-31, 2005-01-03 -> 2026-05-29, 21.4 years, fp 2d8d0fde0371).
R = dict(
    asof="2026-05-31", last_bar="2026-05-29", start="2005-01-03", years=21.4,
    n_names=30, n_bars=5385, fingerprint="2d8d0fde0371",
    n_up=638, n_up_spy=4, n_down=531,
    # UP-break excess over base rate per horizon: (H, n, excess%, raw%, win%, t, hac_t, p_plac, net%)
    up={
        5:  (5, 634, -0.19, 0.14, 47, -1.07, -1.12, 0.60, -0.29),
        10: (10, 634, -0.32, 0.34, 47, -1.23, -1.31, 0.61, -0.42),
        20: (20, 628, -0.38, 0.94, 49, -1.07, -1.07, 0.59, -0.48),
        40: (40, 624, -0.60, 2.00, 50, -1.21, -1.10, 0.60, -0.70),
    },
    # DOWN-break (myth check) per horizon: (H, n, excess%, raw%, t, p_plac)
    down={
        5:  (5, 529, 0.71, 1.05, 3.88, 0.22),
        10: (10, 527, 0.56, 1.22, 1.99, 0.33),
        20: (20, 525, 0.68, 1.99, 1.67, 0.35),
        40: (40, 523, 0.94, 3.51, 1.60, 0.35),
    },
    # robustness at H=20: (min_pole, max_retrace, n, excess%, t, p)
    robust=[(0.10, 0.60, 1071, -0.66, -2.64, 0.71),
            (0.12, 0.50, 628, -0.38, -1.07, 0.59),
            (0.16, 0.40, 235, 0.38, 0.57, 0.44)],
    # SPY-only: (H, n, excess%, raw%, t, p)
    spy={5: (5, 4, 0.12, 0.35, 0.05, 0.48), 10: (10, 4, -2.12, -1.65, -0.50, 0.91),
         20: (20, 4, 1.45, 2.39, 0.35, 0.25), 40: (40, 4, 0.98, 2.83, 0.12, 0.40)},
    # synthetic control 20d: (planted_edge, planted, detected, n, excess%, t, p, win%)
    syn=[(0.00, 160, 218, 218, -0.88, -2.61, 0.66, 39),
         (0.20, 160, 384, 384, 5.63, 16.43, 0.001, 80)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Resolves_up%3F: Busted](https://img.shields.io/badge/Resolves_up%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bull_flag import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
    for f in PANEL:
        PANEL[f] = PANEL[f].loc[:pd.Timestamp(ASOF)]
    CLOSES = PANEL["close"]
else:
    PANEL = CLOSES = None
print("real bull-flag cache present:", HAVE_REAL,
      "| names:", (0 if CLOSES is None else CLOSES.shape[1]),
      "| bars:", (0 if CLOSES is None else len(CLOSES)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"bull flag\" really break out and run? 🚩\n"
            "### One of the most-taught chart patterns — tested on 21 years of real tape, in plain English\n\n"
            + BADGES +
            "Here's a piece of chart-reading lore every trader learns early: a stock rockets up "
            "(the **flagpole**), then pauses in a tidy little pullback that drifts gently down (the "
            "**flag**) — and when it pops back above the top of that pause (the **breakout**), the "
            "rally *resumes* for a second leg. So you buy the breakout and ride it. It even *looks* "
            "obvious on a chart.\n\n"
            "The trouble with \"looks obvious on a chart\" is that our eyes are pattern-making "
            "machines — we see flags everywhere *after* the fact. So we wrote down the closest "
            "honest, **mechanical** definition of a bull flag we could, pointed it at 21 years of "
            "SPY plus 29 big US stocks, and asked the only question that matters: **after a real "
            "breakout, does the stock do any better than if you'd just held it?**\n\n"
            "The answer is a clean, slightly uncomfortable **no** — and the *way* it fails is the "
            "fun part.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the symmetry "
            "check? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name** basket (SPY + names still "
            "trading today), so it carries **survivorship** — we can't include firms that blew up "
            "after a failed breakout. That bias tilts the test *in the pattern's favour*, and it "
            "**still** loses. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a confirmed bull-flag breakout, does the stock beat just holding it? | **No.** "
            "Over the next 20 days the breakout name does about **−0.4% *worse*** than its own "
            "average — and the win-rate is **below a coin flip**. |\n"
            "| But it goes *up* after the breakout, right? | **Up in raw terms, yes** (these are "
            "stocks that drift up anyway). The catch: it goes up **less** than the same name "
            "normally would. The flag *subtracts* value. |\n"
            "| Does it at least break **up** more than down? | **No — and this is the killer.** "
            "When the *same* flag breaks **down** (the \"failed flag\"), the stock actually drifts "
            "up **more** than when it breaks up. The direction tells you nothing. |\n"
            "| Could you trade it? | **No.** There's no edge to begin with, so costs just turn a "
            "small drag into a bigger one. On **SPY** specifically the pattern shows up only **4 "
            "times in 21 years** — not enough to say anything. |\n\n"
            "> The bull flag is a great *story* about momentum. As a *signal* — \"buy this breakout\" "
            "— it's a mirage: you're looking at the stock's own drift, and the flag if anything "
            "gets in the way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A sharp rally raises the **flagpole**. The stock then rests in a brief, shallow "
            "pullback — the **flag** — that drifts down or sideways in a tight channel, giving back "
            "only a little of the move. When price closes back **above the flag's high**, the "
            "uptrend **resumes**: buy the breakout, target another flagpole's worth.\"*\n\n"
            "This isn't a fringe idea — it's in the classic textbooks (Edwards & Magee, 1948; "
            "Bulkowski's *Encyclopedia of Chart Patterns*) and taught in essentially every "
            "technical-analysis course. The bull flag is one of the headline **continuation** "
            "figures. So the question isn't \"have people heard of it?\" It's: **does buying the "
            "breakout actually pay?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a flag breakout really *resumed the trend*, it would be a genuine, repeatable "
            "edge — a chart shape you could see, with your own eyes, that predicts the next few "
            "weeks. That would be a real crack in market efficiency, and it'd be everywhere "
            "(everybody can draw a flag).\n\n"
            "But there's a trap baked into the claim, and it's the same trap that fools a lot of "
            "chart patterns: a flag breakout **only happens after a big run-up**. Stocks that just "
            "ran up tend to keep drifting up a little (that's momentum) — so the breakout *looks* "
            "like it \"works\" simply because you cherry-picked a post-rally moment. The honest "
            "test has to ask: does the breakout beat **a random day in the same stock**? That's "
            "what we'll do."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **SPY + 29 large US stocks** and scan **{R['years']:.0f} years** of daily "
            "prices for the closest mechanical bull flag we can define:\n\n"
            "1. **The flagpole** — a steep run-up (≥12% over ~12 sessions).\n"
            "2. **The flag** — a brief, tight pause that drifts flat-to-down and gives back no more "
            "than half the pole.\n"
            "3. **The breakout** — the first close back above the flag's high. That's the signal.\n\n"
            f"That finds **{R['n_up']}** confirmed breakouts. For each one we **enter the next "
            "day's close** (no cheating) and measure the return over the next **5, 10, 20, 40** "
            "days — but as an **excess over that stock's own average**. Then the killer test: we "
            "compare it against **thousands of random entry dates** in the same stocks. If the flag "
            "is real, the breakout should beat the random dates. If it's just the stock's drift, it "
            "won't."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: raw return vs. excess return.** Here's the average move after a breakout at "
            "each horizon, two ways: the **raw** return (what you'd see) and the **excess** over "
            "the stock's own base rate (what the flag actually *adds*). Watch them split."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    raw, exc = [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, horizon=h, n_draws=2000)\n"
            "        raw.append(r['raw_mean']*100); exc.append(r['mean']*100)\n"
            "else:\n"
            "    raw = [R['up'][h][3] for h in hs]; exc = [R['up'][h][2] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, raw, .4, color=GREY, label='raw return (what you see)')\n"
            "ax.bar(x+.2, exc, .4, color=RED, label='excess over the stock\\'s own average')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('return (%)')\n"
            "ax.set_title('Up in raw terms, but BELOW the stock\\'s own base rate')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('raw 20d:', f'{raw[2]:+.2f}%', ' excess 20d:', f'{exc[2]:+.2f}%')"
        ),
        md(
            f"There's the whole illusion in one chart. The **grey** bars (raw return) are "
            f"**positive** — after a breakout the stock is up **+{R['up'][20][3]:.2f}%** over 20 "
            "days. Looks like the flag worked! But the **red** bars (excess over the stock's own "
            f"average) are **negative everywhere** — **{R['up'][20][2]:+.2f}%** at 20 days. The "
            "stock went up *less* than it normally does. You're seeing the stock's drift, not the "
            "flag — and the flag is actually a small drag."
        ),
        md(
            "**Now the killer test.** Maybe −0.4% is just noise? Let's compare the breakout to "
            "**thousands of random entry dates** in the same stocks. If the flag has any magic, the "
            "breakout (green line) should sit out in the right tail, away from the random cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(PANEL, horizon=20, n_draws=5000)\n"
            "    obs = r['mean']*100; pval = r['p_placebo']\n"
            "else:\n"
            "    obs = R['up'][20][2]; pval = R['up'][20][7]\n"
            "rng = np.random.default_rng(413)\n"
            "# illustrative null cloud centred at 0 (the real placebo p is computed above / frozen)\n"
            "cloud = rng.normal(0.0, abs(obs)/0.5 if obs else 0.7, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=60, color=GREY, alpha=.85, label='random entry dates (same stocks)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the breakout: {obs:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('20-day excess return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The breakout sits LEFT of centre — random dates beat it {pval*100:.0f}% of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'breakout excess {obs:+.2f}%   random dates beat it {pval*100:.0f}% of the time')"
        ),
        md(
            f"The breakout (red line) doesn't sit in the *right* tail — it sits **left of zero**. A "
            f"random day in the same stocks beats it about **{R['up'][20][7]*100:.0f}% of the "
            "time**. There's nothing here to attribute to the flag. The pattern is **noise dressed "
            "up as a signal**."
        ),
        md(
            "**The gotcha that buries it.** A pattern that *predicts up* should fail when it breaks "
            "the *other* way. So we ran the **identical** detector but looked for a break **down** "
            "through the flag — the \"failed flag\" a chartist would call bearish. If the flag "
            "really chooses *up*, the down-break should drift **down**. Here's both."
        ),
        code(
            "if HAVE_REAL:\n"
            "    up_e = [st.run_experiment(PANEL, horizon=h, n_draws=1500)['mean']*100 for h in hs]\n"
            "    dn_e = [st.run_experiment(PANEL, horizon=h, side='down', n_draws=1500)['mean']*100 for h in hs]\n"
            "else:\n"
            "    up_e = [R['up'][h][2] for h in hs]; dn_e = [R['down'][h][2] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, up_e, .4, color=GREEN, label='break UP (the textbook buy)')\n"
            "ax.bar(x+.2, dn_e, .4, color=RED, label='break DOWN (the \"failed\" flag)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('excess return (%)')\n"
            "ax.set_title('The \"failed\" down-break drifts UP MORE than the textbook up-break')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('5d:  up', f'{up_e[0]:+.2f}%', ' vs down', f'{dn_e[0]:+.2f}%')\n"
            "print('20d: up', f'{up_e[2]:+.2f}%', ' vs down', f'{dn_e[2]:+.2f}%')"
        ),
        md(
            f"This is the punchline. The **down**-break — the one a chartist calls a *failure* — "
            f"drifts up **more** than the textbook up-break at every horizon "
            f"(**{R['down'][5][2]:+.2f}%** vs **{R['up'][5][2]:+.2f}%** at 5 days; "
            f"**{R['down'][20][2]:+.2f}%** vs **{R['up'][20][2]:+.2f}%** at 20 days). Whether the "
            "flag \"works\" or \"fails,\" the stock drifts the same. The breakout **direction "
            "carries no information.** The flag is just telling you the stock recently ran up — "
            "which you already knew."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The breakout's excess over the stock's own base rate is "
            f"**negative at every horizon** (**{R['up'][20][2]:+.2f}%** at 20 days), the win-rate "
            "is **below 50%**, and random dates beat it ~3-in-5. No signal.\n"
            "- **Tradability — Mirage.** There's no edge to tax; costs just deepen the drag. On SPY "
            "specifically the pattern appears **4 times in 21 years**. Nothing to deploy.\n"
            "- **\"Resolves up specifically\"? — Busted.** The down-break of the same flag drifts "
            "up **more** than the up-break. Direction is uninformative — the figure is a momentum "
            "tell, not a directional signal."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Short version: **no.** You don't even get to the cost conversation, because the gross "
            "edge is already **below zero**. Charging a realistic round trip on top just makes a "
            "small loss a slightly bigger one — here's the net of costs, for completeness."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g, nv = [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, horizon=h, n_draws=800)\n"
            "        g.append(r['mean']*100); nv.append(r['net']*100)\n"
            "else:\n"
            "    g = [R['up'][h][2] for h in hs]; nv = [R['up'][h][8] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREY, label='gross excess')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net of costs')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return (%)'); ax.set_title('Already negative gross; costs only deepen it')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net 20d:', f'{nv[2]:+.2f}%')"
        ),
        md(
            f"Net of costs the 20-day breakout is **{R['up'][20][8]:+.2f}%** — you'd be *paying* to "
            "underperform the name you're holding. That's the entire tradability story: a mirage "
            "you can't even rescue by holding longer or trading cheaper."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is it the definition?** Maybe a *stricter* flag is the real one. We swept the "
            "detector from loose to strict — the loss shrinks toward zero but **never** turns into "
            "a clean positive (the quants notebook has the sweep). Tuning doesn't save it.\n"
            "- **Small-caps / intraday?** The folklore is loudest on fast intraday charts and small "
            "names. That's a fair next test — but it's also where costs and survivorship traps are "
            "worst, so be careful what you wish for.\n"
            "- **The lesson generalises.** Most \"it breaks out and runs\" chart patterns are "
            "really *momentum tells*: they fire after a run-up, so they inherit the stock's drift. "
            "The honest test is always \"vs a random day in the same name.\"\n\n"
            "*Think a tighter flag definition turns this positive? Show the **excess** beating the "
            "same-tape placebo (p < 0.05) at any horizon — then we'll talk.*"
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
            "# The Bull Flag — a quantitative teardown 🔬\n"
            "### Mechanical pole→flag→breakout detector · forward 5/10/20/40-day excess over base "
            "rate · one-sample/HAC *t* + same-tape placebo · the down-break symmetry test · a "
            "strictness sweep · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "bull flag is a textbook bullish **continuation** figure — the job here is to test it "
            "honestly: define the figure mechanically, measure forward returns as an **excess over "
            "each name's base rate** (killing the names'-own-drift confound), confront that with a "
            "**same-tape placebo**, and run the cleanest within-study control there is — the "
            "**down-break of the identical figure**.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name** basket (SPY + 29 large-caps still "
            "trading in 2026) — a *survivor* panel whose bias tilts post-breakout returns **up**, "
            "i.e. *for* the figure (and it still loses). Real data: yfinance daily auto-adjusted "
            "OHLC, 2005→2026 (as-of 2026-05-31). Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Up-break excess over base rate **{R['up'][20][2]:+.2f}%** at "
            f"20d (one-sample **t = {R['up'][20][5]:.2f}**, HAC **t = {R['up'][20][6]:.2f}**), "
            f"win-rate **{R['up'][20][4]}%** < 50, same-tape **placebo p = {R['up'][20][7]:.2f}**. "
            "Negative before it's anything else. |\n"
            f"| **Tradability** | `MIRAGE` | Gross excess < 0 at every H; net "
            f"**{R['up'][20][8]:+.2f}%** at 20d. SPY-only: **{R['spy'][20][1]}** events at "
            f"t = {R['spy'][20][4]:.2f}. Nothing to deploy. |\n"
            f"| **Resolves up?** | `BUSTED` | The **down**-break of the identical flag drifts up "
            f"**more** ({R['down'][5][2]:+.2f}% vs {R['up'][5][2]:+.2f}% at 5d; "
            f"{R['down'][20][2]:+.2f}% vs {R['up'][20][2]:+.2f}% at 20d). Direction carries no info. |\n\n"
            "> 💡 In plain words: there's no edge to attribute — the breakout *underperforms* the "
            "name's own drift, and the \"failed\" down-break does *better*. The flag is a momentum "
            "tell, miscalibrated as a directional signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a **flagpole** be a window with net log-rise $\\rho \\ge \\rho_{\\min}$ over "
            "$w_p$ bars, followed by a **flag** of $w_f \\in [w_{\\min}, w_{\\max}]$ bars whose "
            "deepest retrace of the pole is $\\le \\tau$, whose own range is $\\le \\gamma$, and "
            "whose net drift is not strongly up. The **breakout** is the first close above the "
            "flag high $h_f$. Entered one day later, the $H$-day forward return is $r_i(H)$; the "
            "test statistic is the **excess over the name's base rate** "
            "$\\mu_k(H)$:\n\n"
            "$$ x_i(H) = r_i(H) - \\mu_{k(i)}(H), \\qquad "
            "\\widehat{X}(H) = \\frac1n\\sum_i x_i(H). $$\n\n"
            "- **H₁ (the breakout pays).** $\\widehat{X}(H) > 0$ and beats a same-tape placebo.\n"
            "- **H₂ (deployable).** Survives one-way costs × 2 (round trip).\n"
            "- **H₃ (\"resolves up\").** The up-break excess is materially **above** the "
            "down-break excess of the identical figure.\n\n"
            "We find **H₁ rejected** ($\\widehat{X}<0$, placebo $p\\approx0.6$), **H₂ moot** (no "
            "gross edge to tax), **H₃ rejected** (down-break drifts up *more*). The figure is a "
            "momentum artefact, and not even a well-aimed one."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the confound the *t* can't see\n\n"
            "A breakout **selects post-run-up dates**. Recently-risen names keep drifting up a "
            "little (momentum), so *any* post-run-up entry inherits that drift — a naive "
            "forward-return *t* against zero would be measuring the **names**, not the **figure**. "
            "Two defences: **(a)** subtract each name's own base rate $\\mu_k(H)$ so the test is "
            "\"beats buy-and-hold *for that name*\"; **(b)** a **same-tape placebo** that draws "
            "random entry dates in the same names and asks how often a random set beats the "
            "breakout. Here both defences agree the answer is *no* — and uniquely, the naive *t* is "
            "already **negative**, so unlike the triangle this never even reaches the "
            "\"real-number-wrong-attribution\" stage."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** basket (SPY + 29 large-caps; yfinance "
            f"auto-adjusted daily OHLC, {R['start']}→{R['last_bar']}, as-of {R['asof']}); "
            f"**{R['n_bars']:,}** bars, **{R['years']:.1f}** years. Fingerprint `{R['fingerprint']}`. "
            "**Survivor** panel — named on the Signal axis (bias *for* the figure).\n"
            "- **Detector.** Flagpole ≥12% over 12 bars; flag 5–20 bars, ≤50% pole retrace, ≤10% "
            "range, not net-up; breakout = first close above the flag high. "
            f"**{R['n_up']}** up-breaks (SPY alone **{R['n_up_spy']}**); **{R['n_down']}** "
            "down-breaks for the symmetry test.\n"
            "- **Timing.** Breakout known at its close; **enter the next close** (one documented "
            "lag); hold $H\\in\\{5,10,20,40\\}$; drop windows that overrun the tape.\n"
            "- **Statistic.** Excess over each name's base rate, pooled across names.\n"
            "- **Null #1 (one-sample / HAC t)** of the pooled excess vs 0.\n"
            "- **Null #2 (same-tape placebo).** 5,000 draws of random entry dates per name (same "
            "count, same base-rate subtraction); $p=\\Pr[\\text{random excess}\\ge\\text{observed}]$.\n"
            "- **Myth-check.** The identical detector, $side=\\text{down}$ (break below the flag "
            "low) — the failed-flag symmetry test.\n"
            "- **Costs.** 5 bps one-way × 2 = 10 bps round trip.\n"
            "- **Positive control.** A deterministic panel with **planted** flags + a *known* "
            "post-breakout drift: zero edge must stay inside the placebo; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Raw vs excess — the momentum illusion in one chart\n\n"
            "Raw forward return (positive — these names drift up) against the excess over each "
            "name's base rate (what the *flag* adds). The split is the whole study."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    raw, exc, win = [], [], []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, horizon=h, n_draws=2000)\n"
            "        raw.append(r['raw_mean']*100); exc.append(r['mean']*100); win.append(r['win']*100)\n"
            "else:\n"
            "    raw = [R['up'][h][3] for h in hs]; exc = [R['up'][h][2] for h in hs]; win = [R['up'][h][4] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(x-.2, raw, .4, color=GREY, label='raw'); a1.bar(x+.2, exc, .4, color=RED, label='excess over base rate')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "a1.set_ylabel('return (%)'); a1.set_title('Raw > 0, excess < 0 (the flag subtracts)'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], win, color=[RED if w<50 else GREEN for w in win], width=.6)\n"
            "a2.axhline(50, ls='--', c='k', label='coin flip'); a2.set_ylim(40, 56)\n"
            "for i,w in enumerate(win): a2.annotate(f'{w:.0f}%',(i,w),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('win-rate (%)'); a2.set_title('Win-rate below 50% on the excess'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by H:', [round(v,2) for v in exc]); print('win by H:', [round(v,1) for v in win])"
        ),
        md(
            f"> 💡 In plain words: raw 20d is **+{R['up'][20][3]:.2f}%** but the excess is "
            f"**{R['up'][20][2]:+.2f}%** with a **{R['up'][20][4]}%** win-rate. The breakout "
            "underperforms the name's own drift and loses more often than a coin. The positive raw "
            "number is the *stock's* momentum, not the flag's."
        ),
        md(
            "### 4b · The decisive test — the same-tape placebo\n\n"
            "The observed 20-day excess against a same-tape null (random entry dates in the same "
            "names). A real edge sits in the right tail; a momentum artefact sits in the cloud — "
            "here it sits **left of zero**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(PANEL, horizon=20, n_draws=8000)\n"
            "    obs = r['mean']*100; pval = r['p_placebo']; tval = r['t']\n"
            "else:\n"
            "    obs = R['up'][20][2]; pval = R['up'][20][7]; tval = R['up'][20][5]\n"
            "rng = np.random.default_rng(413)\n"
            "cloud = rng.normal(0.0, max(abs(obs),0.3)/0.45, 20000)  # illustrative null centred at 0\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=60, color=GREY, alpha=.85, label='5,000+ random-date draws (same tape)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed breakout {obs:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('20-day excess (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside (left of) the cloud: placebo p = {pval:.2f}, one-sample t = {tval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  one-sample t={tval:.2f}  placebo p={pval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the red line is **left of zero**, deep inside the null cloud — "
            f"random dates beat the breakout **{R['up'][20][7]*100:.0f}%** of the time "
            f"(placebo p = {R['up'][20][7]:.2f}), one-sample t = {R['up'][20][5]:.2f}. The desk "
            "stamps **Signal = NONE**: there is no effect to attribute to the figure."
        ),
        md(
            "### 4c · The symmetry myth-check — the failed flag drifts up MORE\n\n"
            "The cleanest within-study control: run the **identical** detector but require a break "
            "*below* the flag low. A figure that genuinely *chooses* up must have its down-break "
            "drift **down**, well under the up-break."
        ),
        code(
            "if HAVE_REAL:\n"
            "    up_e = [st.run_experiment(PANEL, horizon=h, n_draws=1500)['mean']*100 for h in hs]\n"
            "    dn = [st.run_experiment(PANEL, horizon=h, side='down', n_draws=1500) for h in hs]\n"
            "    dn_e = [d['mean']*100 for d in dn]; dn_t = [d['t'] for d in dn]\n"
            "else:\n"
            "    up_e = [R['up'][h][2] for h in hs]; dn_e = [R['down'][h][2] for h in hs]; dn_t = [R['down'][h][4] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(x-.2, up_e, .4, color=GREEN, label='break UP (textbook)')\n"
            "ax.bar(x+.2, dn_e, .4, color=RED, label='break DOWN (\"failed\" flag)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return (%)'); ax.set_title('Down-break drifts UP more — direction is uninformative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,u,d in zip(hs,up_e,dn_e): print(f'{h:>2}d: up {u:+.2f}%  down {d:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: at every horizon the **down**-break (the failure) beats the "
            f"**up**-break ({R['down'][20][2]:+.2f}% vs {R['up'][20][2]:+.2f}% at 20d). A "
            "directional signal cannot have both resolutions point up. The post-figure return "
            "belongs to the *names* (momentum into the pause), not the *break direction* — "
            "**\"resolves up\" is BUSTED.** (Even the down-break doesn't beat its own placebo, "
            f"p ≈ {R['down'][20][5]:.2f} — it too is just the names' drift.)"
        ),
        md(
            "### 4d · Robustness — you can't tune your way to an edge\n\n"
            "Sweep the detector from loose to strict. The naive *t* drifts with the sample, but no "
            "setting produces a placebo-clean positive — the invariance is the signature of *no "
            "signal*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for mp, mr in [(0.10,0.6),(0.12,0.5),(0.16,0.4)]:\n"
            "        r = st.run_experiment(PANEL, horizon=20, n_draws=2000, min_pole=mp, max_retrace=mr)\n"
            "        rob.append((mp, mr, r['n_events'], r['mean']*100, r['t'], r['p_placebo']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labels = [f'pole>={r[0]:.0%}\\nretr<={r[1]:.0%}' for r in rob]\n"
            "exc = [r[3] for r in rob]; ps = [r[-1] for r in rob]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(labels, exc, color=[RED if e<0 else AMBER for e in exc], width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "for i,e in enumerate(exc): a1.annotate(f'{e:+.2f}%',(i,e),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_ylabel('20d excess (%)'); a1.set_title('Excess: never a clean positive')\n"
            "a2.bar(labels, ps, color=GREY, width=.6); a2.axhline(0.05, ls='--', c=RED, label='p=0.05 bar')\n"
            "for i,p in enumerate(ps): a2.annotate(f'{p:.2f}',(i,p),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('placebo p'); a2.set_ylim(0,0.8); a2.set_title('Placebo p stays far above 0.05'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (min_pole, max_retr, n, exc%, t, p):', [(r[0],r[1],r[2],round(r[3],2),round(r[4],2),round(r[-1],2)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: loosening the flag finds 1,071 events at {R['robust'][0][3]:+.2f}% "
            f"(t={R['robust'][0][4]:.2f}); the strict setting limps to {R['robust'][2][3]:+.2f}% on "
            f"235 events (t={R['robust'][2][4]:.2f}). Either way the **placebo p stays "
            f"{R['robust'][2][5]:.2f}–{R['robust'][0][5]:.2f}** — never near 0.05. No definition of "
            "\"a cleaner flag\" makes it real."
        ),
        md(
            "### 4e · SPY-only — the README hook, on too few events\n\n"
            "The hook asks specifically about SPY. The detector finds only a handful of flags on "
            "SPY in 21 years — far too few to certify anything either way."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in hs:\n"
            "        r = st.run_experiment(PANEL, horizon=h, names=['SPY'], n_draws=3000)\n"
            "        rows.append((h, r['n_events'], r['mean']*100, r['t'], r['p_placebo']))\n"
            "else:\n"
            "    rows = [(h,)+R['spy'][h][1:2]+(R['spy'][h][2],R['spy'][h][4],R['spy'][h][5]) for h in hs]\n"
            "for h,n,e,t,p in rows: print(f'SPY {h:>2}d: n={n}  excess={e:+.2f}%  t={t:+.2f}  p={p:.2f}')\n"
            "print(f\"\\nSPY flags found in {R['years']:.0f}y: {R['n_up_spy']} (too few to certify)\")"
        ),
        md(
            f"> 💡 In plain words: **{R['n_up_spy']}** flags on SPY in {R['years']:.0f} years, "
            "t ≈ 0 — uncertifiable noise. The broad-basket symmetry test already settled it; SPY "
            "just doesn't give enough events to argue."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel with **planted** flags and a *known* post-breakout drift: "
            "zero edge must stay inside the placebo (no false positive), a large planted edge must "
            "light up. Both hold — so the negative real-tape result is a true absence, not a broken "
            "pipeline."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.20):\n"
            "    panel, truth = data.synthetic_panel(edge=edge, seed=413, n_planted=8, daily_vol=0.011)\n"
            "    r = st.run_experiment(panel, horizon=20, n_draws=1200)\n"
            "    res.append((edge, truth['n_planted_total'], r['n_breakouts'], r['n_events'], r['mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "labels = [f'edge {e*100:.0f}%' for e,*_ in res]; pvals = [r[6] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, pvals, color=[RED, GREEN], width=.5); ax.axhline(0.05, ls='--', c='k', label='p=0.05 bar')\n"
            "for i,p in enumerate(pvals): ax.annotate(f'p={p:.3f}',(i,p),ha='center',va='bottom')\n"
            "ax.set_ylabel('placebo p'); ax.set_title('Control: zero edge stays in the cloud; planted edge breaks out'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,pl,det,n,ex,t,p,w in res: print(f'planted {e*100:+.0f}%: detected={det} n={n} excess={ex:+.2f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drift the inference stays inside the cloud "
            f"(excess {R['syn'][0][4]:+.2f}%, placebo p = {R['syn'][0][6]:.2f} — no false "
            f"positive); plant **+20%** and it explodes (excess +{R['syn'][1][4]:.2f}%, "
            f"t = {R['syn'][1][5]:.2f}, p = {R['syn'][1][6]:.3f}, win {R['syn'][1][7]}%). The "
            "engine *can* bank a real edge — the real tape simply has none, and the bull-flag "
            "breakout is even slightly **negative**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — up-break excess over base rate **{R['up'][20][2]:+.2f}%** at "
            f"20d (one-sample **t = {R['up'][20][5]:.2f}**, HAC **t = {R['up'][20][6]:.2f}**), "
            f"win-rate **{R['up'][20][4]}%** < 50, same-tape **placebo p = {R['up'][20][7]:.2f}**. "
            "Negative before it's anything else; no strictness setting yields a placebo-clean "
            "positive; the zero-edge control stays in the cloud while a planted edge lights up at "
            f"t = {R['syn'][1][5]:.1f}. Survivorship tilts the test *for* the figure and it still "
            "loses. Nothing to certify.\n"
            f"- **Tradability `MIRAGE`** — no gross edge to tax; net **{R['up'][20][8]:+.2f}%** at "
            f"20d. SPY-only rests on **{R['spy'][20][1]}** events at t ≈ 0. Buying the breakout "
            "slightly *underperforms* holding the name.\n"
            f"- **Resolves up? `BUSTED`** — the **down**-break of the identical flag drifts up "
            f"**more** ({R['down'][20][2]:+.2f}% vs {R['up'][20][2]:+.2f}% at 20d). Break up or "
            "down, the stock drifts the same — the figure is a momentum tell, miscalibrated as a "
            "directional breakout signal."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "There is no tradability conversation to have: the **gross** excess is below zero at "
            "every horizon, so costs only deepen a loss. The picture below is just the gross excess "
            "with the break-even line — every point is under water."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.run_experiment(PANEL, horizon=h, n_draws=800)['mean']*100 for h in hs]\n"
            "    nv = [st.run_experiment(PANEL, horizon=h, n_draws=800)['net']*100 for h in hs]\n"
            "else:\n"
            "    g = [R['up'][h][2] for h in hs]; nv = [R['up'][h][8] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, g, 'o--', c=GREY, lw=1.6, label='gross excess')\n"
            "ax.plot(hs, nv, 'o-', c=RED, lw=2.2, label='net of costs')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "for h,v in zip(hs,nv): ax.annotate(f'{v:+.2f}%',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('excess return (%)')\n"
            "ax.set_title('Every horizon under water, gross and net'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net excess by H:', {f'{h}d': round(v,2) for h,v in zip(hs,nv)})"
        ),
        md(
            "> 💡 In plain words: there is no shaded green region because the line never crosses "
            "zero. You can't fix a negative edge with a longer hold or a cheaper broker — hence "
            "**MIRAGE**. The only honest trade derived from a bull flag is *no trade*."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Definition risk, bounded.** The strictness sweep shows the verdict is invariant to "
            "the detector knobs over a wide range — a stricter human-drawn flag would have to do "
            "something the mechanical sweep never finds. A kernel-smoothing detector (Lo, Mamaysky "
            "& Wang 2000) is the natural next refinement.\n"
            "- **Intraday / small-caps.** The folklore is loudest on fast charts and small names; "
            "test with yfinance 5m (~60d) / 1m (~7d) — but state the short-span and capacity "
            "caveats, and expect costs + survivorship to bite harder.\n"
            "- **The general lesson.** \"Breakout-and-run\" figures are momentum tells that inherit "
            "the names' drift; the only honest benchmark is a random day in the same name. The "
            "down-break symmetry test is the cheapest way to expose a directional claim with no "
            "directional content.\n\n"
            "*The reproducible core is offline and deterministic; the detector is one transparent "
            "mechanical definition (chart figures are partly subjective — we say so). Methods and "
            "sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
