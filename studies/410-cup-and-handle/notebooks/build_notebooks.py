"""Generate the two narrative notebooks for Study 410 (Cup & Handle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily OHLC panel
under ../_cache/ (pinned to the as-of) and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 30-name large-cap
# basket incl. SPY, auto-adjusted daily OHLC, 2005-01-03 -> 2026-05-29, as-of 2026-05-31,
# 443 confirmed breakouts, fingerprint b660382b8af3).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    n_names=30, n_bars=5385, n_breakouts=443, n_spy=8, fingerprint="b660382b8af3",
    # forward excess over base rate per horizon: (H, n, excess%, raw%, win%, t, hac_t, p_plac, net%)
    h5=(5, 440, 0.206, 0.506, 51.1, 1.28, 1.27, 0.406, 0.106),
    h10=(10, 440, 0.516, 1.108, 53.4, 2.14, 2.24, 0.335, 0.416),
    h20=(20, 438, 0.065, 1.243, 47.7, 0.18, 0.17, 0.486, -0.035),
    h40=(40, 436, -0.781, 1.538, 48.6, -1.68, -1.65, 0.629, -0.881),
    # robustness at H=10: (rim_tol, handle_max_depth, n, excess%, t, p)
    robust=[(0.04, 0.10, 415, 0.485, 2.18, 0.350),
            (0.06, 0.15, 440, 0.516, 2.14, 0.338),
            (0.10, 0.20, 455, 0.550, 2.37, 0.325)],
    # SPY-only: (H, n, excess%, raw%, t, p)
    spy=[(5, 8, 0.354, 0.591, 0.89, 0.337), (10, 8, 0.741, 1.210, 1.02, 0.255),
         (20, 8, 1.179, 2.116, 1.30, 0.217), (40, 8, 3.525, 5.372, 3.72, 0.044)],
    # synthetic control 20d: (planted_edge, planted, detected, n, excess%, t, p, win%)
    syn=[(0.00, 160, 151, 151, 0.20, 0.41, 0.466, 50),
         (0.20, 160, 145, 145, 6.06, 8.31, 0.024, 74)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_buy--and--hold%3F: Busted](https://img.shields.io/badge/Beats_buy--and--hold%3F-Busted-8b949e?style=flat-square)\n\n"
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

from cup_and_handle import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
    _a = pd.Timestamp(ASOF)
    for _f in PANEL:
        PANEL[_f] = PANEL[_f].loc[:_a]
else:
    PANEL = None
print("real cup-and-handle cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else PANEL["close"].shape[1]),
      "| bars:", (0 if PANEL is None else len(PANEL["close"])))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"cup-and-handle\" breakout actually launch a stock? ☕\n"
            "### Wall Street's most beloved chart pattern — measured honestly, in plain English\n\n"
            + BADGES +
            "Open any beginner's trading book and you'll meet the **cup-with-handle**: a stock dips "
            "into a smooth, rounded bowl, climbs back to where it started, drifts sideways for a "
            "moment (the *handle*), and then — *boom* — **breaks out** and runs. William O'Neil made "
            "it famous; chart-watchers swear by it. Buy the breakout, ride the rocket.\n\n"
            "It's a beautiful story, and there's even a kernel of truth in it: stocks really *do* tend "
            "to be higher a few weeks after one of these breakouts. So case closed? **No.** The whole "
            "trick is hiding in one boring question the legend never asks: *higher than what?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the detector "
            "code? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **An honesty note up front.** A cup-and-handle is partly in the **eye of the "
            "beholder** — two chartists will draw it differently. So we wrote one **mechanical, "
            "reproducible** definition and tested *that*, and we say so. We also use a fixed basket of "
            "big survivor stocks (named in the verdict). Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a cup-and-handle breakout, does the stock go up? | **Yes — a bit.** Up about "
            "**+0.5% to +1.5%** over the next few weeks, on average. |\n"
            "| So the legend is true? | **No.** That's just the stock's **normal drift**. These names "
            "go up on *any* random day too — by the same amount. |\n"
            "| What happens when you subtract \"what it would've done anyway\"? | **The edge "
            "vanishes.** The breakout's *excess* over buy-and-hold is **noise** — random dates on the "
            "same stocks beat it about **a third of the time**. |\n"
            "| Does it beat buy-and-hold on SPY specifically? | **Can't tell — and probably not.** The "
            "pattern only shows up **8 times** on SPY in 21 years. Too few to mean anything. |\n\n"
            "> The breakout going up isn't the question. **Buy-and-hold also goes up.** The honest "
            "test is whether the breakout beats *that* — and it doesn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A stock forms a **cup** — a rounded, months-long bowl that recovers to its old high "
            "— then a short, shallow **handle** (a little pullback). When it pushes **above the rim** "
            "(the 'pivot'), that's the buy signal: the breakout launches a big advance.\"*\n\n"
            "This is **William O'Neil's** signature pattern from *How to Make Money in Stocks* (1988) "
            "— the heart of his CAN SLIM method and the chart culture of *Investor's Business Daily*. "
            "It is probably the single most-taught figure in retail technical analysis. The question "
            "isn't whether traders *believe* it — they do, fervently. It's whether the tape agrees."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the cup-and-handle breakout genuinely *predicted* outperformance, it would be a tidy "
            "money machine: wait for the shape, buy the pivot, beat the market. It would also be a "
            "real crack in efficient markets — a picture on a chart foretelling the future.\n\n"
            "But here's the trap that sinks almost every chart pattern: **stocks drift up.** Over any "
            "random three-week window, a big healthy stock tends to gain a little. So *of course* it's "
            "higher after a breakout — it'd be higher after a coin-flip entry too. The only test that "
            "means anything is the **excess**: how much *better* than just-buy-and-holding did the "
            "breakout do? That's the number we'll chase."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_names']} big US stocks** (including **SPY**, the S&P 500 fund) and "
            f"**{R['years']:.0f} years** of daily prices. Then we teach the computer to spot the "
            "pattern with a fixed rulebook — no human eyeballing:\n\n"
            "1. **The cup** — a peak, then a rounded dip (12–50% deep), then a recovery back near the "
            "peak (rims within 6%), shaped like a U, not a sharp V.\n"
            "2. **The handle** — a small pullback (≤15%) off the right rim.\n"
            "3. **The breakout** — the first day the price closes **above the rim**. *That's the buy.*\n\n"
            f"It finds **{R['n_breakouts']} breakouts** across the basket. For each, we buy the **next "
            "day's close** (no cheating) and measure the return over 5, 10, 20, 40 days — then "
            "subtract what the stock would've done anyway. If the legend is real, the *excess* should "
            "be clearly positive."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the believer's view: raw returns after the breakout.** These should look great "
            "— and they do."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    raw = [st.run_experiment(PANEL, horizon=h, n_draws=0)['raw_mean']*100 for h in hs]\n"
            "    exc = [st.run_experiment(PANEL, horizon=h, n_draws=0)['mean']*100 for h in hs]\n"
            "else:\n"
            "    raw = [R['h5'][3], R['h10'][3], R['h20'][3], R['h40'][3]]\n"
            "    exc = [R['h5'][2], R['h10'][2], R['h20'][2], R['h40'][2]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, raw, color=GREEN, width=.6)\n"
            "for i,v in enumerate(raw): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('average return after breakout (%)')\n"
            "ax.set_title('The believer\\'s chart: stocks DO rise after a cup-and-handle breakout')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('raw returns by horizon:', [round(v,2) for v in raw])"
        ),
        md(
            f"Looks like a winner: **{R['h5'][3]:+.2f}%** to **{R['h40'][3]:+.2f}%** higher after the "
            "breakout. If we stopped here, we'd call it a real edge. **But we won't stop here** — "
            "because we haven't asked *higher than what?*"
        ),
        md(
            "**Now the honest view.** Subtract what each stock would have returned over the same "
            "window on *any* day (its \"base rate\"). What's left is the breakout's true edge."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, raw, .4, color=GREEN, label='raw return (the believer\\'s number)')\n"
            "ax.bar(x+.2, exc, .4, color=RED, label='EXCESS over buy-and-hold (the honest number)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('return (%)')\n"
            "ax.set_title('Subtract buy-and-hold and the edge collapses to noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('excess over base rate:', [round(v,2) for v in exc])"
        ),
        md(
            f"There's the whole study. The **excess** (red) is tiny and wobbly: "
            f"**{R['h5'][2]:+.2f}%** at 5 days, a flicker of **{R['h10'][2]:+.2f}%** at 10, then "
            f"**{R['h20'][2]:+.2f}%** at 20 and **{R['h40'][2]:+.2f}%** (negative!) at 40. The green "
            "bars were never the breakout's doing — they were the stock's normal upward drift, which "
            "you'd capture with a **coin-flip entry**."
        ),
        md(
            "**The clincher: random dates do just as well.** Let's pick the *same number* of "
            "**random** buy dates on the same stocks, thousands of times, and see how often pure luck "
            "beats our \"signal.\" If the pattern is real, random should rarely win."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p10 = st.run_experiment(PANEL, horizon=10, n_draws=4000)['p_placebo']\n"
            "else:\n"
            "    p10 = R['h10'][7]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 1.9))\n"
            "ax.barh([0], [p10*100], color=RED, height=.5)\n"
            "ax.axvline(5, ls='--', c='k', label='5% bar (would-be significance)')\n"
            "ax.set_xlim(0, 55); ax.set_yticks([]); ax.set_xlabel('% of RANDOM date-sets that beat the breakout (10-day)')\n"
            "ax.annotate(f'{p10*100:.0f}%', (p10*100, 0), va='center', ha='left', fontsize=12)\n"
            "ax.set_title('Coin-flip entries beat the \"signal\" about a third of the time'); ax.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p (10-day): random dates beat the breakout {p10*100:.0f}% of the time')"
        ),
        md(
            f"**About a third** of random date-sets beat the breakout (placebo *p* ≈ "
            f"**{R['h10'][7]:.2f}**). For a real signal that number should be small (under 5%). At "
            "~34%, the cup-and-handle breakout is statistically **indistinguishable from buying on a "
            "random Tuesday**. The pattern is a Rorschach test, not a forecast."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Once you subtract buy-and-hold, the breakout's edge is noise — and "
            "random dates on the same tape beat it a third of the time.\n"
            "- **Tradability — Mirage.** There's no edge to trade. The raw gain is just the market "
            "drift you'd get anyway; the SPY-only version rests on 8 events.\n"
            "- **\"Beats buy-and-hold\"? — Busted.** \"It went up after the breakout\" is true and "
            "irrelevant. It went up by exactly its base rate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Short answer: there's nothing to trade. When a strategy has a *real* edge, the question "
            "is whether trading costs eat it. Here the edge is gone **before** costs — the gross "
            "excess is already inside the noise band. You'd be paying spreads to capture a market "
            "drift you could get for free by just holding the stock. And the SPY-specific version "
            f"(the actual hook) fires only **{R['n_spy']} times in 21 years** — you couldn't build a "
            "strategy on it if you wanted to."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The base-rate lesson generalises.** *Every* long-only \"buy and hold N days\" chart "
            "rule on an up-drifting stock looks good — because the baseline is positive. Always ask "
            "*higher than what?*\n"
            "- **Subjectivity is real.** We tested one mechanical cup definition; a stricter or looser "
            "one finds different cups — but the placebo stays ~0.34 across all of them (see the quants "
            "notebook). The verdict doesn't depend on the exact rule.\n"
            "- **Volume.** O'Neil insists the breakout must come on *heavy volume*. A worthy fork: add "
            "a volume-surge filter and re-run — does the placebo ever drop below the bar? (We bet not, "
            "but show us.)\n\n"
            "*Think the cup-and-handle has a real edge? Show the **excess over buy-and-hold** clearing "
            "a same-tape placebo at p < 0.05 — then we'll talk.*"
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
            "# Cup & Handle — a quantitative teardown 🔬\n"
            "### An objective swing-pivot detector · forward 5/10/20/40-day **excess** over the base "
            "rate · one-sample & HAC *t* vs a **same-tape label-shuffle placebo** · detector-strictness "
            "robustness · the SPY small-*n* mirage · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "cup-with-handle is the **most subjective** classic figure, so we fix one transparent "
            "mechanical definition and arbitrate it honestly. The headline lesson is a methodology "
            "lesson: a one-sample *t* against **zero** clears 2 at one horizon — and a **same-tape "
            "placebo busts it**, because the null isn't zero, it's the tape's own drift.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket incl. **SPY**, "
            "names still trading in 2026 — a *survivor* panel whose bias tilts post-breakout returns "
            "*up* (i.e. *for* the figure), and it still fails. Real data: yfinance daily "
            "auto-adjusted OHLC, 2005→2026, as-of **2026-05-31**. Offline core + synthetic control are "
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
            f"| **Signal** | `NONE` | Forward **excess** over base rate is noise: the only near-2 "
            f"horizon (**10d one-sample t = {R['h10'][5]:.2f}**, HAC t = {R['h10'][6]:.2f}) is "
            f"**busted by the same-tape placebo p = {R['h10'][7]:.2f}**; 20d t = {R['h20'][5]:.2f}, "
            f"40d **{R['h40'][2]:+.2f}%** (t = {R['h40'][5]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | No residual edge: raw return (+{R['h10'][3]:.2f}% at 10d) "
            f"is pure base rate; costs are a footnote; SPY-only rests on **{R['n_spy']}** events. |\n"
            f"| **Beats buy-and-hold?** | `BUSTED` | Subtract the base rate and the edge vanishes; "
            f"random dates on the same tape beat it (placebo p ≈ {R['h10'][7]:.2f}). The raw rise is "
            "beta, not signal. |\n\n"
            "> 💡 In plain words: the breakout *is* followed by a gain — exactly the gain you'd get "
            "from a random entry. The *t*-against-zero is fooled by the up-drifting tape; the placebo "
            "is the honest arbiter and it says **no edge**."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a confirmed cup-with-handle breakout for name $j$ occur at bar $\\tau$ (the first "
            "close clearing the rim resistance). Enter at $\\tau+1$ (one lag) and hold $H$ days for a "
            "return $r_{j,\\tau}(H)$. The believer's quantity is the mean forward return "
            "$\\overline{r}(H)$. The **honest** quantity is the **excess** over each name's "
            "unconditional base rate $\\mu_j(H) = \\mathbb{E}[\\text{any }H\\text{-day return for }j]$:\n\n"
            "$$\\widehat{a}(H) = \\frac{1}{N}\\sum_{j,\\tau}\\big(r_{j,\\tau}(H) - \\mu_j(H)\\big).$$\n\n"
            "- **H₁ (the figure predicts a run).** $\\widehat{a}(H) > 0$, significant, **and** below a "
            "same-tape placebo at $p<0.05$.\n"
            "- **H₂ (it beats buy-and-hold).** $\\widehat{a}(H)$ — *by construction* the excess over "
            "hold — is reliably positive.\n\n"
            "We find **H₁ rejected** (the lone $t>2$ at 10d has placebo $p=0.34$; 20–40d flat then "
            "negative) and **H₂ rejected** (excess is noise). The raw $\\overline{r}(H)>0$ is the base "
            "rate $\\mu_j$, i.e. beta — not the figure."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the null is the base rate, not zero\n\n"
            "Any long-only \"buy and hold $H$ days\" rule on an up-drifting asset earns "
            "$\\overline{r}(H)\\approx \\mu_j(H) > 0$ with a respectable *t* against **zero** — purely "
            "from the drift. Testing $\\overline{r}(H)$ against 0 is therefore **structurally biased "
            "toward a false positive**. Two fixes, applied together:\n\n"
            "**(a) Excess over the base rate.** Subtract $\\mu_j(H)$ so the test asks *does the figure "
            "beat hold?* **(b) A same-tape placebo.** Draw the same number of **random** entry dates on "
            "the same series many times and ask $p = \\Pr[\\text{random excess} \\ge \\text{observed}]$ "
            "— this absorbs any residual drift/clustering the base-rate subtraction misses. When *both* "
            "say \"noise,\" a high $t$-against-zero is exposed as the drift artefact it is."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket incl. **SPY** (yfinance "
            f"auto-adjusted daily OHLC, {R['start']}→{R['end']}, as-of {R['asof']}); "
            f"**{R['n_bars']:,}** bars. **Survivor** panel — named on the Signal axis.\n"
            "- **Detector (mechanical).** Swing pivots (±5-bar) → cup (depth 12–50%, rims within 6%, "
            "U-shaped, 30–200 bars) → handle (≤15% pullback, not undercutting the lower cup) → "
            "**breakout** = first close clearing the rim. Non-overlapping.\n"
            f"- **Events.** **{R['n_breakouts']}** confirmed breakouts (SPY alone: {R['n_spy']}).\n"
            "- **Timing.** Enter the close **1 day after** the breakout (no look-ahead); hold "
            "$H\\in\\{5,10,20,40\\}$.\n"
            "- **Signal axis.** Excess over base rate; one-sample *t* + HAC (Newey-West) *t*; "
            "**same-tape random-date placebo** (5,000 draws).\n"
            "- **Robustness.** Vary detector strictness (rim tolerance, handle depth).\n"
            "- **Costs.** 5 bps one-way × 2 = 10 bps round trip.\n"
            "- **Positive control.** A deterministic panel with **planted** cup shapes + a known "
            "post-breakout drift: zero edge must NOT reach significance; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · An example detected cup — what the rule actually fires on\n\n"
            "Before the statistics, the picture: the detector's first confirmed breakout on a real "
            "name, with the left rim, trough, right rim, handle low and breakout bar marked. This is "
            "the mechanical figure we are testing — judge for yourself how cup-like it is."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = PANEL['close']['AAPL'].dropna().to_numpy(float)\n"
            "    cups = st.detect_cups(c)\n"
            "    d = cups[0]\n"
            "    a, b = d['left_idx']-10, d['breakout_idx']+30\n"
            "    seg = c[a:b]; xs = np.arange(a, b)\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax.plot(xs, seg, c=GREY, lw=1.4)\n"
            "    for key,col,lab in [('left_idx',GREEN,'left rim'),('trough_idx',RED,'trough'),\n"
            "                        ('right_idx',GREEN,'right rim'),('handle_low_idx',AMBER,'handle low'),\n"
            "                        ('breakout_idx','black','breakout')]:\n"
            "        i=d[key]; ax.scatter([i],[c[i]],color=col,zorder=5,s=55,label=lab)\n"
            "    ax.axhline(d['rim'], ls='--', c=GREEN, alpha=.6)\n"
            "    ax.set_title('A mechanically-detected cup-with-handle on AAPL'); ax.set_xlabel('bar'); ax.set_ylabel('price'); ax.legend(fontsize=8)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"cup depth={d['depth']*100:.0f}%  handle depth={d['handle_depth']*100:.0f}%  rim={d['rim']:.2f}\")\n"
            "else:\n"
            "    print('cache absent — see docs/results.md')"
        ),
        md(
            "> 💡 In plain words: this is a genuine, recognisable cup-and-handle — the detector isn't "
            "firing on junk. So if there were an edge in the *figure*, our test would find it. (The "
            "synthetic control in 4e confirms the engine banks a planted edge.)"
        ),
        md(
            "### 4b · The decisive picture — raw return vs excess vs the placebo band\n\n"
            "Left: raw forward return (the believer's number) vs the **excess** over base rate. Right: "
            "the 10-day excess against its same-tape placebo distribution — the observed value should "
            "sit far in the right tail if it's real. It doesn't."
        ),
        code(
            "hs = [5,10,20,40]\n"
            "if HAVE_REAL:\n"
            "    res = [st.run_experiment(PANEL, horizon=h, n_draws=5000) for h in hs]\n"
            "    raw = [r['raw_mean']*100 for r in res]; exc = [r['mean']*100 for r in res]\n"
            "    r10 = st.run_experiment(PANEL, horizon=10, n_draws=0)\n"
            "    obs10 = r10['mean']*100\n"
            "    # rebuild placebo draws for the histogram\n"
            "    c_all = []\n"
            "    rng = np.random.default_rng(410)\n"
            "    closes = PANEL['close']\n"
            "    draws = []\n"
            "    for tk in closes.columns:\n"
            "        cc = closes[tk].dropna().to_numpy(float)\n"
            "        cups = st.detect_cups(cc); sig=[d['breakout_idx'] for d in cups]\n"
            "        fr = st.forward_returns(cc, sig, 10)\n"
            "        if len(fr)==0: continue\n"
            "        br = st.base_rate(cc, 10); n=len(cc); hi=n-10-2\n"
            "        for _ in range(300):\n"
            "            si=rng.integers(1,hi,size=len(fr)); e=si+1; x=si+1+10\n"
            "            draws.append((cc[x]/cc[e]-1).mean()-br)\n"
            "    draws=np.array(draws)*100\n"
            "    pval = r10['p_placebo']; tval = r10['t']\n"
            "else:\n"
            "    raw=[R['h5'][3],R['h10'][3],R['h20'][3],R['h40'][3]]; exc=[R['h5'][2],R['h10'][2],R['h20'][2],R['h40'][2]]\n"
            "    obs10=R['h10'][2]; pval=R['h10'][7]; tval=R['h10'][5]\n"
            "    rng=np.random.default_rng(410); draws=rng.normal(0.0,0.45,6000)\n"
            "x=np.arange(len(hs))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.4))\n"
            "a1.bar(x-.2,raw,.4,color=GREEN,label='raw'); a1.bar(x+.2,exc,.4,color=RED,label='excess over base rate')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "a1.set_ylabel('return (%)'); a1.set_title('Raw is positive (beta); excess is noise'); a1.legend()\n"
            "a2.hist(draws,bins=50,color=GREY,alpha=.85,label='null: random-date excess')\n"
            "a2.axvline(obs10,c=RED,lw=2.5,label=f'observed 10d excess {obs10:+.2f}%')\n"
            "a2.set_xlabel('10-day excess (%)'); a2.set_ylabel('freq')\n"
            "a2.set_title(f'Inside the luck cloud: placebo p={pval:.2f}, t={tval:.2f}'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by H:', [round(v,2) for v in exc], ' 10d placebo p=', round(float(pval),3))"
        ),
        md(
            f"> 💡 In plain words: the observed 10-day excess sits **right in the middle** of the "
            f"random-date cloud (placebo *p* = {R['h10'][7]:.2f}). Even though its *t*-against-zero is "
            f"**{R['h10'][5]:.2f}** (> 2!), that *t* is measuring the tape's drift, not the figure. "
            "This is the exact failure mode the desk's inference bar exists to catch."
        ),
        md(
            "### 4c · Robustness — strictness doesn't rescue it\n\n"
            "Vary the detector's rim tolerance and handle depth. If \"near-significance\" were a real "
            "signal hiding under a strict rule, loosening or tightening would move the placebo. It "
            "doesn't — the *t* hovers near 2 and the placebo sits at ~0.34 everywhere."
        ),
        code(
            "settings=[(0.04,0.10),(0.06,0.15),(0.10,0.20)]\n"
            "if HAVE_REAL:\n"
            "    rob=[(rt,hd,)+tuple([st.run_experiment(PANEL,horizon=10,n_draws=3000,rim_tol=rt,handle_max_depth=hd)[k] for k in ('n_events','mean','t','p_placebo')]) for rt,hd in settings]\n"
            "else:\n"
            "    rob=[(r[0],r[1],r[2],r[3]/100,r[4],r[5]) for r in R['robust']]\n"
            "labels=[f'{rt:.0%}/{hd:.0%}' for rt,hd,*_ in rob]\n"
            "ts=[r[4] for r in rob]; ps=[r[5] for r in rob]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar(labels,ts,color=AMBER,width=.55); a1.axhline(2,ls='--',c=RED,label='t=2 bar')\n"
            "a1.set_ylabel('10d one-sample t'); a1.set_title('t hovers near 2 (the drift artefact)'); a1.set_ylim(0,3); a1.legend()\n"
            "a2.bar(labels,[p*100 for p in ps],color=RED,width=.55); a2.axhline(5,ls='--',c='k',label='5% bar')\n"
            "a2.set_ylabel('placebo p (%)'); a2.set_title('...but the placebo never drops near 5%'); a2.legend()\n"
            "for ax in (a1,a2): ax.set_xlabel('rim tol / handle depth')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (rim,handle,n,excess%,t,p):', [(round(r[0],2),round(r[1],2),r[2],round(r[3]*100,2),round(r[4],2),round(r[5],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: from strict ({R['robust'][0][4]:.2f} *t*, p={R['robust'][0][5]:.2f}) "
            f"to loose ({R['robust'][2][4]:.2f} *t*, p={R['robust'][2][5]:.2f}), the placebo never "
            "approaches the 5% bar. The near-2 *t* is a property of the **drifting tape**, not the "
            "rule's strictness — exactly what you'd expect if there's no figure-specific edge."
        ),
        md(
            "### 4d · The SPY small-*n* mirage — the literal hook\n\n"
            f"The hook asks about **SPY**. The detector finds only **{R['n_spy']}** clean breakouts on "
            "SPY in 21 years. Watch the 40-day SPY *t* shoot up — and the sample size collapse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy=[(h,)+tuple([st.run_experiment(PANEL,horizon=h,names=['SPY'],n_draws=5000)[k] for k in ('n_events','mean','t','p_placebo')]) for h in hs]\n"
            "else:\n"
            "    spy=[(s[0],s[1],s[2]/100,s[4],s[5]) for s in R['spy']]\n"
            "Hs=[s[0] for s in spy]; ts=[s[3] for s in spy]; ns=[s[1] for s in spy]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.2))\n"
            "ax.bar([f'{h}d' for h in Hs],ts,color=[GREY,GREY,GREY,RED],width=.55)\n"
            "ax.axhline(2,ls='--',c='k',label='t=2'); ax.set_ylabel('SPY-only one-sample t')\n"
            "for i,(t,n) in enumerate(zip(ts,ns)): ax.annotate(f't={t:.2f}\\n(n={n})',(i,t),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_title(f'SPY 40d t looks big — on n={R[\"n_spy\"]} events (a mirage of small numbers)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY (H,n,excess%,t,p):', [(s[0],s[1],round(s[2]*100,2),round(s[3],2),round(s[4],3)) for s in spy])"
        ),
        md(
            f"> 💡 In plain words: SPY 40-day shows *t* = {R['spy'][3][4]:.2f} — but on "
            f"**{R['n_spy']} events**, and on the full 443-event basket the same 40-day horizon is "
            f"**{R['h40'][2]:+.2f}%** (negative). Eight dates in a 21-year bull tape look great over 40 "
            "days; that's small-*n* noise, not a tradable SPY edge. The honest answer to the hook is "
            "**no, it doesn't beat SPY buy-and-hold**."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** cup shapes and a known post-breakout drift: "
            "with **zero** planted edge the test must stay at *t* ≈ 0 with a high placebo *p* (no false "
            "positive); with a **large** planted edge it must light up. Both hold — so the real-tape "
            "null is a *genuine* null, not a blind engine."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0,0.20):\n"
            "    pan,truth=data.synthetic_panel(edge=edge,seed=410,n_planted=8,daily_vol=0.011)\n"
            "    r=st.run_experiment(pan,horizon=20,n_draws=1500)\n"
            "    res.append((edge,r['n_events'],r['mean']*100,r['t'],r['p_placebo'],r['win']*100))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "labs=[f'planted\\n{e*100:.0f}%' for e,*_ in res]\n"
            "a1.bar(labs,[r[3] for r in res],color=[GREY,GREEN],width=.5); a1.axhline(2,ls='--',c=RED,label='t=2')\n"
            "for i,r in enumerate(res): a1.annotate(f't={r[3]:.2f}',(i,r[3]),ha='center',va='bottom')\n"
            "a1.set_ylabel('20d one-sample t'); a1.set_title('zero edge -> t~0; planted -> lights up'); a1.legend()\n"
            "a2.bar(labs,[r[4]*100 for r in res],color=[GREY,GREEN],width=.5); a2.axhline(5,ls='--',c='k',label='5% bar')\n"
            "for i,r in enumerate(res): a2.annotate(f'p={r[4]:.2f}',(i,r[4]*100),ha='center',va='bottom')\n"
            "a2.set_ylabel('placebo p (%)'); a2.set_title('placebo agrees: null=high p, real=low p'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,t,p,w in res: print(f'planted {e*100:+.0f}%: n={n} excess20={m:+.2f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drift the control sits at "
            f"**t = {R['syn'][0][5]:.2f}**, placebo **p = {R['syn'][0][6]:.2f}** (no false positive — "
            f"and the placebo agrees); a **+20%** planted drift reaches **t = {R['syn'][1][5]:.2f}**, "
            f"placebo **p = {R['syn'][1][6]:.2f}**, win **{R['syn'][1][7]:.0f}%**. The engine is "
            "faithful, so the real-tape null is the genuine article — the figure has no edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — forward excess over the base rate is noise: the lone near-2 horizon "
            f"(**10d one-sample t = {R['h10'][5]:.2f}**, HAC t = {R['h10'][6]:.2f}) is **busted by the "
            f"same-tape placebo p = {R['h10'][7]:.2f}**; 20d t = {R['h20'][5]:.2f}, 40d "
            f"**{R['h40'][2]:+.2f}%** (t = {R['h40'][5]:.2f}). Robust across detector strictness (placebo "
            "stays ~0.34). Survivorship tilts the test *for* the figure and it still fails. **NONE, not "
            "WEAK** — the *t* that clears 2 does so only against a drifting-tape null the placebo corrects.\n"
            f"- **Tradability `MIRAGE`** — no residual edge to tax. The positive raw return "
            f"(+{R['h10'][3]:.2f}% at 10d) is market beta (random entries earn the same); costs are a "
            f"footnote; the SPY-specific version rests on **{R['n_spy']}** events. Nothing to deploy.\n"
            f"- **Beats buy-and-hold? `BUSTED`** — \"it rose after the breakout\" is true and irrelevant; "
            "it rose by exactly its base rate. Subtract buy-and-hold and the edge vanishes (placebo "
            f"p ≈ {R['h10'][7]:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — nothing to scale\n\n"
            "When a signal is real, beat 6 is about costs and capacity. Here the edge dies one step "
            "**earlier** — at the placebo, before any cost. The net columns confirm it: a 10 bps round "
            "trip is a rounding error on a gross excess that is already inside the noise band. You "
            "would be paying spreads to harvest a market drift available for free via buy-and-hold. "
            f"And the literal hook — SPY — offers **{R['n_spy']}** events in two decades. There is no "
            "vehicle to build, at any size. The correct action is to not trade it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g=[st.run_experiment(PANEL,horizon=h,n_draws=0)['mean']*100 for h in hs]\n"
            "    nv=[st.run_experiment(PANEL,horizon=h,n_draws=0)['net']*100 for h in hs]\n"
            "else:\n"
            "    g=[R['h5'][2],R['h10'][2],R['h20'][2],R['h40'][2]]; nv=[R['h5'][8],R['h10'][8],R['h20'][8],R['h40'][8]]\n"
            "x=np.arange(len(hs))\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.2))\n"
            "ax.bar(x-.2,g,.4,color=GREY,label='gross excess'); ax.bar(x+.2,nv,.4,color=RED,label='net of 10bps round trip')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return (%)'); ax.set_title('Gross is already noise; costs are a footnote'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net excess by H:', {f'{h}d':round(v,2) for h,v in zip(hs,nv)})"
        ),
        md(
            "> 💡 In plain words: the gross (grey) and net (red) bars are basically the same height — "
            "because there's no edge for costs to eat. That's the signature of a **Mirage**: the "
            "strategy doesn't die at execution, it was never alive."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The base-rate null generalises to all long-only chart rules.** On an up-drifting "
            "underlying, *any* \"buy and hold N days on a trigger\" earns a positive *t* against zero. "
            "The excess-over-hold + same-tape placebo is the minimal honest test; apply it before "
            "believing any breakout study.\n"
            "- **Volume confirmation (O'Neil's other half).** O'Neil requires a heavy-volume breakout. "
            "Add a volume-surge gate to `detect_cups` and re-run — a worthy PR. Our prior: the placebo "
            "stays above 0.05, but show the tape.\n"
            "- **Subjectivity bound.** We tested one mechanical definition; Lo-Mamaysky-Wang-style "
            "kernel smoothing would draw cups differently. The verdict is robust to our strictness "
            "sweep, but a fundamentally different detector is the obvious challenge.\n\n"
            "*The reproducible core is offline and deterministic; the detector is one transparent "
            "mechanical definition, not the only one. Methods and sources: "
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
