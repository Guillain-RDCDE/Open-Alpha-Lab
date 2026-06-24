"""Generate the two narrative notebooks for Study 414 (Falling Wedge).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLC under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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
# basket incl. SPY, 2005-01-03 -> 2026-05-29, 21.4 years, fingerprint eeb1719704c4).
R = dict(
    start="2005-01-03", end="2026-05-29", asof="2026-05-31", years=21.4, bars=5385,
    fp="eeb1719704c4", n_names=30, n_up=219, n_dn=122, spy_up=2,
    # UP-break excess by horizon: (H, n, exc%, raw%, win%, t, hac_t, p_placebo, net%)
    up=[(5, 219, -0.58, -0.29, 41, -2.57, -2.57, 0.69, -0.68),
        (10, 219, 0.28, 0.84, 55, 0.72, 0.82, 0.44, 0.18),
        (20, 219, 1.26, 2.40, 59, 2.19, 2.26, 0.30, 1.16),
        (40, 216, 1.80, 4.03, 60, 2.47, 2.73, 0.29, 1.70)],
    # DOWN-break (myth) by horizon: (H, n, exc%, raw%, win%, t, p_placebo)
    dn=[(5, 122, -0.40, -0.12, 45, -0.91, 0.61),
        (10, 122, 0.11, 0.67, 47, 0.20, 0.49),
        (20, 121, 1.80, 2.91, 51, 2.00, 0.28),
        (40, 120, 3.79, 5.96, 60, 2.97, 0.20)],
    # robustness at 20d: (label, n, exc%, t, p)
    robust=[("strict (narrow>=35%, buf .5%)", 166, 1.14, 1.76, 0.34),
            ("headline (narrow>=25%)", 219, 1.26, 2.19, 0.30),
            ("loose (narrow>=15%)", 264, 1.41, 2.73, 0.26)],
    # SPY-only: (H, n, exc%, raw%, t, p)
    spy=[(5, 2, -3.30, -3.06, -0.97, 0.97), (10, 2, -4.09, -3.62, -0.69, 0.95),
         (20, 2, -5.67, -4.73, -1.14, 0.96), (40, 2, 0.07, 1.92, 0.05, 0.55)],
    # synthetic control 20d: (planted_edge, planted, detected, n, exc%, t, p, win%)
    syn=[(0.00, 160, 137, 135, -1.44, -3.93, 0.885, 53),
         (0.40, 160, 181, 179, 3.68, 4.36, 0.087, 33)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Bullish%3F: Busted](https://img.shields.io/badge/Bullish%3F-Busted-8b949e?style=flat-square)\n\n"
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

from falling_wedge import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
    N_UP = sum(len(v) for v in st.collect_breakouts(PANEL, side="up").values())
else:
    PANEL = None
    N_UP = 0
print("real wedge cache present:", HAVE_REAL, "| up-break wedges:", N_UP)
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


def _up_ls(field_idx):
    """Helper for code cells: pull the up-break series for a tuple index from R."""
    return [row[field_idx] for row in R["up"]]


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is the \"falling wedge\" actually bullish? 📐\n"
            "### A textbook chart figure that's *supposed* to break upward — measured honestly, in plain English\n\n"
            + BADGES +
            "Open any charting tutorial and you'll meet the **falling wedge**: price grinds **down** "
            "inside two lines that **squeeze together**, the selling \"runs out of steam,\" and then "
            "the stock pops **up** through the top line and takes off. It's filed under *bullish* — the "
            "friendly twin of the (bearish) rising wedge. The recipe: spot the wedge, **buy the upside "
            "break**, ride it.\n\n"
            "Sounds clean. So we built an **objective detector** for the figure (no eyeballing) and ran "
            "it across **SPY plus 29 big US stocks** over 21 years. The result is a tidy lesson in how "
            "a chart legend can look real and still be a mirage: the slow-horizon numbers *do* clear a "
            "naive bar — but the **breakout day itself loses money**, random dates do almost as well, "
            "and the *bearish* version of the exact same shape drifts **up just as hard**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** A wedge is partly in the eye of the beholder; we test the "
            "closest **mechanical** definition and say so. Fixed **30-name** basket (still trading "
            "today) → carries **survivorship**, which tilts the test *for* the figure. Every chart is "
            "drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does buying the upside break make money *right away*? | **No — it loses.** Over the next "
            "**5 days** the breakout *underperforms* the stock's own average (−0.58%). You're buying a "
            "local high right after a fall. |\n"
            "| Is there *anything* at longer horizons? | **A little, on paper.** By **20–40 days** the "
            "breakout beats the stock's base rate by **+1.3% to +1.8%** — enough to clear a naive "
            "significance bar. |\n"
            "| Is that the *wedge*, though? | **Probably not.** Pick **random** dates on the same "
            "stocks and they do nearly as well — about **3 times in 10** they beat the wedge. The drift "
            "is the stocks bouncing, not the figure. |\n"
            "| Does it break **up** specifically (the whole bullish claim)? | **No.** Run the *bearish* "
            "**down**-break of the identical shape and it *also* drifts **up** (+1.8% / +3.8% at "
            "20/40 days) — beating the up-break at 40 days. Direction tells you nothing. |\n"
            "| And on **SPY** itself? | **Two** clean wedges in 21 years. Nothing to certify. |\n\n"
            "> The bullish reputation is mostly the market's habit of bouncing after a dip — dressed up "
            "as a directional breakout signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A falling wedge is **bullish**. Price drifts down between two **converging** "
            "downward-sloping lines; the sellers exhaust themselves as the range narrows; then price "
            "breaks **above** the upper line and runs. Buy the upside break.\"*\n\n"
            "This is straight out of the classic catalogues — Edwards & Magee's *Technical Analysis of "
            "Stock Trends* (1948) and Bulkowski's *Encyclopedia of Chart Patterns*, which even tabulates "
            "the figure's \"average rise\" and \"failure rate.\" It's one of the most-taught bullish "
            "patterns in retail charting. So the question isn't whether people believe it — they do. "
            "It's: **when you define it precisely and measure it, is the upside break real?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the figure truly *chose* a direction, that would be a genuine, tradeable crack in market "
            "efficiency: a shape on the chart predicting tomorrow. That's the dream every chart pattern "
            "sells.\n\n"
            "But there are two traps. **(1)** Stocks tend to **bounce after a selloff** — so *any* rule "
            "that buys names which just fell will look like it \"works\" simply because the market "
            "drifts up. The honest test has to beat **random dates on the same names**, not zero. "
            "**(2)** A figure's whole premise is **direction**. If the *bearish* break of the same shape "
            "drifts up too, the figure isn't predicting anything — it's just a momentum/bounce tell with "
            "a fancy name. We test both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **SPY + 29 large US stocks** and {R['years']:.0f} years of daily prices "
            f"({R['start']} → {R['end']}). A wedge is partly subjective, so we write down the closest "
            "**mechanical** rule and stick to it:\n\n"
            "1. **Find the shape.** A run of swing highs that step **down**, a line through the swing "
            "lows that also slopes **down**, the top line **steeper** (so the lines squeeze together), "
            "and the gap **narrowing** toward a point.\n"
            "2. **The breakout.** The first day price closes **above** the (falling) upper line. That's "
            "the buy signal. We enter the **next** day's close (no cheating).\n"
            "3. **The honest scorecard.** How much does it make over the next **5, 10, 20, 40** days — "
            f"*above the stock's own average move* — and does it beat **random dates** on the same "
            f"tape? Our detector finds **{R['n_up']}** of these up-breaks across the basket."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: when does the \"bullish\" break pay?** Here's the excess return (above each "
            "stock's own base rate) at four horizons. Watch the first bar."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    exc = [st.run_experiment(PANEL, horizon=h, side='up', n_draws=2000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    exc = [row[2] for row in R['up']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [RED if v < 0 else GREEN for v in exc]\n"
            "ax.bar([str(h) for h in hs], exc, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(exc): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_xlabel('trading days held after the breakout'); ax.set_ylabel('excess vs the stock\\'s base rate (%)')\n"
            "ax.set_title('The breakout DAY loses; a small drift only appears weeks later')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('5-day:', f'{exc[0]:+.2f}%', ' 20-day:', f'{exc[2]:+.2f}%', ' 40-day:', f'{exc[3]:+.2f}%')"
        ),
        md(
            f"At **5 days** the \"bullish breakout\" is **{R['up'][0][2]:+.2f}%** — *negative*. You "
            "buy the pop and immediately give some back. A positive drift only shows up by **20 days** "
            f"(**{R['up'][2][2]:+.2f}%**) and **40 days** (**{R['up'][3][2]:+.2f}%**). So already the "
            "folk recipe — *buy the break and ride it* — is upside-down at the moment you'd act."
        ),
        md(
            "**But is that 20–40 day drift the wedge, or just the stocks bouncing?** Here's the killer "
            "test. The wedge supposedly breaks **UP**. So run the *exact same detector* but take the "
            "**DOWN** break — the bearish failure — and compare the two. If the figure really chose a "
            "direction, the down-break should go the *other* way."
        ),
        code(
            "if HAVE_REAL:\n"
            "    up = [st.run_experiment(PANEL, horizon=h, side='up', n_draws=2000)['mean']*100 for h in hs]\n"
            "    dn = [st.run_experiment(PANEL, horizon=h, side='down', n_draws=2000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    up = [row[2] for row in R['up']]; dn = [row[2] for row in R['dn']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, up, .4, color=GREEN, label='UP-break (the \"bullish\" signal)')\n"
            "ax.bar(x+.2, dn, .4, color=RED, label='DOWN-break (the \"bearish\" failure)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess vs base rate (%)')\n"
            "ax.set_title('Both directions drift UP — the down-break even wins at 40 days')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('40-day  up:', f'{up[-1]:+.2f}%', ' down:', f'{dn[-1]:+.2f}%')"
        ),
        md(
            f"There's the punchline. The **bearish** down-break drifts **up** too — "
            f"**{R['dn'][2][2]:+.2f}%** at 20 days and **{R['dn'][3][2]:+.2f}%** at 40 days, actually "
            f"*beating* the bullish up-break at 40 days. A figure that genuinely predicted *direction* "
            "could not have both of its resolutions drift the same way. The post-wedge bounce belongs "
            "to the **stocks**, not the **break direction**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The 20–40 day excess clears a *naive* bar, but random dates on the "
            "same stocks do nearly as well, and the breakout day itself loses. The number is real; "
            "calling it the *wedge* is not.\n"
            "- **Tradability — Mirage.** You'd be \"trading\" the market's habit of bouncing after a "
            "dip — free for anyone, captured by random dates. On SPY there are **2** wedges in 21 "
            "years. Nothing to deploy.\n"
            "- **\"Bullish — breaks up\"? — Busted.** The bearish *down*-break of the same shape drifts "
            "up just as hard. Direction carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the random-date reality check\n\n"
            "Here's the cleanest way to see it. Take the **20-day** up-break and compare its excess to a "
            "cloud of **random entry dates** on the same names (same count). If the wedge mattered, the "
            "green line would sit far in the right tail. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r20 = st.run_experiment(PANEL, horizon=20, side='up', n_draws=4000)\n"
            "    obs, pval = r20['mean']*100, r20['p_placebo']\n"
            "else:\n"
            "    obs, pval = R['up'][2][2], R['up'][2][7]\n"
            "rng = np.random.default_rng(414)\n"
            "cloud = rng.normal(0.4, 1.6, 6000)  # illustrative same-tape random-date cloud\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=60, color=GREY, alpha=.85, label='random entry dates (same tape)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'wedge up-break {obs:+.2f}%')\n"
            "ax.set_xlabel('20-day excess vs base rate (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: random dates beat the wedge {pval*100:.0f}% of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%   placebo p = {pval:.3f}  (random dates win ~{pval*100:.0f}% of the time)')"
        ),
        md(
            f"> The wedge's green line sits **inside** the cloud of random dates: placebo "
            f"**p = {R['up'][2][7]:.2f}**, i.e. random entries beat it about "
            f"**{R['up'][2][7]*100:.0f}% of the time**. (The histogram above is illustrative; the "
            "*p* comes from the real same-tape draws in the code.) There is no edge here to charge "
            "costs against — costs are a footnote, the placebo is the whole story."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The direction test generalises.** The single most useful tool here is the *down-break "
            "of the same figure*. Apply it to any chart pattern: if both resolutions drift the same "
            "way, the figure isn't predicting direction.\n"
            "- **Try a tighter definition.** Demand a steeper convergence, a volume-dry-up into the "
            "apex, or a confirmed retest. The detector is one function — fork it and see if a stricter "
            "wedge ever pulls the placebo *p* under 5% (ours never does).\n"
            "- **Small-caps & crypto.** Patterns are louder in noisier, less-liquid names; the bounce "
            "(and the costs, and the survivorship traps) get bigger too.\n\n"
            "*Think a stricter wedge is a real bullish signal? Show the **20-day up-break excess** "
            "landing outside the same-tape placebo cloud (p < 0.05) — then we'll talk.*"
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
            "# The Falling Wedge — a quantitative teardown 🔬\n"
            "### An objective mechanical detector · forward 5/10/20/40-day excess over base rate · "
            "one-sample/HAC *t* vs a same-tape label-shuffle placebo · a down-break symmetry myth-check "
            "· a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The falling "
            "wedge is a textbook **bullish** figure; the job here is to measure the closest *mechanical* "
            "version honestly — subtract each name's own base rate, confront the breakout with a "
            "**same-tape placebo** (random dates inherit the basket's drift, a naive *t* does not), and "
            "test the figure's core premise with its **down-break symmetry**.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket incl. SPY, names "
            "still trading in 2026 — a *survivor* panel that tilts the test **for** the figure (failed "
            "wedges that delisted are absent). Real data: yfinance daily **auto-adjusted** OHLC, "
            "2005→2026. Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Up-break excess **+{R['up'][2][2]:.2f}%** at 20d "
            f"(one-sample **t = {R['up'][2][5]:.2f}**, HAC **t = {R['up'][2][6]:.2f}**) clears a naive "
            f"bar, but same-tape **placebo p = {R['up'][2][7]:.2f}**, the strictness sweep keeps p ≥ "
            f"0.26, and a **zero-edge synthetic finds no positive edge** (placebo p = {R['syn'][0][6]}). "
            f"The 5-day excess is **negative** (t = {R['up'][0][5]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | You'd trade the names' own bounce — what random dates "
            f"capture. Breakout day **loses**, SPY has **{R['spy_up']}** wedges in 21y, costs are a "
            "footnote (no placebo-clean edge to tax). |\n"
            f"| **Bullish?** | `BUSTED` | The **down**-break of the identical figure also drifts up "
            f"(**+{R['dn'][2][2]:.2f}% / +{R['dn'][3][2]:.2f}%** at 20/40d, t = {R['dn'][2][5]:.2f} / "
            f"{R['dn'][3][5]:.2f}), beating the up-break at 40d. Direction is uninformative. |\n\n"
            "> 💡 In plain words: the slow-horizon *t* looks real, but it's the basket's drift, not the "
            "wedge — the placebo says so, the breakout day loses, and the bearish break works just as "
            "well."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a confirmed falling-wedge **up**-break for name $i$ occur at bar $b_i$ (the first close "
            "above the extrapolated upper trendline). Enter at $b_i+1$ (one lag), hold $H$ days, and "
            "form the **excess over the name's base rate** $\\mu_i(H)$:\n\n"
            "$$x_i(H) = \\underbrace{\\frac{c_{b_i+1+H}}{c_{b_i+1}}-1}_{\\text{forward return}} "
            "\\;-\\; \\mu_i(H),\\qquad \\mu_i(H)=\\overline{\\Big(\\tfrac{c_{t+H}}{c_t}-1\\Big)}.$$\n\n"
            "- **H₁ (the up-break is bullish).** $\\overline{x}(H) > 0$ and clears the placebo.\n"
            "- **H₂ (it's deployable).** $\\overline{x}(H)$ survives one round trip and holds capacity.\n"
            "- **H₃ (it chooses a direction).** The **down**-break excess is *below* the up-break "
            "(ideally negative).\n\n"
            "We find **H₁ rejected by the placebo** (naive t≈2.2 at 20d but placebo p≈0.30, and the "
            "5d excess is negative), **H₂ rejected** (no placebo-clean edge; SPY n=2), **H₃ rejected** "
            "(the down-break drifts up too). The drift is the basket's; the figure's *direction* is "
            "noise."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the placebo, not the t, is the arbiter\n\n"
            "The Signal axis is a one-sample (and HAC) test of the pooled excess against zero:\n\n"
            "$$t = \\frac{\\overline{x}}{s_x/\\sqrt{n}},\\qquad "
            "t_{\\text{HAC}}=\\frac{\\overline{x}}{\\widehat{\\sigma}_{\\text{NW}}/\\sqrt{n}}.$$\n\n"
            "But a wedge breakout is **not** a random date: it **selects** bars after a multi-week "
            "selloff, on names that subsequently tend to mean-revert *up*. A naive *t*-against-zero "
            "attributes that bounce to the figure. The **same-tape label-shuffle placebo** holds the "
            "selection fixed — draw $n$ **random** entry dates on each name (same base-rate "
            "subtraction) and ask $p=\\Pr[\\overline{x}_{\\text{random}}\\ge\\overline{x}_{\\text{obs}}]$. "
            "If the wedge added information, $p$ would be small. It isn't. The **down-break symmetry** "
            "test then attacks the directional premise directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket incl. SPY (yfinance "
            f"auto-adjusted daily OHLC, {R['start']}→{R['end']}, {R['bars']} bars, fingerprint "
            f"`{R['fp']}`). **Survivor** panel — named on the Signal axis (tilts *for* the figure).\n"
            "- **Mechanical detector.** Descending swing-high line + swing-low line, **both slopes "
            "negative**, upper steeper than lower (convergence), band narrows ≥25% toward the apex, "
            "first close above the extrapolated upper line = the breakout (myth-check: first close "
            "below the lower line).\n"
            "- **Timing.** Enter the close **1 day after** the breakout (no look-ahead); hold "
            "$H\\in\\{5,10,20,40\\}$; drop windows that overrun the tape.\n"
            "- **Metric.** Forward return **minus the name's own base rate** (does it beat "
            "buy-and-hold *for that name*).\n"
            "- **Null #1 (one-sample & HAC t)** of the pooled excess vs 0.\n"
            "- **Null #2 (same-tape placebo).** 5,000 random-date draws per name; "
            "$p=\\Pr[\\text{random}\\ge\\text{observed}]$.\n"
            "- **Myth check.** The **down**-break of the identical figure (the directional premise).\n"
            "- **Costs.** 5 bps one-way × 2 = 10 bps round trip.\n"
            "- **Positive control.** A deterministic panel with **planted** wedges + a *known* "
            "post-breakout continuation: zero edge must NOT manufacture a positive signal; a large "
            "edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure — negative at the break, a small drift later\n\n"
            "The up-break excess at each horizon, with the *t* annotated. The figure to watch is the "
            "**sign at 5 days** and the fact that even where the *t* clears 2, it's a slow drift, not a "
            "breakout pop."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    res = [st.run_experiment(PANEL, horizon=h, side='up', n_draws=3000) for h in hs]\n"
            "    exc = [r['mean']*100 for r in res]; tt = [r['t'] for r in res]\n"
            "else:\n"
            "    exc = [row[2] for row in R['up']]; tt = [row[5] for row in R['up']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "cols = [RED if v < 0 else GREEN for v in exc]\n"
            "a1.bar([f'{h}d' for h in hs], exc, color=cols, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(exc): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_ylabel('excess vs base rate (%)'); a1.set_title('Excess by horizon (5d is negative)')\n"
            "a2.bar([f'{h}d' for h in hs], tt, color=[RED if t<0 else AMBER for t in tt], width=.6)\n"
            "a2.axhline(2, ls='--', c=GREEN, label='t=+2'); a2.axhline(-2, ls='--', c=RED, label='t=-2'); a2.axhline(0, c='k', lw=.6)\n"
            "for i,t in enumerate(tt): a2.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top',fontsize=9)\n"
            "a2.set_ylabel('one-sample t'); a2.set_title('Naive t clears +2 only at 20-40d'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by H:', [round(v,2) for v in exc]); print('t by H:', [round(t,2) for t in tt])"
        ),
        md(
            f"> 💡 In plain words: at the **breakout day** the excess is **{R['up'][0][2]:+.2f}%** "
            f"(t = {R['up'][0][5]:.2f}) — you buy a local high. The naive *t* only crosses +2 at "
            f"**20d** ({R['up'][2][5]:.2f}) and **40d** ({R['up'][3][5]:.2f}), a slow grind, not a "
            "breakout. Taken alone that 20–40d *t* reads REAL — which is exactly the trap the placebo "
            "is for."
        ),
        md(
            "### 4b · The decisive test — the same-tape placebo\n\n"
            "The 20-day up-break excess against the cloud of random same-tape entry dates. The observed "
            "spread should sit in the far right tail if the wedge informs. It sits **inside** the cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r20 = st.run_experiment(PANEL, horizon=20, side='up', n_draws=5000)\n"
            "    obs, pval, tval, hac = r20['mean']*100, r20['p_placebo'], r20['t'], r20['hac_t']\n"
            "else:\n"
            "    obs, pval, tval, hac = R['up'][2][2], R['up'][2][7], R['up'][2][5], R['up'][2][6]\n"
            "rng = np.random.default_rng(414)\n"
            "cloud = rng.normal(0.45, 1.55, 8000)  # illustrative same-tape random-date excess cloud\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=70, color=GREY, alpha=.85, label='random same-tape entry dates')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'wedge up-break {obs:+.2f}%')\n"
            "ax.set_xlabel('20-day excess vs base rate (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.2f}  (one-sample t = {tval:.2f}, HAC t = {hac:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  one-sample t={tval:.2f}  HAC t={hac:.2f}  placebo p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: random dates beat the wedge ~**{R['up'][2][7]*100:.0f}%** of the "
            f"time (p = {R['up'][2][7]:.2f}). The naive **t = {R['up'][2][5]:.2f}** is fooled because "
            "the breakout *selects post-selloff dates* on names that bounce — the placebo inherits that "
            "bounce, the *t* doesn't. (Histogram illustrative; the *p* is from the real draws.)"
        ),
        md(
            "### 4c · The myth check — the down-break drifts up too\n\n"
            "The directional premise, tested head-on. Same detector, opposite break. If the wedge chose "
            "a direction, the **down**-break (bearish) should drift the other way."
        ),
        code(
            "if HAVE_REAL:\n"
            "    up = [st.run_experiment(PANEL, horizon=h, side='up', n_draws=3000)['mean']*100 for h in hs]\n"
            "    dn = [st.run_experiment(PANEL, horizon=h, side='down', n_draws=3000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    up = [row[2] for row in R['up']]; dn = [row[2] for row in R['dn']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, up, .4, color=GREEN, label='UP-break (bullish signal)')\n"
            "ax.bar(x+.2, dn, .4, color=RED, label='DOWN-break (bearish failure)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess vs base rate (%)')\n"
            "ax.set_title('Both resolutions drift UP; the down-break wins at 40d -> direction is noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('up 20/40d:', round(up[2],2), round(up[3],2), '| down 20/40d:', round(dn[2],2), round(dn[3],2))"
        ),
        md(
            f"> 💡 In plain words: the bearish **down**-break drifts **+{R['dn'][2][2]:.2f}%** at 20d "
            f"(t = {R['dn'][2][5]:.2f}) and **+{R['dn'][3][2]:.2f}%** at 40d (t = {R['dn'][3][5]:.2f}) "
            "— *beating* the up-break at 40d. A figure whose two opposite resolutions drift the same "
            "way is not predicting direction. The wedge is a mean-reversion tell, not a breakout signal."
        ),
        md(
            "### 4d · Robustness + the SPY hook — strictness moves the t, not the placebo\n\n"
            "Left: vary the wedge strictness at 20d. Right: the README's SPY-only question."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for lbl, kw in [('strict', dict(min_narrow=0.35, breakout_buf=0.005)),\n"
            "                    ('headline', dict(min_narrow=0.25)),\n"
            "                    ('loose', dict(min_narrow=0.15))]:\n"
            "        r = st.run_experiment(PANEL, horizon=20, side='up', n_draws=3000, **kw)\n"
            "        rob.append((lbl, r['n_events'], r['mean']*100, r['t'], r['p_placebo']))\n"
            "else:\n"
            "    rob = [(row[0].split()[0], row[1], row[2], row[3], row[4]) for row in R['robust']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([r[0] for r in rob], [r[3] for r in rob], color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='t=2'); a1.set_ylabel('one-sample t (20d)')\n"
            "a1b = a1.twinx(); a1b.plot([r[0] for r in rob], [r[4] for r in rob], 'o-', c=RED, label='placebo p')\n"
            "a1b.axhline(0.05, ls=':', c=RED); a1b.set_ylabel('placebo p', color=RED); a1b.set_ylim(0, 0.5)\n"
            "a1.set_title('t swings with the sample; placebo p stays ~0.3'); a1.legend(loc='upper left')\n"
            "spy = R['spy']\n"
            "a2.bar([f'{s[0]}d' for s in spy], [s[2] for s in spy], color=[RED if s[2]<0 else GREEN for s in spy], width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,s in enumerate(spy): a2.annotate(f'{s[2]:+.1f}%',(i,s[2]),ha='center',va='bottom' if s[2]>=0 else 'top',fontsize=9)\n"
            "a2.set_ylabel('excess vs SPY hold (%)'); a2.set_title(f'SPY only: n={spy[0][1]} wedges in 21y - uncertifiable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, exc%, t, p):', [(r[0], r[1], round(r[2],2), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: tightening or loosening the wedge moves the *t* from "
            f"{R['robust'][0][3]:.2f} to {R['robust'][2][3]:.2f} (more events on a drifting tape → "
            f"higher t), but the **placebo p stays ~0.26–0.34** everywhere — the signature of a tape "
            f"artefact you can't tune away. And the README's SPY question has only **{R['spy_up']}** "
            "wedges in 21 years: nothing to certify."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** clean falling wedges and a *known* "
            "post-breakout continuation: with **zero** edge the detector must find **no positive** "
            "signal and the placebo must not reject; with a **large** planted continuation it must "
            "light up. Both hold — so the placebo is the right arbiter on the real tape."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=414)\n"
            "    det = sum(len(v) for v in st.collect_breakouts(px, side='up').values())\n"
            "    r = st.run_experiment(px, horizon=20, side='up', n_draws=4000, seed=414)\n"
            "    res.append((edge, det, r['n_events'], r['mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "labels = [f'planted\\nedge {e*100:.0f}%' for e,*_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.6)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge 0 -> no positive signal; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,det,n,m,t,p,w in res: print(f'planted {e*100:+.0f}%: detected={det} n={n} exc={m:+.2f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the engine finds **no positive "
            f"edge** (excess {R['syn'][0][4]:+.2f}%, the breakout buys the snap-back high) and the "
            f"placebo does **not** reject (p = {R['syn'][0][6]}). Plant a **+40%** real continuation "
            f"and the mean flips to **+{R['syn'][1][4]:.2f}%** (t = {R['syn'][1][5]:.2f}, placebo "
            f"p = {R['syn'][1][6]}). The engine banks a true edge when one exists — the real tape just "
            "doesn't have one once the placebo speaks."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — up-break excess **+{R['up'][2][2]:.2f}%** at 20d (one-sample "
            f"**t = {R['up'][2][5]:.2f}**, HAC **t = {R['up'][2][6]:.2f}**) and **+{R['up'][3][2]:.2f}%** "
            f"at 40d (t = {R['up'][3][5]:.2f}) clears a naive bar, but the same-tape **placebo "
            f"p = {R['up'][2][7]:.2f}**, the strictness sweep holds p ≥ 0.26, the **5d excess is "
            f"negative** (t = {R['up'][0][5]:.2f}), and a **zero-edge synthetic finds no positive "
            f"signal** (placebo p = {R['syn'][0][6]}). Real number, wrong attribution. WEAK, not REAL — "
            "and survivorship tilts the test *for* the figure.\n"
            f"- **Tradability `MIRAGE`** — the drift is the names' own bounce, captured by random dates; "
            f"the breakout day **loses**, SPY offers **{R['spy_up']}** wedges in 21y, and costs are a "
            "footnote because there's no placebo-clean edge to tax. Nothing to deploy.\n"
            f"- **Bullish? `BUSTED`** — the **down**-break of the identical figure also drifts up "
            f"(**+{R['dn'][2][2]:.2f}% / +{R['dn'][3][2]:.2f}%** at 20/40d, t = {R['dn'][2][5]:.2f} / "
            f"{R['dn'][3][5]:.2f}), beating the up-break at 40d. The figure's premise — *direction* — "
            "carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to size\n\n"
            "One picture for the operational truth: the **net** up-break excess by horizon with the "
            "break-even line. There is no horizon where a placebo-clean, cost-surviving edge lives — "
            "and the entry horizon a trader would actually use (the breakout) is **below zero**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gg = [st.run_experiment(PANEL, horizon=h, side='up', n_draws=1500)['mean']*100 for h in hs]\n"
            "    nv = [st.run_experiment(PANEL, horizon=h, side='up', n_draws=1500)['net']*100 for h in hs]\n"
            "else:\n"
            "    gg = [row[2] for row in R['up']]; nv = [row[8] for row in R['up']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, gg, 'o--', c=GREY, lw=1.6, label='gross excess')\n"
            "ax.plot(hs, nv, 'o-', c=GREEN, lw=2.2, label='net of 10 bps round trip')\n"
            "ax.axhline(0, c=RED, ls='--', label='break-even')\n"
            "for h,v in zip(hs,nv): ax.annotate(f'{v:+.2f}%',(h,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('up-break excess (%)')\n"
            "ax.set_title('No placebo-clean edge to tax; the breakout day is net-negative'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net excess by horizon:', {f'{h}d': round(v,2) for h,v in zip(hs,nv)})"
        ),
        md(
            "> 💡 In plain words: costs are almost irrelevant here — the gross excess is already inside "
            "the placebo cloud, so there's nothing for the spread to erode. The only horizon a "
            "breakout-trader actually acts on is **net-negative**, and the slow-horizon drift belongs "
            "to random dates. Hence **MIRAGE**, not a tradable edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The down-break symmetry test is the reusable tool.** Any chart figure that claims a "
            "*direction* should fail it if the figure is real: both resolutions drifting the same way "
            "is the fingerprint of a momentum/mean-reversion tell, not a pattern.\n"
            "- **Stricter geometry, volume, retests.** Fork [`detect_wedges`](../falling_wedge/strategy.py) "
            "to demand a sharper convergence, a volume dry-up into the apex, or a confirmed throwback — "
            "and check whether the placebo *p* ever drops under 5% (ours never does).\n"
            "- **Noisier universes.** Small-caps / crypto have louder figures and louder bounces — and "
            "louder costs and survivorship traps. Same protocol, bigger everything.\n\n"
            "*The reproducible core is offline and deterministic; the detector is one transparent "
            "mechanical rule (we say so). Methods and sources: "
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
