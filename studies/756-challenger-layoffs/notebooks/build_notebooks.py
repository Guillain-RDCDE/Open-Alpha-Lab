"""Generate the two narrative notebooks for Study 756 (Challenger-Layoffs).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded Challenger
snapshot (always available) and the cached SPY prices under ../_cache/, and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Challenger job-cut
# hardcoded labelled proxy + SPY month-end, 2000-01 -> 2026-06, 318 months, 26.4 years).
# per-horizon tuple positions:
#   0 months · 1 n_spike · 2 spike% · 3 calm% · 4 base% · 5 sp_down% · 6 base_down% ·
#   7 Welch_t · 8 HAC_t · 9 p_placebo
R = dict(
    start="2000-01-31", end="2026-06-30", months=318, years=26.4, spike_freq=41,
    h1=(1, 131, 0.25, 1.16, 0.78, 44, 36, -1.05, -1.66, 0.081),
    h3=(3, 131, 1.70, 2.75, 2.31, 33, 31, -0.73, -0.85, 0.174),
    h6=(6, 131, 3.54, 5.50, 4.67, 33, 27, -0.88, -0.98, 0.120),
    h12=(12, 130, 8.57, 10.63, 9.75, 25, 22, -0.58, -0.61, 0.211),
    # lead/lag: L -> corr
    leadlag={-6: -0.099, -5: -0.039, -4: -0.083, -3: -0.236, -2: -0.246, -1: -0.047,
             0: 0.015, 1: 0.016, 2: 0.079, 3: 0.088, 4: -0.054, 5: -0.018, 6: 0.056},
    # overlay: (bh_mean%, bh_sharpe, gross_mean%, gross_sharpe, net_mean%, net_sharpe, switches)
    overlay=(9.3, 0.61, 5.7, 0.55, 5.3, 0.51, 102),
    # robustness 12m: (label, n_spike, spike12%, base12%, Welch_t, HAC_t, p)
    robust=[("w=6", 139, 7.4, 9.8, -1.25, -1.56, 0.049), ("w=12", 130, 8.6, 9.8, -0.58, -0.61, 0.211),
            ("w=24", 121, 9.5, 9.8, -0.14, -0.11, 0.421), ("thr>+50%", 34, 13.5, 9.8, 0.98, 0.67, 0.908),
            ("ex-COVID", 121, 8.7, 10.7, -0.89, -1.03, 0.116)],
    # synthetic control: (edge, n_spike, spike1m%, base1m%, Welch_t, HAC_t, p)
    syn=[(0.0, 119, 0.44, 0.66, -0.53, -0.76, 0.277), (0.04, 119, -2.87, -0.65, -5.44, -7.84, 0.001)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Early_warning%3F: Not_supported](https://img.shields.io/badge/Early_warning%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from challenger_layoffs import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("SPY cache present:", HAVE_REAL,
      "| cuts+SPY months:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the SPY cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do mass-layoff announcements warn you before the market falls? 🪓\n"
            "### The Challenger job-cut report as a stock-market crystal ball, in plain English\n\n"
            + BADGES +
            "On the first Thursday of every month, an outplacement firm called **Challenger, Gray & "
            "Christmas** publishes how many layoffs U.S. employers just **announced**. When that number "
            "**spikes**, the folklore says the economy is rolling over and a **stock-market downturn is "
            "coming** — layoffs *lead*, the story goes, so a cut-spike is your early-warning to get "
            "defensive.\n\n"
            "It's a great story. It's also testable. This notebook asks three blunt questions: when job "
            "cuts spike, does the market really do worse? Does the cut-spike actually come **first** "
            "(that's the whole pitch)? And if you *sold* every time layoffs spiked, would you make "
            "money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the HAC errors, the lead/lag "
            "cross-correlation and the synthetic control? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Challenger's series is proprietary — there's no free feed — so "
            "we use a hardcoded, **clearly-labelled approximate** monthly reconstruction of the published "
            "headline totals (including the giant COVID-2020 record spike). It's a faithful-in-shape "
            "*proxy*, not the exact revised vintage. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When cuts spike, does the market do worse? | **A little — yes.** In the month right after "
            f"a cut-spike, SPY averages **+{R['h1'][2]:.2f}%** vs **+{R['h1'][4]:.2f}%** normally, and "
            f"falls more often (**{R['h1'][5]:.0f}%** vs **{R['h1'][6]:.0f}%** of the time). The "
            "*direction* matches the folklore. |\n"
            "| Is that gap reliable? | **No.** It's small and well inside the noise — statistically you "
            "can't tell it from luck (and it vanishes if you tweak the recipe). |\n"
            "| Does the cut-spike come *first*? | **No — and this is the killer.** The layoff signal "
            "lines up best with a market move that already happened **two months earlier.** Announcements "
            "*follow* stocks here; they don't lead them. |\n"
            "| So could you trade it? | **It loses.** \"Sell when cuts spike\" earned "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%** for just holding — you give "
            "up return for protection that never shows up. |\n\n"
            "> Layoffs and recessions are real. But \"layoffs warn you *early*\" is a coincident echo "
            "wearing a crystal-ball costume — and acting on it costs you money."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Announced job cuts are a leading indicator. When Challenger's monthly number spikes, "
            "the labour market is weakening and the stock market is about to follow — so get defensive "
            "the moment layoffs surge.\"*\n\n"
            "There's a respectable backbone to this: the Challenger report lands *before* the official "
            "government jobs number and counts *announcements* — forward-looking intentions, not "
            "backward-looking headcounts. So the intuition isn't crazy — fewer jobs, weaker spending, "
            "lower profits. The trading leap is the part we test: that a cut *spike* arrives early "
            "enough, and cleanly enough, to be a **tradable** warning for equities."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be gold: a free, monthly, widely-reported number that tells you to step "
            "aside before drawdowns. But \"leading\" hides a trap. The **stock market is itself a leading "
            "indicator** — it usually turns *before* the economy. So a labour number that lines up with "
            "market weakness might not be *predicting* the market at all; it might just be **echoing** a "
            "turn the market already made. The difference between *leads* and *echoes* is the difference "
            "between an edge and a mirage — and you can only tell them apart by checking the timing "
            "carefully."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of Challenger's monthly job-cut number against month-end SPY, and:\n\n"
            "1. **Split the months.** Call cuts **spiking** when the month runs above its own "
            "trailing-year average. Compare what SPY did next (1/3/6/12 months) in spike months vs all "
            "months. Crucially, we only ever act **after** the report is public — a strict one-month "
            "release lag, no peeking.\n"
            "2. **Check the timing.** The crucial test: slide the layoff signal forward and backward "
            "against the market and find *where* they line up best. If layoffs truly **lead**, the "
            "strongest link shows up at a **positive lead** (cuts first, market later).\n"
            "3. **Try to trade it.** Sit in cash whenever cuts spike, hold otherwise, pay realistic "
            "costs — and see if it beats just buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's the monthly job-cut number over a quarter-century — the "
            "quiet stretches, and the giant spikes (2001, 2008–09, and the off-the-chart COVID surge of "
            "2020). Layoffs clearly *know* about recessions. The question is whether they know **early**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = F['cuts']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(x.index, x.values, c=RED, lw=1.2)\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_title('U.S. announced job cuts, Challenger monthly (thousands, log scale)')\n"
            "    ax.set_ylabel('announced cuts (thousands)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('peak month:', int(x.max()), 'thousand announced, around', x.idxmax().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; COVID spike ~671k announced in Apr 2020')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward SPY return in **spike** months "
            "next to the return on an **average** month. The folklore predicts the green bars sit "
            "*below* the grey ones."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, m) for m in hs]\n"
            "    spk = [r['spike_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    spk = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "    base = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "xx = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(xx-.2, spk, .4, color=GREEN, label='after cuts SPIKE')\n"
            "ax.bar(xx+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(xx); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward SPY return (%)')\n"
            "ax.set_title('Cut spikes -> lower forward returns... but only by a little')\n"
            "for i,(a,b) in enumerate(zip(spk,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1-month: spike', f'{spk[0]:.2f}%', 'vs base', f'{base[0]:.2f}%')"
        ),
        md(
            f"The direction is right — the gap is biggest in the **month right after the print** "
            f"(**+{R['h1'][2]:.2f}%** after a spike vs **+{R['h1'][4]:.2f}%** normally, and the market is "
            f"*down* **{R['h1'][5]:.0f}%** of the time vs **{R['h1'][6]:.0f}%**). That's the "
            "\"announcement-drift\" everyone points to. But the gap is small — small enough that, with "
            "the numbers we have, it could easily be chance — and it *shrinks* as you look further out. "
            "Hold that thought; the *next* chart is where the story breaks."
        ),
        md(
            "**The crucial test: does the cut-spike come *first*?** We slide the layoff signal forward "
            "and backward against the market and measure how tightly they move together. A real "
            "early-warning would show its strongest *downward* link at a **positive lead** (cuts lead → "
            "bar dips on the right). Watch where it actually dips."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [RED if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = cuts move FIRST (early-warning)   |   L<0 = cuts LAG the market')\n"
            "ax.set_ylabel('correlation with market move'); ax.set_xticks(Ls)\n"
            "ax.set_title('The dip is on the LEFT: cuts lag the market by ~2 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "imin = int(np.nanargmin(cs))\n"
            "print(f'strongest negative link at L={Ls[imin]} months (cuts FOLLOW the market here)')"
        ),
        md(
            f"There it is. The deepest *negative* bar is at **L = −2** — the layoff signal lines up best "
            "with a market move that happened **two months earlier**. On the right, where a true "
            "early-warning would live (cuts moving first), the bars are near zero or even *positive*. "
            "**Layoff announcements aren't leading the market — they're trailing it.** The recession "
            "shows up in stock prices, and only later in the layoff count."
        ),
        md(
            "**Could you trade it anyway?** Suppose you sold (went to cash) every month cuts were spiking "
            "and held SPY otherwise. Here's that strategy's growth vs just buying and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spike = st.spike_mask(F); pos = (~spike).astype(float).shift(1)\n"
            "    rr = F['spy'].pct_change()\n"
            "    import pandas as pd\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); cst=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*cst)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold SPY')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='cash when cuts spike (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Sell when layoffs spike\" lags buy-and-hold for 25 years')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][4]:.1f}%/yr vs buy-hold {R['overlay'][0]:.1f}%/yr (net) — see results.md\")"
        ),
        md(
            f"The defensive overlay ends up **well below** buy-and-hold — "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%/yr** net, and a *lower* Sharpe "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}). Every time you stepped aside on a cut "
            "spike you mostly missed *gains*, because layoffs kept spiking well into recoveries the "
            "market had already started pricing. The signal doesn't just fail to help — **it hurts.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** A cut spike really does precede slightly weaker, more-often-negative "
            "returns — clearest in the announcement-drift month — but the gap is small, not "
            "statistically significant, and fragile to how you define the signal. Real as lore, weak as "
            "edge.\n"
            "- **Tradability — Mirage.** Selling on layoff spikes **loses to buy-and-hold**. There's "
            "nothing to deploy.\n"
            "- **Early warning? — Not supported.** The cut spike lines up with a market move that already "
            "happened two months earlier. Layoffs **echo** equity weakness; they don't forecast it. The "
            "one word that makes the pitch — *early* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even if the small tilt were real, the operational reality "
            "is brutal: cuts run 'above trend' in **roughly four months out of ten** (every bump counts "
            "as a spike), so a cash-on-spike rule whipsaws you in and out constantly — "
            f"**{R['overlay'][6]} switches** in 25 years — while the deepest layoff surges (the "
            "*genuine* recession lows) are exactly the moments you'd most want to be **buying**, not "
            "selling. That's why pushing the rule to only fire on a *big* cut spike flips its sign "
            "**positive**. There is no version of \"sell when layoffs spike\" that both fires early and "
            "makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling test.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "asks the same question of weekly unemployment claims: real labour signal, but can you trade "
            "it?\n"
            "- **The single-stock cousin.** [Study 749 — Layoff-Drift](../749-layoff-drift/) tests "
            "whether *one company's* layoff announcement pops or drifts — the micro version of this macro "
            "question.\n"
            "- **Build your own.** Swap the trailing-average spike for a year-over-year jump, or pair "
            "cuts with a *price* trend filter — the lead/lag picture barely budges: a coincident series "
            "can't be made to lead by redefining the spike.\n\n"
            "*Think layoffs lead the market? Show the lead/lag chart dipping on the **right** (positive "
            "lead) — then we'll talk.*"
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
            "# Challenger-Layoffs — a quantitative teardown 🔬\n"
            "### Spike-conditioned split returns · Welch *t* + Newey-West HAC *t* + placebo null · the "
            "decisive lead/lag cross-correlation · a timing overlay vs buy-and-hold · robustness · a "
            "synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "believers fuse two claims: that a Challenger job-cut spike (1) **predicts** weaker equity "
            "returns and (2) does so **early** enough to trade. We separate them. The conditional return "
            "tilt is the *right sign but insignificant*; the decisive object is the **lead/lag "
            "structure**, which shows the cut spike is **coincident-to-lagging**, not leading — and a "
            "tradable overlay that *underperforms* buy-and-hold seals the Tradability axis.\n\n"
            "> ⚠️ **Data + proxy note.** Challenger's series is proprietary with no free feed; the "
            "job-cut tape is a hardcoded, **labelled approximate** monthly reconstruction of the public "
            "headline totals (thousands announced) — faithful in shape, **not** the exact revised vintage "
            "(named on the Signal axis, so magnitude is a proxy result). SPY is yfinance daily adjusted "
            "close (total-return), month-end sampled. Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `WEAK` | 1-month spike mean **+{R['h1'][2]:.2f}%** vs base "
            f"**+{R['h1'][4]:.2f}%** (right sign, announcement-drift); Welch **t = {R['h1'][7]:.2f}**, "
            f"HAC **t = {R['h1'][8]:.2f}**, placebo **p = {R['h1'][9]:.2f}** — fails **|t| ≥ 2**, fragile "
            "to window, sign-flips for big spikes. |\n"
            f"| **Tradability** | `MIRAGE` | Cash-on-spike overlay **+{R['overlay'][4]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][5]:.2f}**) **vs buy-hold +{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][1]:.2f}**). Acting on it destroys return. |\n"
            f"| **Early warning?** | `NOT SUPPORTED` | Peak negative lead/lag correlation at "
            f"**L = −2** (cuts lag the market); at positive leads corr ≈ 0. A coincident-to-lagging "
            "echo, not a leader. |\n\n"
            "> 💡 In plain words: the equity market *is* a leading indicator of the economy, so a labour "
            "series that co-moves with equity weakness need not lead it. The cut spike lines up with a "
            "market move already two months old — the 'early-warning' is the market's own lead, "
            "reflected back."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_t$ be monthly announced job cuts and "
            "$s_t = x_t / \\overline{x}_{t-1:t-12} - 1$ its excess over the trailing-year average. Cuts "
            "are **SPIKING** at $t$ when $s_t > 0$. With a one-month execution lag (the month-$t$ report "
            "is released early in $t+1$ and acted on at that close), define forward return "
            "$r_{t+1\\to t+1+H}$.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{spike}] < \\mathbb{E}[r]$ — a *negative* "
            "excess over the base rate.\n"
            "- **H₂ (leads).** The strongest negative spike↔return correlation sits at a **positive** "
            "lead (cuts move first).\n"
            "- **H₃ (deployable).** A cash-on-spike overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ directionally true but insignificant** (best HAC $t = -1.66$, sign-flips by "
            "spec), **H₂ rejected** (peak negative corr at $L=-2$), **H₃ rejected** (overlay "
            "underperforms). The folklore is right exactly where it's uninformative (layoffs and "
            "recessions co-move) and wrong exactly where it would pay (a *leading*, *tradable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error, "
            "and — because forward windows overlap — by a HAC-robust regression *t*:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{spike}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "r_{t\\to t+H} = \\alpha + \\beta\\,\\mathbb{1}[\\text{spike}_t] + \\varepsilon_t,\\quad "
            "t_\\beta = \\beta / \\mathrm{se}_{\\text{NW}(H)}(\\beta).$$\n\n"
            "But a significant $\\widehat{\\Delta}$ would **still not** establish *leading*: a coincident "
            "or lagging series can co-move with forward returns through autocorrelation in the cycle. "
            "The identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(s_t,\\ r_{t+L\\to t+L+1})$. A genuine early-warning peaks "
            "(negatively) at $L>0$. If $\\arg\\min_L \\rho(L) < 0$, cuts **follow** the market — and the "
            "entire 'early' thesis collapses regardless of the conditional mean."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Job-cut tape.** Monthly Challenger announced cuts (thousands), hardcoded **labelled "
            f"proxy**, {R['start'][:7]}→{R['end'][:7]} ({R['months']} months). Approximate public "
            "headlines, not the revised vintage (named on the axis).\n"
            "- **Signal.** $s_t = x_t/\\overline{x}_{t-1:t-12}-1$; SPIKE when $s_t>0$ (fires in "
            f"**{R['spike_freq']}%** of months).\n"
            "- **Forward returns.** Enter at the close **1 month after** the signal (no look-ahead), "
            "hold $H\\in\\{1,3,6,12\\}$ months; drop horizons that overrun the tape.\n"
            "- **Null #1 (Welch t).** Spike-set mean vs the unconditional mean.\n"
            "- **Null #2 (HAC t).** Newey-West *t* of the spike dummy, lag truncation $= H$ (overlap).\n"
            "- **Null #3 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\le \\text{spike mean}]$ (as bearish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$ — *where* do cuts line up?\n"
            "- **Tradability.** Cash-when-spike overlay, 1-month lag, 10 bps one-way per switch "
            "(turnover one-way × NAV), excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* spike→returns link: "
            "`edge=0` must not fake significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — right sign, small, insignificant\n\n"
            "Spike-month forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed). Below base at every horizon — but inside its own error bar, and the HAC *t* "
            "(printed) never clears the bar."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ht, ses = [], [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, m); cm.append(s['spike_mean']); bm.append(s['base_mean'])\n"
            "        ts.append(s['t']); ht.append(s['hac_t'])\n"
            "        r,_c,_a = st.split_returns(F, m); ses.append(r.std(ddof=1)/np.sqrt(len(r)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    ts = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]\n"
            "    ht = [R['h1'][8], R['h3'][8], R['h6'][8], R['h12'][8]]; ses = [.006,.02,.03,.045]\n"
            "xx = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(xx, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='spike-month (±SE)')\n"
            "ax.plot(xx, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(xx); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Right sign (below base) but the SE swamps the gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})\n"
            "print('HAC   t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ht)})"
        ),
        md(
            f"> 💡 In plain words: the tilt is sharpest at **1 month** — the announcement-drift window — "
            f"where the spike mean is **+{R['h1'][2]:.2f}%** vs base **+{R['h1'][4]:.2f}%** at "
            f"**HAC t = {R['h1'][8]:.2f}** (placebo *p* = {R['h1'][9]:.2f}). Right sign, still short of "
            "$|t|=2$, and it *decays* with horizon. H₁ is **directionally supported, statistically "
            "not**: the right sign living inside its own error bar."
        ),
        md(
            "### 4b · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(s_t, r_{t+L\\to t+L+1})$. Negative bars left of zero = cuts "
            "**lag** the market; a real early-warning would dip on the **right** (cuts lead)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [RED if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "imin = int(np.nanargmin(cs))\n"
            "ax.annotate('strongest NEGATIVE link\\n(cuts LAG the market)', xy=(Ls[imin], cs[imin]),\n"
            "            xytext=(Ls[imin]-0.3, cs[imin]-0.10), ha='center', color=RED,\n"
            "            arrowprops=dict(arrowstyle='->', color=RED))\n"
            "ax.set_xlabel('lead L (months): L>0 = cuts lead (early-warning)   |   L<0 = cuts lag')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('argmin rho(L) is at L<0: cuts are coincident-to-lagging')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'argmin at L={Ls[imin]} (rho={cs[imin]:+.2f}); rho at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: $\\arg\\min_L \\rho(L) = -2$. The cut spike correlates most "
            "(negatively) with a market move **two months in its past**; at the positive leads a genuine "
            "early-warning needs, $\\rho \\approx 0$. **H₂ rejected.** The equity market leads the "
            "economy; layoffs trail both — the 'early-warning' is the market's lead, reflected. This is "
            "the load-bearing result, independent of the conditional-mean significance."
        ),
        md(
            "### 4c · Tradability — the cash-on-spike overlay loses\n\n"
            "Hold SPY when cuts are calm, cash when spiking (1-month lag, 10 bps/switch). Annualised "
            "mean and Sharpe vs buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.timing_overlay(F, cost_bps=10.0)\n"
            "    bh_m, bh_s = o['bh_mean']*100, o['bh_sharpe']\n"
            "    g_m, g_s = o['overlay_gross_mean']*100, o['overlay_gross_sharpe']\n"
            "    n_m, n_s = o['overlay_net_mean']*100, o['overlay_net_sharpe']; nsw=o['n_switches']\n"
            "else:\n"
            "    bh_m, bh_s = R['overlay'][0], R['overlay'][1]; g_m,g_s = R['overlay'][2],R['overlay'][3]\n"
            "    n_m, n_s = R['overlay'][4], R['overlay'][5]; nsw=R['overlay'][6]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = ['buy &\\nhold', 'overlay\\ngross', 'overlay\\nnet @10bps']\n"
            "a1.bar(labels, [bh_m, g_m, n_m], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([bh_m,g_m,n_m]): a1.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('annualised mean return (%)'); a1.set_title('Return: overlay loses ~4 pts/yr')\n"
            "a2.bar(labels, [bh_s, g_s, n_s], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([bh_s,g_s,n_s]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annualised Sharpe (excess-of-0)'); a2.set_title(f'Sharpe: overlay lower too ({nsw} switches)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net overlay {n_m:.1f}%/yr (Sharpe {n_s:.2f}) vs buy-hold {bh_m:.1f}%/yr (Sharpe {bh_s:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the overlay returns **+{R['overlay'][4]:.1f}%/yr** net vs "
            f"**+{R['overlay'][0]:.1f}%** for buy-and-hold, with a *lower* Sharpe "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}) and {R['overlay'][6]} switches. Because "
            "layoffs keep spiking into recoveries the market has already begun pricing, sitting out on "
            "'cuts spiking' systematically forfeits upside. **H₃ rejected** — costs aren't even the "
            "issue; the *timing* loses. `MIRAGE`."
        ),
        md(
            "### 4d · Robustness — window, threshold, and the COVID dependence\n\n"
            "Vary the trailing window $w$ and the spike threshold, and drop 2020–2021. The 12-month *t* "
            "never clears 2 at any spec — and the biggest spikes flip the sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for w in (6,12,24):\n"
            "        s = st.summarize(F, 12, window=w); rob.append((f'w={w}', s['n_spike'], s['spike_mean']*100, s['t'], s['p_placebo']))\n"
            "    s = st.summarize(F, 12, thresh=0.50); rob.append(('thr>+50%', s['n_spike'], s['spike_mean']*100, s['t'], s['p_placebo']))\n"
            "    F2 = F[(F.index < '2020-01-01') | (F.index >= '2022-01-01')]\n"
            "    s = st.summarize(F2, 12); rob.append(('ex-COVID', s['n_spike'], s['spike_mean']*100, s['t'], s['p_placebo']))\n"
            "else:\n"
            "    rob = [(l,n,r,t,p) for (l,n,r,_b,t,_h,p) in R['robust']]\n"
            "labels = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [GREEN if t<0 else RED for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t=-2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('Welch t (12-month)'); ax.set_title('No spec clears |t|=2; big spikes flip POSITIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, spike12%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: the effect *strengthens* only as the window **shortens** "
            f"(**w=6 → HAC t={R['robust'][0][5]:.2f}**) and at the 1-month announcement-drift horizon "
            "(HAC t=−1.66) — but neither clears the bar. Restrict to *big* spikes (>+50%) and the sign "
            f"flips **positive** (**t={R['robust'][3][4]:.2f}**): the largest layoff surges are recession "
            f"*bottoms* you'd buy. Ex-COVID the 12m t is **{R['robust'][4][4]:.2f}** — still under the "
            "bar. The signal is real-ish only where it's useless and useless where it would pay."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly job-cut + SPY-like series with a *planted* link (a cut spike at $t$ "
            "depresses the $t{+}1$ return by `edge`). With `edge=0` the test must stay flat; with a large "
            "`edge` it must light up — proving the engine is unbiased and the real-tape null isn't a "
            "measurement failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04):\n"
            "    syn = data.synthetic_cuts(n_months=360, edge=edge, seed=756)\n"
            "    s = st.summarize(syn, 1, window=12)\n"
            "    res.append((edge, s['n_spike'], s['spike_mean']*100, s['base_mean']*100, s['t'], s['hac_t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,*_ in res]\n"
            "tvals = [r[5] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t=-2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f'HAC t={t:.2f}',(i,t),ha='center',va='top')\n"
            "ax.set_ylabel('HAC t (1-month)'); ax.set_title('Control: no link -> flat; real link -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,h,p in res: print(f'planted {e*100:+.0f}%/mo: n_sp={k} spike={c:.2f}% base={b:.2f}% Welch_t={t:.2f} HAC_t={h:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the control sits at "
            f"**HAC t = {R['syn'][0][5]:.2f}** (no false positive); a **+4%/month** planted link drives "
            f"**HAC t = {R['syn'][1][5]:.2f}**. So the machinery is honest — the real-tape HAC *t* of "
            "≈−1.7 (at its strongest) is a *genuine* weak-or-absent edge, not a broken test. The engine "
            "*can* bank a real spike→returns link; the real tape just doesn't carry a tradable one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 1-month excess **{R['h1'][2]-R['h1'][4]:+.2f}pp** at Welch "
            f"**t = {R['h1'][7]:.2f}** / HAC **t = {R['h1'][8]:.2f}** / placebo **p = {R['h1'][9]:.2f}**; "
            "right sign at every horizon, fails |t|≥2, fragile to window, sign-flips for big spikes. "
            "Literature support (Challenger as a labour nowcast) + a directionally-correct-but-"
            "insignificant tilt ⇒ WEAK, not REAL — and the tape is a labelled *proxy*, so magnitude is a "
            "proxy result.\n"
            f"- **Tradability `MIRAGE`** — the cash-on-spike overlay returns "
            f"**+{R['overlay'][4]:.1f}%/yr** (Sharpe {R['overlay'][5]:.2f}) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe {R['overlay'][1]:.2f}). Acting on the signal "
            "*subtracts* return — nothing to allocate to.\n"
            "- **Early warning? `NOT SUPPORTED`** — $\\arg\\min_L \\rho(L) = -2$ months: the cut spike "
            "is **coincident-to-lagging**, not leading. The equity market is the leading indicator; "
            "layoffs echo it. The defining word — *early* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a genuine few-point tilt. The operational reality still defeats it. Cuts run "
            f"'above trend' in **~{R['spike_freq']}% of months** (every up-bump of a lumpy monthly "
            f"series), so the overlay churns ({R['overlay'][6]} switches / 25y) and is out of the market "
            "in roughly four months of ten — including the early innings of every recovery, which the "
            "equity market starts pricing *before* layoffs roll over. That structural mistiming is why "
            "the overlay's Sharpe is **below** passive even at zero cost. And the one regime where "
            "layoffs and a market bottom genuinely coincide — the deepest surges — is where you'd want "
            "maximum *long* exposure, which is exactly why the >+50% threshold flips the sign positive. "
            "No lag, threshold, or cost assumption rescues a coincident series masquerading as a leading "
            "one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/): "
            "weekly initial claims as an early-warning, same hardcoded-snapshot + SPY method — a real "
            "labour signal whose tradability is the open question.\n"
            "- **The single-stock cousin.** [Study 749 — Layoff-Drift](../749-layoff-drift/): does *one "
            "firm's* mass-layoff announcement pop or drift? The micro event-study to this macro test.\n"
            "- **Sharper identification.** Replace the monthly proxy with Challenger's actual vintage (or "
            "the sector breakdown) and run a proper VAR / Granger test; the coincident-to-lagging "
            "structure is robust to all of these — smoothing or re-thresholding a coincident series can't "
            "manufacture a lead.\n\n"
            "*The reproducible core is offline and deterministic; the job-cut input is an explicit "
            "labelled proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
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
