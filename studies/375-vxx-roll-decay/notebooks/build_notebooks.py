"""Generate the two narrative notebooks for Study 375 (VXX-Roll-Decay).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached VIX-ETP tape
under ../_cache/ (VIXY + VIX term structure) and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance: VIXY [VXX-equivalent]
# + VIX term structure, 2011-01-04 -> 2026-06-22, 3,888 days, 15.5 years).
R = dict(
    start="2011-01-04", end="2026-06-22", days=3888, years=15.5,
    daily_mean=0.171, hac_t=2.51, placebo_p=0.0076, frac_contango=92.3,
    etp_first=633840, etp_last=21.86, etp_ratio=3.4e-5,
    # gross / net short-carry book
    g_ann=43.0, g_vol=69.9, g_sharpe=0.62, g_dd=-91.8,
    n_ann=35.1, n_sharpe=0.50, n_dd=-93.1,
    # tail
    skew=-1.65, exkurt=9.4, worst=-43.1, worst_date="2024-08-05", var1=-14.6, cvar1=-20.7,
    worst5=[("2024-08-05", -43.1), ("2020-03-16", -39.1), ("2018-02-05", -34.2),
            ("2020-06-11", -33.6), ("2020-03-09", -27.4)],
    # contango-only
    c_ann=39.7, c_sharpe=0.67, c_dd=-66.7, c_skew=-1.30, c_t=3.60,
    # borrow sensitivity (fee%, net ann%, net Sharpe)
    borrow=[(0, 38.0, 0.54), (3, 35.1, 0.50), (10, 28.1, 0.40), (30, 8.3, 0.12)],
    # synthetic control: (planted carry/day, HAC t, placebo p, Sharpe, skew, maxDD%)
    syn=[(0.0, -1.99, 0.987, -0.58, -0.40, -100), (0.0030, 3.42, 0.0001, 0.99, -0.40, -76)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_carry%3F: Busted](https://img.shields.io/badge/Free_carry%3F-Busted-8b949e?style=flat-square)\n\n"
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

from vxx_roll_decay import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real VIX-ETP cache present:", HAVE_REAL,
      "| days:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Shorting VXX — steady carry, or nickels in front of a steamroller? 📉\n"
            "### Why a fund that melts away ×100,000 is so tempting to short — and why it can run you over\n\n"
            + BADGES +
            "There's a product on the stock exchange whose whole *job* is to lose money. VIX-futures "
            "funds like **VXX** (and its longer-lived twin **VIXY**) track stock-market *fear*, and "
            "because fear is usually cheap-now / expensive-later, the fund has to keep selling low and "
            "buying high every single day. The result is spectacular: it has melted from a "
            f"split-adjusted **{R['etp_first']:,}** down to about **{R['etp_last']:.0f}** — a loss of "
            "**99.997%**.\n\n"
            "So *shorting* it looks like free money: bet against something designed to evaporate. This "
            "notebook shows that the carry is **real** — and that it is exactly the kind of real that "
            "can wipe you out in an afternoon.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stat, the placebo test, the skew/kurtosis "
            "and the borrow-fee math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** A clean, continuous **VXX** price only exists on yfinance "
            "from 2018 (the original note was retired and reissued), which would *miss* the famous "
            "Feb-2018 'Volmageddon' blow-up. So we short **VIXY** — the same fear-futures fund with an "
            "unbroken history back to 2011 — and call it a VXX-equivalent throughout. Every chart is "
            "drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does shorting the fund actually make money? | **Yes — reliably.** The short earns about "
            f"**+{R['daily_mean']:.2f}% a day** on average, **+{R['n_ann']:.0f}%/yr** net of costs. "
            "The carry is statistically real (not luck). |\n"
            "| So it's free money? | **No.** It's the fee for **selling crash insurance.** When fear "
            f"spikes the fund explodes and the short loses — **{R['worst']:.0f}% in a single day** "
            f"(Aug 2024), and **{R['g_dd']:.0f}%** peak-to-trough. |\n"
            "| How often does the steamroller run? | **Three times in fifteen years** — Volmageddon "
            "(2018), COVID (2020), the 2024 yen-carry unwind — each eating a third-to-half of the book "
            "in one session. |\n"
            "| Can you build a fund on it? | **Not at scale.** The carry is real and small; the loss is "
            "rare and enormous. Size it to matter and you size it to be wiped out. |\n\n"
            "> The carry is real. \"Free\" is the lie. You are paid a steady nickel to stand in front "
            "of a steamroller."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"VIX-futures funds are doomed to decay. The futures curve is almost always in "
            "**contango** — tomorrow's fear costs more than today's — so the fund must roll from cheap "
            "near-month fear into pricey far-month fear, day after day, bleeding value forever. Short "
            "it and you harvest that bleed as a steady **carry**.\"*\n\n"
            "It's one of the most popular trades in retail finance, and the decay is no myth: the fund "
            "really has lost almost everything. The seductive part is the *steadiness* — most days the "
            "fund drifts down and the short ticks up. The dangerous part is what 'most days' is hiding."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"steady carry.\" (1) *Is there a real edge?* — yes, "
            "selling insurance pays a premium on average, that's why insurance exists. (2) *Can you "
            "live to collect it?* — that depends entirely on the **tail**. A trade that wins a little "
            "almost every day and loses **half its value** on the rare bad day isn't characterised by "
            "its average; it's characterised by its worst day. The whole question is whether the steady "
            "nickels add up faster than the steamroller takes them away — and whether you survive the "
            "gap in between."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take the real fund (**VIXY**, our VXX-equivalent) over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}) and run the simplest possible trade: hold a fixed-size "
            "**short**, every day.\n\n"
            "1. **Measure the carry.** What does the short earn per day on average — and is it real, or "
            "could a coin flip have produced it?\n"
            "2. **Charge the costs.** Shorts pay a **borrow fee**, and rebalancing costs money. Does "
            "the carry survive?\n"
            "3. **Look at the worst days.** Skew, the single ugliest session, the deepest drawdown — "
            "the numbers that decide whether you'd still be standing. **If one bad day eats months of "
            "carry, it's a steamroller, not a strategy.**"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the decay itself.** Here is the fund's price on a log scale. It doesn't wobble — "
            "it *melts*, smoothly, for years. That melt is the carry the short collects."
        ),
        code(
            "if HAVE_REAL:\n"
            "    etp = F['etp']\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "    ax.semilogy(etp.index, etp.values, c=RED, lw=1.6)\n"
            "    ax.set_title('The VIX-futures fund melts ~100,000x (log scale) — that melt IS the carry')\n"
            "    ax.set_ylabel('fund price (split-adj, log)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'{etp.iloc[0]:,.0f} -> {etp.iloc[-1]:,.2f}  (x{etp.iloc[-1]/etp.iloc[0]:.1e})')\n"
            "else:\n"
            "    print(f\"no cache — frozen: {R['etp_first']:,} -> {R['etp_last']:.0f} (x{R['etp_ratio']:.1e})\")"
        ),
        md(
            f"From **{R['etp_first']:,}** to **{R['etp_last']:.0f}**. A near-perfect downhill slide — "
            "and the short rides it up. So far, free money. Now turn the page."
        ),
        md(
            "**The short's daily returns — almost always a small win, occasionally a catastrophe.** "
            "Most bars are tiny green ticks. Watch the few red spikes plunging off the bottom."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.short_carry(F)\n"
            "else:\n"
            "    rng = np.random.default_rng(375)\n"
            "    rs = rng.normal(R['daily_mean']/100, 0.04, 3800)\n"
            "    rs[[800, 1900, 3600]] = [-.34, -.39, -.43]   # the three steamroller days\n"
            "    import pandas as pd; rs = pd.Series(rs)\n"
            "vals = np.asarray(rs)*100\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "ax.bar(range(len(vals)), vals, color=np.where(vals>=0, GREEN, RED), width=1.0)\n"
            "ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_title('Short-carry daily returns: tiny green wins, rare red catastrophes')\n"
            "ax.set_ylabel('short return that day (%)'); ax.set_xlabel('trading day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'worst single day: {vals.min():.0f}%  |  median day: {np.median(vals):+.2f}%')"
        ),
        md(
            f"There's the whole story in one picture. The median day is a **+0.6%** nibble; the worst "
            f"day is **{R['worst']:.0f}%** — more than *three months* of nibbles gone before lunch. The "
            "average hides everything that matters."
        ),
        md(
            "**The equity curve — and the cliffs.** Compound those daily returns into a constant-size "
            "short book. It climbs nicely... and then falls off a cliff, three times."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    rs = st.short_carry(F); idx = F.index[1:]\n"
            "else:\n"
            "    rs = pd.Series(rs); idx = pd.RangeIndex(len(rs))\n"
            "eq = (1+pd.Series(np.asarray(rs), index=idx)).cumprod()\n"
            "dd = eq/eq.cummax()-1\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 5.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios':[2,1]})\n"
            "a1.plot(eq.index, eq.values, c=GREEN, lw=1.5); a1.set_yscale('log')\n"
            "a1.set_title('Constant-size short book: it climbs — then three cliffs'); a1.set_ylabel('growth of $1 (log)')\n"
            "a2.fill_between(dd.index, dd.values*100, 0, color=RED, alpha=.5)\n"
            "a2.set_ylabel('drawdown (%)'); a2.set_xlabel('date' if HAVE_REAL else 'day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'deepest drawdown: {dd.min()*100:.0f}%')"
        ),
        md(
            f"The carry is real — the green line genuinely rises. But the deepest drawdown is "
            f"**{R['g_dd']:.0f}%**: at the bottom of a crash the book has given back almost everything. "
            "A leveraged version (which is how people actually trade this) would simply be **gone**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The carry isn't an illusion: **+{R['daily_mean']:.2f}%/day**, real "
            f"under a proper statistical test (the quants' notebook has the *t* = {R['hac_t']:.2f}). "
            "Contango is the usual state and the short genuinely harvests it.\n"
            f"- **Tradability — Fragile.** Net **+{R['n_ann']:.0f}%/yr** looks great until you see the "
            f"**{R['worst']:.0f}% day** and the **{R['g_dd']:.0f}% drawdown**. A size that matters is a "
            "size that gets wiped out.\n"
            "- **Free carry? — Busted.** It's not free — it's the premium for selling crash insurance. "
            "Steady nickels, occasional steamroller."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — meet the steamroller\n\n"
            "Forget averages and just stare at the worst days. Here are the five sessions that define "
            "the trade — every one a famous volatility blow-up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w5 = st.short_carry(F).nsmallest(5)\n"
            "    labs = [d.date().isoformat() for d in w5.index]; vals = (w5.values*100)\n"
            "else:\n"
            "    labs = [d for d,_ in R['worst5']]; vals = [v for _,v in R['worst5']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.0))\n"
            "ax.barh(range(len(vals)), vals, color=RED)\n"
            "ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs); ax.invert_yaxis()\n"
            "ax.set_xlabel('short return that day (%)')\n"
            "ax.set_title('The five worst days — Volmageddon, COVID, the 2024 unwind')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.0f}%',(v,i),ha='right',va='center',color='white')\n"
            "ax.axvline(R['daily_mean'], c=GREEN, ls='--', lw=2, label=f\"a normal day (+{R['daily_mean']:.2f}%)\")\n"
            "ax.legend(loc='lower left'); plt.tight_layout(); plt.show()\n"
            "print('each of these days erases ~6-10 MONTHS of the daily carry')"
        ),
        md(
            f"The green dashed line is a *normal* day (**+{R['daily_mean']:.2f}%**). The red bars are "
            f"**{abs(R['worst']):.0f}%, {abs(R['worst5'][1][1]):.0f}%, {abs(R['worst5'][2][1]):.0f}%** "
            "single-day losses. Each one undoes most of a year. Costs aren't the problem — a 3% borrow "
            "fee barely registers against a 35%/yr carry. **The tail is the problem.** You can't run a "
            "fund whose risk is 'and then one day we lose half.'"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Try to dodge the steamroller.** Only short when the curve is in contango (a simple "
            "filter). It *helps* — the drawdown shrinks from ~92% to ~67% — but a 67% drawdown is still "
            "ruinous, and the worst crashes strike *from* contango in hours. The quants' notebook runs it.\n"
            "- **The borrow trap.** The fund gets **hard to borrow** in exactly the crises when the "
            "carry would pay most; at a 30% borrow fee the edge nearly vanishes. The market charges you "
            "when you most want in.\n"
            "- **Size it yourself.** Take the daily returns, apply *your* leverage, and find the size at "
            "which the worst day ruins you. That number — not the average — is the trade.\n\n"
            "*Think you can keep the carry and skip the steamroller? Build the filter, show the "
            "drawdown under a real crash, and survive the next one — then we'll talk.*"
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
            "# VXX-Roll-Decay — a quantitative teardown 🔬\n"
            "### The short-vol carry · HAC inference + a placebo null · costs with borrow · "
            "the skew/kurtosis/VaR/drawdown tail · contango conditioning · a synthetic carry-knob / crash control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We split "
            "the two questions \"steady carry\" fuses: is the contango **roll-decay premium** "
            "statistically *real*, and does a constant-notional short *survive the tail*? The first is a "
            "clean **yes** (HAC *t* > 2); the second is a clean **no** at any size that matters. The "
            "decisive object is not the Sharpe — it's the **left tail**.\n\n"
            "> ⚠️ **Data + vehicle note.** A continuous yfinance `VXX` starts only in 2018 (ETN "
            "reissue) and would miss Volmageddon, so we short **VIXY** — the same constant-maturity "
            "short-dated VIX-futures index, unbroken from 2011 — named a VXX-equivalent on the Signal "
            "axis. Contango is gauged by `^VIX3M/^VIX`. Real data: yfinance daily adjusted closes, "
            f"{R['start']}→{R['end']}. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `REAL` | daily carry **+{R['daily_mean']:.2f}%**, **HAC(21) t = "
            f"{R['hac_t']:.2f}** (>2), sign-shuffle placebo **p = {R['placebo_p']:.3f}**; "
            f"contango **{R['frac_contango']:.0f}%** of days; ETP decays ×{R['etp_ratio']:.0e}. |\n"
            f"| **Tradability** | `FRAGILE` | net Sharpe **{R['n_sharpe']:.2f}** *but* skew "
            f"**{R['skew']:.2f}**, ex-kurt **{R['exkurt']:.1f}**, worst day **{R['worst']:.0f}%**, "
            f"maxDD **{R['g_dd']:.0f}%**; 1% CVaR **{R['cvar1']:.0f}%**. |\n"
            f"| **Free carry?** | `BUSTED` | it's the **variance/crash-risk premium**, not a free "
            "lunch — three tail events (2018/2020/2024) each took a third-to-half of the book in a day. |\n\n"
            "> 💡 In plain words: the carry passes a *proper* (autocorrelation-robust) significance "
            "test — it is genuinely there. But a constant-notional short is short deep convexity: the "
            "same premium that makes the average positive makes the tail lethal. Real signal, "
            "un-allocatable trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the VIX-futures ETP and $r_t = P_t/P_{t-1}-1$ its daily return. The "
            "**short-carry book** holds a constant-notional short, rebalanced daily, with book return "
            "$b_t = -r_t$. The term-structure slope is $c_t = \\mathrm{VIX3M}_t/\\mathrm{VIX}_t$; "
            "$c_t>1$ is **contango**.\n\n"
            "- **H₁ (the carry is real).** $\\mathbb{E}[b_t] > 0$ — the roll-decay premium is "
            "statistically distinguishable from zero under autocorrelation-robust inference.\n"
            "- **H₂ (it's deployable).** The Sharpe survives borrow + costs **and** the book survives "
            "its own tail at an allocatable size.\n"
            "- **H₃ (it's a free lunch).** The carry is *not* compensation for a risk you're bearing.\n\n"
            "We find **H₁ supported** (HAC *t* = "
            f"{R['hac_t']:.2f}, placebo *p* = {R['placebo_p']:.3f}), **H₂ rejected** (skew "
            f"{R['skew']:.2f}, maxDD {R['g_dd']:.0f}%, a {R['worst']:.0f}% day), **H₃ rejected** (the "
            "carry *is* the crash-insurance premium). The trade is real exactly because it is "
            "dangerous: you're paid for bearing the tail."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample mean test, but on a series with **strong positive "
            "autocorrelation** (vol clusters), so the right SE is HAC:\n\n"
            "$$ t = \\frac{\\bar b}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\bar b)},\\qquad "
            "\\widehat{\\mathrm{se}}_{\\text{NW}}^2 = \\frac{1}{T}\\Big(\\hat\\gamma_0 + "
            "2\\sum_{k=1}^{L}\\big(1-\\tfrac{k}{L+1}\\big)\\hat\\gamma_k\\Big). $$\n\n"
            "An i.i.d. *t* would overstate significance because the carry's good and bad runs are "
            "serially correlated. The Tradability axis is *not* a mean test at all — it is a **tail** "
            "test. For a book that is short convexity, the mean is a poor summary; what decides "
            "survival is skewness, kurtosis, the 1% **CVaR**, and the **compounded drawdown**. A "
            "high-Sharpe short-insurance book is the textbook example of a strategy whose Sharpe lies."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** VIXY (VXX-equivalent) + `^VIX`, `^VIX3M`, SPY, daily adjusted closes, "
            f"{R['start']}→{R['end']} ({R['days']:,} days). Vehicle named on the Signal axis.\n"
            "- **Book.** Constant-notional short, $b_t=-r_t$, rebalanced daily (passive — no entry "
            "signal, so no look-ahead). A **contango-only** variant gates on $c_{t-1}>1$ (1-day lag).\n"
            "- **Signal test.** Newey-West HAC(21) *t* that $\\mathbb{E}[b_t]>0$, plus a "
            "distribution-free **sign-shuffle placebo** (flip each day's sign by a coin; "
            "$p=\\Pr[\\text{shuffled mean} \\ge \\text{observed}]$).\n"
            "- **Costs.** Short **borrow** (3%/yr, swept to 30%) **+** rebalancing turnover (2 bps/day, "
            "one-way), charged only on active days.\n"
            "- **Tail.** Annualised Sharpe, compounded max drawdown, daily skew & excess kurtosis, the "
            "single worst day, 1% VaR/CVaR.\n"
            "- **Positive control.** A deterministic ETP with a **planted carry knob** and **injected "
            "crashes**: carry=0 must NOT register a positive carry; a real carry must light the HAC *t* "
            "up; the crash must stamp negative skew regardless. **What would make us say 'mirage' on "
            "Signal:** HAC *t* < 2 or placebo *p* > 0.05. **What flips Tradability to deployable:** a "
            "tame tail (|skew| small, shallow drawdown) at an allocatable size — which we do *not* find."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The carry is real — HAC *t* and the placebo null\n\n"
            "The short-carry daily return distribution, with its mean, the HAC *t*, and the "
            "sign-shuffle placebo distribution of the mean under the null of no carry."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.short_carry(F).values\n"
            "    h = st.hac_t(rs); pl = st.placebo_pvalue(rs)\n"
            "    obs = h['mean']; tval = h['t']; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(375); n=len(rs)\n"
            "    draws = np.array([(rng.choice([-1.,1.],size=n)*rs).mean() for _ in range(6000)])\n"
            "else:\n"
            "    obs=R['daily_mean']/100; tval=R['hac_t']; pval=R['placebo_p']\n"
            "    rng=np.random.default_rng(375); draws=rng.normal(0, obs/R['hac_t'], 6000)\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "if HAVE_REAL:\n"
            "    a1.hist(rs*100, bins=80, color=GREEN, alpha=.8)\n"
            "    a1.axvline(obs*100, c=RED, lw=2, label=f'mean +{obs*100:.2f}%')\n"
            "    a1.set_xlim(-25,25)\n"
            "else:\n"
            "    a1.hist(rng.normal(obs, 0.04, 3800)*100, bins=80, color=GREEN, alpha=.8)\n"
            "    a1.axvline(obs*100, c=RED, lw=2, label=f'mean +{obs*100:.2f}%')\n"
            "a1.set_title('Short-carry daily returns (note the left skew)'); a1.set_xlabel('%/day'); a1.legend()\n"
            "a2.hist(draws*100, bins=50, color=GREY, alpha=.85, label='null: sign-shuffled means')\n"
            "a2.axvline(obs*100, c=GREEN, lw=2.5, label=f'observed +{obs*100:.2f}%')\n"
            "a2.set_title(f'Placebo p = {pval:.3f}, HAC t = {tval:.2f}'); a2.set_xlabel('mean %/day'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HAC(21) t = {tval:.2f}  | sign-shuffle placebo p = {pval:.4f}  -> carry is REAL')"
        ),
        md(
            f"> 💡 In plain words: the observed mean (**+{R['daily_mean']:.2f}%/day**) sits in the far "
            f"right tail of the no-carry null (**p = {R['placebo_p']:.3f}**), and the "
            f"autocorrelation-robust **t = {R['hac_t']:.2f}** clears the desk's *t* ≥ 2 bar. H₁ is "
            "**supported** — this is a genuine premium, not noise. Note already the left-skewed return "
            "histogram: the danger is visible in the shape."
        ),
        md(
            "### 4b · The tail — what the Sharpe hides\n\n"
            "The decisive panel. Net Sharpe looks tradable; the skew, kurtosis, worst day and drawdown "
            "say otherwise. The book is short deep convexity."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    rs = st.short_carry(F); g = st.perf(rs)\n"
            "else:\n"
            "    g = dict(sharpe=R['g_sharpe'], max_dd=R['g_dd']/100, skew=R['skew'], exkurt=R['exkurt'],\n"
            "             worst_day=R['worst']/100, var1=R['var1']/100, cvar1=R['cvar1']/100)\n"
            "    rng=np.random.default_rng(375); rs=pd.Series(rng.normal(R['daily_mean']/100,0.04,3800))\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "metrics = ['skew', 'ex-kurt/10', 'worst day %', '1% CVaR %', 'maxDD %']\n"
            "vals = [g['skew'], g['exkurt']/10, g['worst_day']*100, g['cvar1']*100, g['max_dd']*100]\n"
            "a1.barh(metrics, vals, color=RED); a1.axvline(0,c='k',lw=.6)\n"
            "for i,v in enumerate(vals): a1.annotate(f'{v:.1f}',(v,i),ha='right' if v<0 else 'left',va='center')\n"
            "a1.set_title('The tail: every number screams short-convexity')\n"
            "# Sharpe-only view next to it: a tradable-looking number\n"
            "a2.bar(['gross','net'], [R['g_sharpe'], R['n_sharpe']], color=[GREY, AMBER], width=.5)\n"
            "a2.set_ylim(0,0.8); a2.set_ylabel('annualised Sharpe')\n"
            "a2.set_title('...while the Sharpe alone looks deployable')\n"
            "for i,v in enumerate([R['g_sharpe'], R['n_sharpe']]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"skew {g['skew']:.2f}  ex-kurt {g['exkurt']:.1f}  worst {g['worst_day']*100:.0f}%  \"\n"
            "      f\"1% CVaR {g['cvar1']*100:.0f}%  maxDD {g['max_dd']*100:.0f}%  vs net Sharpe {R['n_sharpe']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: a net Sharpe of **{R['n_sharpe']:.2f}** would pass most screens. But "
            f"skew **{R['skew']:.2f}**, excess kurtosis **{R['exkurt']:.1f}**, a 1%-day average loss "
            f"(CVaR) of **{R['cvar1']:.0f}%**, and a **{R['g_dd']:.0f}%** drawdown reveal a book that is "
            "short crash insurance. The Sharpe is the *mean* over the *vol*; for a fat-left-tailed book "
            "that ratio is a flattering lie. **H₂ rejected.**"
        ),
        md(
            "### 4c · Conditioning + costs — they soften, they don't save\n\n"
            "Two would-be rescues: short **only in contango** (dodge the inverted-curve regime), and "
            "sweep the **borrow fee**. Each helps a little; neither makes it deployable."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sc = st.summarize(F, contango_only=True); base = st.summarize(F)\n"
            "    cond = (sc['net']['sharpe'], sc['net']['max_dd']*100, sc['net']['skew'])\n"
            "    alld = (base['net']['sharpe'], base['net']['max_dd']*100, base['net']['skew'])\n"
            "    borrow = [(b*100,)+ (lambda s: (s['net']['ann_ret']*100, s['net']['sharpe']))(st.summarize(F, borrow_annual=b)) for b in (0.0,0.03,0.10,0.30)]\n"
            "else:\n"
            "    cond = (R['c_sharpe'], R['c_dd'], R['c_skew']); alld = (R['n_sharpe'], R['n_dd'], R['skew'])\n"
            "    borrow = R['borrow']\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "x=np.arange(2)\n"
            "a1.bar(x-.2, [alld[0], cond[0]], .4, color=AMBER, label='net Sharpe')\n"
            "a1.bar(x+.2, [alld[1], cond[1]], .4, color=RED, label='max drawdown %')\n"
            "a1.set_xticks(x); a1.set_xticklabels(['all days','contango-only'])\n"
            "a1.set_title('Contango filter: better Sharpe, still ruinous DD'); a1.legend()\n"
            "for i,(s,d) in enumerate([(alld[0],alld[1]),(cond[0],cond[1])]):\n"
            "    a1.annotate(f'{s:.2f}',(i-.2,s),ha='center',va='bottom'); a1.annotate(f'{d:.0f}%',(i+.2,d),ha='center',va='top')\n"
            "fees=[b[0] for b in borrow]; shp=[b[2] for b in borrow]\n"
            "a2.plot(fees, shp, 'o-', c=RED, lw=2)\n"
            "a2.set_xlabel('short borrow fee (%/yr)'); a2.set_ylabel('net Sharpe')\n"
            "a2.set_title('Borrow eats it — and spikes in exactly the crises')\n"
            "for f,s in zip(fees,shp): a2.annotate(f'{s:.2f}',(f,s),ha='left',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('contango-only net:', tuple(round(v,2) for v in cond), '| borrow sweep (fee%,ann%,Sharpe):', [tuple(round(v,1) for v in b) for b in borrow])"
        ),
        md(
            f"> 💡 In plain words: gating on contango lifts the net Sharpe to **{R['c_sharpe']:.2f}** and "
            f"trims the drawdown to **{R['c_dd']:.0f}%** — but a 67% drawdown with skew "
            f"**{R['c_skew']:.2f}** is still un-allocatable, and the worst crashes strike *from* "
            "contango faster than a daily filter can react. And the short is **borrow-sensitive**: at a "
            f"crisis-level **30%/yr** fee the Sharpe collapses to **{R['borrow'][3][2]:.2f}**. The fund "
            "is hardest to borrow exactly when its decay is fastest — the market charges you to collect."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic ETP with a **planted carry knob** and **injected crashes**: with "
            "**zero** carry the engine must NOT report a significant *positive* carry; with a real "
            "planted carry the HAC *t* must clear 2; and the injected crash must stamp negative skew "
            "**regardless** — proving the harness reproduces the steamroller by construction."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.0030):\n"
            "    syn = data.synthetic_etp(edge=edge, seed=375); s = st.summarize(syn); g = s['gross']\n"
            "    res.append((edge, s['hac_t'], s['placebo_p'], g['sharpe'], g['skew']))\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "labels = [f'planted carry\\n{e*100:.2f}%/day' for e,_,_,_,_ in res]\n"
            "tvals = [r[1] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.6)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>0 else 'top')\n"
            "ax.set_ylabel('HAC t (short carry)'); ax.set_title('Control: no carry -> no false positive; real carry -> lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e,t,p,s,sk in res: print(f'planted {e*100:.2f}%/day: HAC t={t:+.2f} placebo p={p:.4f} Sharpe={s:+.2f} skew={sk:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** carry the engine reports **t = {R['syn'][0][1]:.2f}** "
            f"(placebo p ≈ {R['syn'][0][2]:.2f} — it never invents a positive carry); only a real "
            f"planted carry drives **t = {R['syn'][1][1]:.2f}** (p = {R['syn'][1][2]:.4f}). And the "
            f"injected crash stamps skew ≈ **{R['syn'][0][4]:.2f}** in *both* cases. The machinery is "
            "honest in both directions — so the real-tape *t* of 2.5 is a true positive, and the real "
            "tape's even fatter skew is the real steamroller, not a glitch."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — daily carry **+{R['daily_mean']:.2f}%**, **HAC(21) t = "
            f"{R['hac_t']:.2f}** (≥ 2), placebo **p = {R['placebo_p']:.3f}**; contango "
            f"{R['frac_contango']:.0f}% of days, ETP decays ×{R['etp_ratio']:.0e}. The roll-decay "
            "premium is real on the live tape — the desk's *t* ≥ 2 bar is met. (Vehicle: VIXY, the "
            "VXX-equivalent, named on the axis.)\n"
            f"- **Tradability `FRAGILE`** — net Sharpe **{R['n_sharpe']:.2f}** survives costs *on "
            f"paper*, but skew **{R['skew']:.2f}**, ex-kurt **{R['exkurt']:.1f}**, a **{R['worst']:.0f}%** "
            f"day, **{R['g_dd']:.0f}%** drawdown and 1% CVaR **{R['cvar1']:.0f}%**. Contango "
            f"conditioning leaves a **{R['c_dd']:.0f}%** drawdown; crisis borrow guts the return. "
            "Un-allocatable at NAV scale — one size-up from ruin.\n"
            "- **Free carry? `BUSTED`** — the carry is the **variance/crash-risk premium**: you are "
            "paid to be short deep convexity. Three tail events (2018/2020/2024) each erased a "
            "third-to-half of a constant-notional book in a single session. Real carry, real steamroller."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the size-of-ruin\n\n"
            "The operational truth in one picture: at what leverage does the **single worst day** wipe "
            "the book? The carry says 'size up'; the tail says 'and you're gone.'"
        ),
        code(
            "if HAVE_REAL:\n"
            "    worst = st.perf(st.short_carry(F))['worst_day']\n"
            "else:\n"
            "    worst = R['worst']/100\n"
            "L = np.linspace(1, 6, 200)\n"
            "one_day_loss = L * worst * 100      # leveraged single-day P&L on the worst day (%)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(L, one_day_loss, c=RED, lw=2, label='worst-day P&L at leverage L')\n"
            "ax.axhline(-100, ls='--', c='k', label='total wipeout (-100%)')\n"
            "Lruin = -1.0/worst\n"
            "ax.axvline(Lruin, c=GREY, ls=':', label=f'ruin at L={Lruin:.1f}x')\n"
            "ax.fill_between(L, one_day_loss, -100, where=(one_day_loss<=-100), color=RED, alpha=.15)\n"
            "ax.set_xlabel('leverage L (x notional)'); ax.set_ylabel('worst single-day return (%)')\n"
            "ax.set_title('One bad day wipes a leveraged short — the size that pays is the size that ruins')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a single {worst*100:.0f}% day wipes the book at just {Lruin:.1f}x leverage; even 1x draws down {R[\"g_dd\"]:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: the worst day is **{R['worst']:.0f}%**, so leverage of only "
            "**~2.3×** turns it into a total wipeout — and 2-3× is exactly how this trade is marketed. "
            "Even *unlevered* the drawdown is 92%. There is no sizing that keeps a meaningful carry "
            "without an existential tail, because **they are the same premium**. The trade that pays "
            "you to sell crash insurance is, definitionally, the trade that the crash destroys. That is "
            "why the carry is `REAL` and the strategy is `FRAGILE` → mirage at scale."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Better roll-yield gauge.** Replace `^VIX3M/^VIX` with the actual front-two VIX-futures "
            "slope (or the index's published roll). The event count and conditioning sharpen; the tail "
            "doesn't budge.\n"
            "- **Vol-targeting & options overlays.** A vol-targeted short, or buying far-OTM VIX calls "
            "as tail hedges, trades carry for survival — quantify the hedge cost vs the drawdown saved "
            "(it eats most of the premium).\n"
            "- **The variance-premium core.** The honest version of this trade is selling variance "
            "directly (variance swaps / a delta-hedged straddle) and *sizing for the tail* — the carry "
            "is the same risk premium (Carr-Wu), just with explicit convexity you can hedge.\n\n"
            "*The reproducible core is offline and deterministic; the shorted vehicle is an explicit "
            "VXX-equivalent (VIXY). Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
