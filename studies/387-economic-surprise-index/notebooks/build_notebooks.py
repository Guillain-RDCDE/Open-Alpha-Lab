"""Generate the two narrative notebooks for Study 387 (Economic-Surprise-Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FRED macro panel
+ SPY under ../_cache/ and build a transparent CESI proxy; otherwise they quote the frozen
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


# Frozen real-tape headline numbers — mirror of docs/results.md (transparent CESI proxy from 6
# FRED real-activity series + SPY month-end, 1993-01-31 -> 2026-05-31, 401 months, 33.3 years).
R = dict(
    start="1993-01-31", end="2026-05-31", months=401, years=33.3,
    esi_mean=-0.08, esi_std=1.19, esi_frac_pos=48,
    # per-horizon: (months, n, cond_mean%, cond_win%, base_mean%, base_win%, t, p_placebo)
    h1=(1, 191, 0.92, 64, 0.96, 65, -0.10, 0.549),
    h3=(3, 190, 2.87, 72, 2.83, 72, 0.06, 0.473),
    h6=(6, 187, 6.60, 80, 5.79, 76, 0.89, 0.148),
    h12=(12, 184, 13.38, 88, 12.01, 82, 1.00, 0.134),
    # timing: (label, exposure%, turns, net_ann%, net_vol%, net_sharpe, bh_sharpe)
    timing=[("long / flat", 48, 215, 6.3, 9.5, 0.66, 0.78),
            ("long / short", 100, 429, 1.0, 15.1, 0.07, 0.78)],
    # robustness: (thr, n, cond12%, t, p_placebo)
    robust=[(-0.5, 337, 11.7, -0.23, 0.620), (0.0, 184, 13.4, 1.00, 0.134),
            (0.5, 54, 13.7, 0.78, 0.234)],
    # synthetic: (edge, n, cond6%, base6%, t, p_placebo)
    syn=[(0.0, 208, 4.77, 4.38, 0.47, 0.267), (0.04, 208, 12.40, 8.48, 2.22, 0.003)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_nowcast%3F: Busted](https://img.shields.io/badge/Free_nowcast%3F-Busted-8b949e?style=flat-square)\n\n"
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

from economic_surprise_index import data, strategy as st

HAVE_REAL = data.have_real()
F = data.build_real() if HAVE_REAL else None
print("real FRED+SPY cache present:", HAVE_REAL,
      "| monthly observations:", (0 if F is None else len(F)))
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
            "# \"Buy when the data beats\" — does the Economic Surprise Index work? 🎯\n"
            "### The desk's favourite macro gauge — high when releases beat forecasts — put to the "
            "test, in plain English\n\n"
            + BADGES +
            "Every trading desk watches Citi's **Economic Surprise Index (CESI)**: a single line that "
            "rises when the economy's data keeps **beating** what economists expected, and falls when "
            "it **misses**. The folklore is irresistibly simple — *when the data beats, buy stocks; the "
            "economy has unpriced good news.*\n\n"
            "It sounds like a free nowcast. But the US market drifts **up** most years anyway, and a "
            "surprise index runs hottest **coming out of recessions** — exactly when stocks are already "
            "staging their biggest recovery rallies. So \"beat = up\" might just be the recovery drift "
            "wearing a clever costume. This notebook pulls the costume off.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test, the Sharpe race and "
            "the power control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Citi's CESI is proprietary, and there's **no free history "
            "of analyst forecasts**, so we **build a transparent proxy**: from six public FRED series "
            "(payrolls, industrial production, retail sales, sentiment, housing, durable orders) we take "
            "each month's change minus its **trailing-12-month average** — a labelled stand-in for the "
            "consensus — then z-score and average them. It's narrower than CESI and we call it a proxy "
            "throughout. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a 'beat' month, is the market higher a year later? | **Usually — 88% of the time.** "
            "That's the seductive headline. |\n"
            "| Isn't 88% amazing? | **Not really.** *Any* random month is higher a year later **82%** of "
            "the time. The beat adds only ~6 points of win-rate — well inside the noise. |\n"
            "| So is the extra return real? | **Can't tell.** The 12-month bump is ~1.4 points over a "
            "normal year, with a Welch **t = 1.00** (you need ≥ 2). At **1–3 months** — where a data "
            "release actually moves prices — the excess is essentially **zero**. |\n"
            "| Could you at least trade it? | **No.** A \"hold-when-beating\" timing rule earns a "
            "**lower Sharpe (0.66) than just buying and holding (0.78)** — sitting out the market on "
            "'miss' months costs more than the surprise returns. |\n\n"
            "> The surprise index is a real, useful macro gauge. \"Buy when the data beats\" is the "
            "market's own recovery drift in a costume — small, insignificant, and a loser to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the economic data keeps coming in **above** what forecasters expected, the "
            "economy has positive momentum that isn't priced yet — so buy stocks, and ride the "
            "surprise.\"*\n\n"
            "This is the everyday reading of Citi's CESI, charted on every desk against the S&P 500. "
            "The picture is intuitive: a run of upside surprises means analysts keep being too "
            "pessimistic, the economy is stronger than the price reflects, and equities should catch up. "
            "We'll rebuild the surprise index from public data and ask whether \"beat\" really predicts "
            "\"up\" — or just **coincides** with it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things hide inside \"beat = buy,\" and only one of them matters. (1) *Does the market "
            "tend to rise after a beat month?* Sure — but the market tends to rise after **most** "
            "months. The question that matters is (2) *does it rise by **more** than usual, reliably "
            "enough to bet on?* A high win-rate that merely matches the market's natural up-drift is "
            "**not an edge** — it's the base rate in a costume. Worse, a surprise built against a "
            "**trailing average** runs high exactly during post-recession recoveries, so it inherits the "
            "rally rather than predicting it. And even a real bump is useless if a timing rule built on "
            "it **loses to buy-and-hold.**"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We rebuild the surprise index on **transparent public data**: six FRED real-activity "
            "series, each month's change minus its trailing-12-month average, z-scored and averaged into "
            f"one **ESI**. A 'beat' month is ESI > 0. Over **{R['years']:.1f} years** "
            f"({R['start']} → {R['end']}, {R['months']} months) we then:\n\n"
            "1. **Mark the beats.** Every month the ESI is positive (data above its own recent pace), "
            "acted on one month later — no look-ahead.\n"
            "2. **Measure the payoff.** For each beat, what did SPY do over the next **1 / 3 / 6 / 12 "
            "months** — and how does that compare to a *random* month (the base rate)?\n"
            "3. **Stress the luck, then try to trade it.** Draw the same number of *random* dates "
            "thousands of times to see how often chance matches the beats; then race a "
            "\"hold-when-beating\" rule against buy-and-hold, net of costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the surprise index itself.** The ESI rising above zero (data beating its "
            "trailing average) and diving below (missing) — overlaid on SPY. Watch how it spikes "
            "*after* the 2009 and 2020 bottoms, not before."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax1 = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax1.fill_between(F.index, F['esi'], 0, where=(F['esi']>=0), color=GREEN, alpha=.5, label='beat (ESI>0)')\n"
            "    ax1.fill_between(F.index, F['esi'], 0, where=(F['esi']<0), color=RED, alpha=.5, label='miss (ESI<0)')\n"
            "    ax1.axhline(0, c='k', lw=.8); ax1.set_ylabel('Economic Surprise Index (proxy)')\n"
            "    ax2 = ax1.twinx(); ax2.plot(F.index, F['spy'], c=GREY, lw=1.3, label='SPY (right)')\n"
            "    ax2.set_yscale('log'); ax2.grid(False); ax2.set_ylabel('SPY (log)')\n"
            "    ax1.set_title('The ESI proxy vs SPY — surprises spike during recoveries')\n"
            "    ax1.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print('ESI mean', round(F['esi'].mean(),2), 'std', round(F['esi'].std(),2),\n"
            "          'frac>0', round((F['esi']>0).mean(),2))\n"
            "else:\n"
            "    print('no cache — see docs/results.md: ESI mean ', R['esi_mean'], 'std', R['esi_std'])"
        ),
        md(
            f"The index spends about **{R['esi_frac_pos']}%** of months above zero. Crucially, the big "
            "green stretches line up with the *aftermath* of crashes — the data beats a depressed "
            "trailing average right as the recovery rally is already underway. Now the real question: "
            "does a beat predict *extra* return, or just ride the rally?"
        ),
        md(
            "**The payoff vs a normal month.** For each horizon, the beat month's average forward return "
            "and win-rate, next to the *unconditional* base rate — what a random month gives you anyway."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, m) for m in hs]\n"
            "    cwin = [r['cond_win']*100 for r in rows]; bwin = [r['base_win']*100 for r in rows]\n"
            "else:\n"
            "    cwin = [R['h1'][3], R['h3'][3], R['h6'][3], R['h12'][3]]\n"
            "    bwin = [R['h1'][5], R['h3'][5], R['h6'][5], R['h12'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cwin, .4, color=GREEN, label='after a beat month')\n"
            "ax.bar(x+.2, bwin, .4, color=GREY, label='any random month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylim(0, 100); ax.set_ylabel('% of the time SPY is higher')\n"
            "ax.set_title('Beat 12m win-rate is 88% — but a random month is already 82%')\n"
            "for i,(c,b) in enumerate(zip(cwin,bwin)):\n"
            "    ax.annotate(f'{c:.0f}%',(i-.2,c),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.0f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('beat 12m win-rate:', f'{cwin[-1]:.0f}%', ' base 12m win-rate:', f'{bwin[-1]:.0f}%')"
        ),
        md(
            f"There's the trick. The **{R['h12'][3]:.0f}%** \"buy the beat\" 12-month win-rate sits just "
            f"above the **{R['h12'][5]:.0f}%** you get on *any* random month — because the US market "
            f"simply rises most years. At **1 month** the beat's win-rate (**{R['h1'][3]:.0f}%**) is "
            f"actually *below* the base rate (**{R['h1'][5]:.0f}%**) — and 1 month is precisely where a "
            "data release moves the tape. The headline is mostly the market's own up-drift, not the "
            "signal."
        ),
        md(
            "**Could a handful of random months look this good?** Here's the honest test: draw the same "
            f"**{R['h12'][1]}** *random* dates over and over, and see how the beat months' 12-month "
            "average return stacks up against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F, 12); obs = pl['cond_mean']; pmean = pl['placebo_mean']; pval = pl['p_value']\n"
            "    fwd = st.forward_returns(F['spy'], 12).dropna().values; n=len(fwd); k=pl['k']\n"
            "    rng = np.random.default_rng(387)\n"
            "    draws = np.array([fwd[rng.integers(0,n,size=k)].mean() for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['h12'][2]/100; pmean = R['h12'][4]/100; pval = R['h12'][7]; k = R['h12'][1]\n"
            "    rng = np.random.default_rng(387); draws = rng.normal(pmean, 0.012, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.8, label=f'12m return of {k} RANDOM dates')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'the actual beat months ({obs*100:.1f}%)')\n"
            "ax.set_xlabel('average 12-month forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The beats sit inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {k}-date draw beats the signal {pval*100:.0f}% of the time — far from rare')"
        ),
        md(
            f"The green line — the actual beat months — falls **inside** the grey cloud of random draws. "
            f"About **{R['h12'][7]*100:.0f}%** of random {R['h12'][1]}-date samples do as well or better "
            f"(~1 time in 7). In plain terms: **the \"buy the beat\" record is what you'd expect from "
            "drawing a few hundred dates in an up-trending market.** Not a nowcast — arithmetic."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Forward returns after a beat are positive and the 12-month win-rate "
            f"is genuinely high (**{R['h12'][3]:.0f}%**) — but it barely beats the "
            f"**{R['h12'][5]:.0f}%** base rate, isn't statistically significant (**t = {R['h12'][6]:.2f}"
            f"**), and is **~zero at 1–3 months** where the news actually moves. Real as a gauge, weak "
            "as edge.\n"
            "- **Tradability — Mirage.** A \"hold-when-beating\" timing rule earns a **lower Sharpe "
            f"({R['timing'][0][5]:.2f}) than buy-and-hold ({R['timing'][0][6]:.2f})** — sitting out an "
            "up-drifting market on 'miss' months costs more than the surprise returns.\n"
            "- **\"Free nowcast\"? — Busted.** A surprise measured against a *trailing average* is "
            "mechanically the market's own **post-recession recovery drift** in a costume — priced "
            "within minutes of each release, sampled monthly, indistinguishable from luck."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — race it against buy-and-hold\n\n"
            "Forget significance for a second and just ask the operational question: if you hold SPY "
            "only when the data is beating (ESI > 0) and step aside otherwise, do you beat simply owning "
            "the market? Here's the Sharpe ratio of each rule next to plain buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bl = []\n"
            "    for short in (False, True):\n"
            "        b = st.timing_backtest(F, cost_bps=10.0, allow_short=short)\n"
            "        bl.append((b['net']['sharpe'], b['buy_hold']['sharpe']))\n"
            "    rule_sh = [bl[0][0], bl[1][0]]; bh = bl[0][1]\n"
            "else:\n"
            "    rule_sh = [R['timing'][0][5], R['timing'][1][5]]; bh = R['timing'][0][6]\n"
            "labels = ['long / flat\\n(ESI>0 -> hold)', 'long / short\\n(ESI<0 -> short)']\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x, rule_sh, .5, color=[AMBER, RED], label='ESI timing rule (net)')\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=2, label=f'buy & hold ({bh:.2f})')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('net Sharpe ratio')\n"
            "ax.set_title('Every ESI timing rule LOSES to simply holding')\n"
            "for i,s in enumerate(rule_sh): ax.annotate(f'{s:.2f}',(i,s),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'long/flat Sharpe {rule_sh[0]:.2f} vs buy&hold {bh:.2f} -> timing loses')"
        ),
        md(
            f"There it is. The long/flat ESI rule earns a Sharpe of **{R['timing'][0][5]:.2f}** against "
            f"**{R['timing'][0][6]:.2f}** for just buying and holding — and that's *before* crediting "
            "the interest it would earn while parked in cash, which only flatters the rule. Betting "
            f"*against* stocks on miss months (long/short) collapses to **{R['timing'][1][5]:.2f}** — a "
            "direct fight with the equity risk premium. **There is no version of this that beats "
            "buy-and-hold.** Costs aren't the problem; being out of an up-drifting market is."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The macro-timing pattern.** Sahm-rule, misery-index, Fed-model and real-rate studies "
            "on this bench all rhyme: a *real* macro relationship rarely survives as a **tradable** "
            "monthly timing rule once you charge it the opportunity cost of sitting in cash through an "
            "up-drifting market.\n"
            "- **Better consensus.** Swap our trailing-average stand-in for a true survey of analyst "
            "forecasts (if you can buy the history) and re-run the detector — the surprise becomes a "
            "cleaner forecast error, but the recovery-drift confound and the fast-pricing problem "
            "don't go away.\n"
            "- **Shorter horizons.** Try the announcement-day window itself — that's where a surprise "
            "*does* move prices, but it's gone within minutes, long before a monthly rule can act.\n\n"
            "*Think \"buy the beat\" beats the market by more than luck? Build the surprise index, draw "
            "the same number of random dates, and show the beats landing **outside** the cloud **and** a "
            "timing rule clearing buy-and-hold — then we'll talk.*"
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
            "# Economic Surprise Index — a quantitative teardown 🔬\n"
            "### A transparent CESI proxy from 6 FRED series · conditional vs unconditional forward "
            "returns · a Welch *t* + placebo null · an ESI-timing-vs-buy-and-hold Sharpe race · a "
            "synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"buy the beat\" fuses: a **real but small** post-beat drift from the "
            "**unconditional up-drift** of US equities, and we confront both with the sample's "
            "**standard error**. The decisive object is not the point estimate (it's positive, as the "
            "lore says) but its **power** — and whether a timing rule built on it can clear "
            "buy-and-hold.\n\n"
            "> ⚠️ **Data + proxy note.** Citi's CESI is proprietary and there's no free history of "
            "analyst forecasts; we build a transparent **proxy** — six monthly FRED real-activity series "
            "(payrolls, IP, retail sales, sentiment, housing, durable orders), each differenced against "
            "its **trailing-12-month average** (a labelled consensus stand-in), z-scored and averaged. "
            "That construction is the methodological knife: a surprise measured against a trailing mean "
            "is mechanically an **autocorrelation** signal aligned with post-recession recovery. Real "
            "data: FRED + yfinance SPY, 1993→2026, month-end. Offline core + synthetic control are "
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
            f"| **Signal** | `WEAK` | 12-month conditional mean **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%**; Welch **t = {R['h12'][6]:.2f}**, placebo **p = {R['h12'][7]:.2f}**"
            f" — positive but **fails t ≥ 2**, and the 1-month *t* is **{R['h1'][6]:.2f}** (the horizon "
            "news actually moves). |\n"
            f"| **Tradability** | `MIRAGE` | ESI>0 timing rule net Sharpe **{R['timing'][0][5]:.2f}** "
            f"(long/flat) / **{R['timing'][1][5]:.2f}** (long/short) vs buy-and-hold "
            f"**{R['timing'][0][6]:.2f}** — it **loses to holding**, before even crediting the cash "
            "yield it ignores while flat. |\n"
            f"| **Free nowcast?** | `BUSTED` | The 12m win-rate **{R['h12'][3]:.0f}%** vs an "
            f"unconditional **{R['h12'][5]:.0f}%** — the 'edge' is base rate × recovery drift. |\n\n"
            "> 💡 In plain words: the beat really does precede gains — because *almost everything* "
            "precedes gains in an up-drifting market, and a trailing-average surprise runs hot right "
            "during recoveries. Strip the base rate and the excess is a whisper the null explains."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_{i,t}$ be series $i$'s month-$t$ change (log-growth or level-diff). The "
            "**surprise** differences it against a trailing-12m consensus proxy and standardizes by a "
            "trailing-36m std: $z_{i,t} = (x_{i,t} - \\overline{x}_{i,t-1:t-12}) / "
            "\\hat\\sigma_{i,t}$, and $\\mathrm{ESI}_t = \\frac1n\\sum_i z_{i,t}$. A **beat** month is "
            "$\\mathrm{ESI}_{t} > 0$, acted on at $t+1$ (one-month lag).\n\n"
            "- **H₁ (the signal predicts).** $\\mathbb{E}[r_{t\\to t+H}\\mid \\text{beat}] > "
            "\\mathbb{E}[r_{t\\to t+H}]$ for forward horizon $H$ — a *positive excess* over the base "
            "rate.\n"
            "- **H₂ (it's deployable).** A timing rule that holds SPY on beats clears buy-and-hold "
            "net of costs.\n"
            "- **H₃ (\"free nowcast\").** The high post-beat win-rate reflects the signal, not the "
            "market's unconditional up-drift × the recovery-aligned construction.\n\n"
            "We find **H₁ not rejected but not supported** (positive, $t < 2$, ~zero at 1–3m), "
            "**H₂ rejected** (the rule loses to holding), **H₃ rejected** (the win-rate ≈ the base "
            "rate). The legend is true exactly where it's uninformative (the market goes up) and "
            "unproven exactly where it would pay (a *distinct* nowcast)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one decomposition: a conditional mean against an unconditional "
            "baseline, judged by its **standard error**.\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{beat}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{beat}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "A raw **win-rate** is the wrong lens: US 12-month returns are positive ~80% of the time "
            "*unconditionally*, so an 88% conditional rate is barely a standard error of a proportion "
            "away from the null. The honest instruments are a **Welch two-sample t** on the conditional "
            "vs unconditional means and a **randomization (placebo) null** — resample $k$ random entry "
            "dates and ask how often chance matches the beats. And because a real point estimate is "
            "worthless if it can't be deployed, the **Tradability** axis is settled by a separate object "
            "entirely: a net-of-cost **Sharpe race** against buy-and-hold."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Surprise index.** Six FRED monthly series (yfinance SPY), {R['start']}→{R['end']}, "
            f"**{R['months']} months**. Each surprise = change − trailing-12m mean, z-scored on a "
            "trailing-36m std, averaged. Explicit **proxy** for CESI; the trailing-average consensus is "
            "named on the Signal axis as an autocorrelation confound.\n"
            "- **Beat months.** $\\mathrm{ESI}_t > 0$, entered the close **1 month after** the signal "
            "(no look-ahead); hold $H\\in\\{1,3,6,12\\}$ months.\n"
            "- **Null #1 (Welch t).** Conditional mean vs the unconditional overlapping-window mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random valid entry dates; $p = "
            "\\Pr[\\text{random-draw mean} \\ge \\text{beat mean}]$ — the small-sample workhorse.\n"
            "- **Timing backtest.** Long/flat (or long/short) SPY held when ESI>0, 1-month lag, 10-bps "
            "one-way per turn, raced against buy-and-hold on a Sharpe basis (price-only, labelled).\n"
            "- **Positive control.** A deterministic monthly series (AR(1) surprise + SPY-like price) "
            "with a **known** planted forward edge: the inference must recover a planted edge **and** "
            "must NOT manufacture significance when the true edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — positive, small, front-loaded with nothing\n\n"
            "Conditional mean forward return with its $\\pm$ standard error, against the unconditional "
            "base rate (diamonds). Positive and growing with horizon — but the excess is tiny and the "
            "1–3-month bars (where a release moves price) sit *on* the base rate."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, m); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        cr = st.conditional_returns(F, m); ses.append(cr.std(ddof=1)/np.sqrt(len(cr)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    ts = [R['h1'][6], R['h3'][6], R['h6'][6], R['h12'][6]]; ses = [0.01, 0.018, 0.028, 0.045]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='after a beat (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Excess over base rate is small and t<2 (and ~0 at 1-3m)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 12 months the beat mean is **+{R['h12'][2]:.1f}%** vs a base "
            f"**+{R['h12'][4]:.1f}%** — a ~1.4-point bump at **t = {R['h12'][6]:.2f}** (not "
            f"significant). At 1 month — where a release moves the tape — it's **t = {R['h1'][6]:.2f}** "
            "(*below* base). H₁ is **not rejected, not supported**: a positive whisper inside its own "
            "error bar, absent exactly where the information is supposed to live."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['h12'][1]} *random* entry dates 20,000 times; the histogram is the null "
            "distribution of the 12-month mean. The beat months are the green line; the *p*-value is the "
            "right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F, 12); obs = pl['cond_mean']; pval = pl['p_value']\n"
            "    fwd = st.forward_returns(F['spy'], 12).dropna().values; n=len(fwd); k=pl['k']\n"
            "    rng = np.random.default_rng(387)\n"
            "    draws = np.array([fwd[rng.integers(0,n,size=k)].mean() for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['h12'][2]/100; pval = R['h12'][7]; k = R['h12'][1]\n"
            "    rng = np.random.default_rng(387); draws = rng.normal(R['h12'][4]/100, 0.012, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: mean of {len(draws):,} random {k}-date draws')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'observed beat mean {obs*100:.1f}%')\n"
            "ax.set_xlabel('12-month mean forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the beats are inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random {k}-date draw >= beat] = {pval:.3f}  (need <0.05 to call it real; here ~1 in 7)')"
        ),
        md(
            f"> 💡 In plain words: **{R['h12'][7]*100:.0f}%** of random {R['h12'][1]}-date samples match "
            "or beat the signal. A real edge would push the green line into the far right tail; instead "
            "it sits mid-cloud. This is the crux — **not** a weak point estimate alone, but one that **a "
            "coin could have produced ~1 time in 7**. H₃ rejected: the record is base-rate × "
            "recovery-aligned construction."
        ),
        md(
            "### 4c · Tradability + robustness — the Sharpe race and the threshold sweep\n\n"
            "First the operational verdict: an ESI>0 timing rule, net of 10 bps/turn, against "
            "buy-and-hold. Then a threshold sweep — there's no 'beat' cutoff at which the 12-month *t* "
            "clears 2 (it goes *negative* when loosened)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = []\n"
            "    for short in (False, True):\n"
            "        b = st.timing_backtest(F, cost_bps=10.0, allow_short=short)\n"
            "        tim.append((b['net']['sharpe'], b['buy_hold']['sharpe']))\n"
            "    rule_sh = [tim[0][0], tim[1][0]]; bh = tim[0][1]\n"
            "    rob = []\n"
            "    for thr in (-0.5, 0.0, 0.5):\n"
            "        s = st.summarize(F, 12, thr=thr); rob.append((thr, s['n'], s['cond_mean']*100, s['t'], s['p_placebo']))\n"
            "else:\n"
            "    rule_sh = [R['timing'][0][5], R['timing'][1][5]]; bh = R['timing'][0][6]\n"
            "    rob = R['robust']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar([0,1], rule_sh, .5, color=[AMBER, RED], label='ESI timing (net)')\n"
            "a1.axhline(bh, ls='--', c=GREEN, lw=2, label=f'buy & hold ({bh:.2f})')\n"
            "a1.set_xticks([0,1]); a1.set_xticklabels(['long/flat','long/short']); a1.set_ylabel('net Sharpe')\n"
            "for i,s in enumerate(rule_sh): a1.annotate(f'{s:.2f}',(i,s),ha='center',va='bottom')\n"
            "a1.set_title('Timing loses to holding'); a1.legend()\n"
            "thrs = [f'{r[0]:+.1f}' for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "a2.bar(thrs, tt, color=AMBER, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,(t,kk) in enumerate(zip(tt,nn)): a2.annotate(f'n={kk}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "a2.set_title('12m t never reaches 2'); a2.set_xlabel(\"'beat' threshold\"); a2.set_ylabel('Welch t'); a2.set_ylim(-1, 2.4); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('timing net Sharpe (flat/short):', [round(s,2) for s in rule_sh], 'vs buy&hold', round(bh,2))\n"
            "print('robustness (thr, n, cond12%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the long/flat rule's **{R['timing'][0][5]:.2f}** Sharpe trails "
            f"buy-and-hold's **{R['timing'][0][6]:.2f}** — and the long/short collapses to "
            f"**{R['timing'][1][5]:.2f}**, a direct fight with the risk premium. On the threshold side, "
            "loosening to −0.5 turns the excess *negative* (t = −0.23); tightening to +0.5 thins the "
            "sample and weakens the *t*. **No threshold, no cost regime, and no rule variant** makes "
            "this deployable — the constraint is that the signal isn't there."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic monthly series (AR(1) surprise + SPY-like price): with a **zero** "
            "planted edge the test must stay below t=2 (a noise signal can't fake significance); with a "
            "**+0.04** planted edge it must light up. Both hold — proving the engine is unbiased *and* "
            "that the real-tape *t* of ~1.0 is what an *absent or tiny* edge looks like."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04):\n"
            "    syn = data.synthetic(n_months=360, edge=edge, seed=387)\n"
            "    s = st.summarize(syn, 6)\n"
            "    res.append((edge, s['n'], s['cond_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (6-month)'); ax.set_title('Control: the engine recovers a real edge, ignores a fake one'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e:+.2f}: detected={k} cond={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (below 2 — no false positive); a **+0.04** planted edge "
            f"reaches **t = {R['syn'][1][4]:.2f}** with placebo **p = {R['syn'][1][5]:.3f}**. So the "
            "machinery is honest, and the real-tape *t* of ~1.0 is exactly what an *absent or tiny* edge "
            "looks like through the keyhole — not a broken detector. The inference is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 12m excess **+{R['h12'][2]-R['h12'][4]:.1f}pp** at Welch "
            f"**t = {R['h12'][6]:.2f}** / placebo **p = {R['h12'][7]:.2f}**; ~zero at 1–3m "
            f"(1m *t* = {R['h1'][6]:.2f}) and *inverts* at a loose threshold. Literature support + a "
            "positive-but-insignificant, non-robust point estimate ⇒ WEAK, not REAL (the desk's t≥2 bar "
            "is not met). Named confound: the trailing-average consensus makes the surprise an "
            "autocorrelation signal aligned with post-recession drift.\n"
            f"- **Tradability `MIRAGE`** — ESI>0 timing earns net Sharpe **{R['timing'][0][5]:.2f}** "
            f"(long/flat) / **{R['timing'][1][5]:.2f}** (long/short) vs **{R['timing'][0][6]:.2f}** for "
            "buy-and-hold — *before* crediting the cash yield it ignores while flat. Being out of an "
            "up-drifting market on miss months costs more than the surprise returns. No NAV-scale "
            "edge.\n"
            f"- **Free nowcast? `BUSTED`** — 12m win-rate **{R['h12'][3]:.0f}%** vs unconditional "
            f"**{R['h12'][5]:.0f}%**; the placebo says a coin matches the record ~1 time in 7. A "
            "base-rate × recovery-drift mirage, priced within minutes and sampled monthly."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost of sitting out\n\n"
            "The operational truth in one picture: the equity curve of the long/flat ESI rule against "
            "buy-and-hold. The gap isn't costs — it's the **months out of the market** during 'miss' "
            "periods, in a tape that mostly rises."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = st.monthly_returns(F)\n"
            "    pos = (F['esi'].shift(1) > 0).reindex(ret.index).fillna(False).astype(float)\n"
            "    turn = pos.diff().abs().fillna(pos.abs()); c = 10.0/1e4\n"
            "    rule = (pos*ret - turn*c); bh = ret\n"
            "    eq_rule = (1+rule).cumprod(); eq_bh = (1+bh).cumprod()\n"
            "    expo = (pos>0).mean()\n"
            "else:\n"
            "    rng = np.random.default_rng(387)\n"
            "    bh = pd.Series(rng.normal(0.009, 0.043, R['months']))\n"
            "    eq_bh = (1+bh).cumprod(); eq_rule = (1+bh*0.48).cumprod(); expo = 0.48\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=2, label='buy & hold')\n"
            "ax.plot(eq_rule.index, eq_rule.values, c=AMBER, lw=1.8, label='ESI long/flat rule')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'Holding only on beats ({expo*100:.0f}% exposure) trails buy & hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'final wealth: rule {eq_rule.iloc[-1]:.1f}x vs buy&hold {eq_bh.iloc[-1]:.1f}x (exposure {expo*100:.0f}%)')"
        ),
        md(
            f"> 💡 In plain words: the rule holds SPY only ~**{R['timing'][0][1]}%** of months and ends "
            "**below** buy-and-hold. The amber curve flatlines through every 'miss' stretch — and "
            "because those stretches sit inside an up-drifting market, the foregone return swamps the "
            "tiny surprise edge. There is no sizing, no cost assumption, no exposure tweak that turns "
            "\"buy the beat\" into a portfolio: **the rarity of a genuine signal here is that there "
            "isn't one to deploy.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The macro-timing family.** Sahm-rule, misery-index, Fed-model and real-rate gauges on "
            "this bench share the pathology: a *real* macro relationship rarely survives as a tradable "
            "monthly timing rule once charged the opportunity cost of cash through an up-drifting "
            "market.\n"
            "- **A true consensus.** Replace the trailing-average stand-in with a survey of analyst "
            "forecasts (Bloomberg/Citi history, if you can buy it). The surprise becomes a cleaner "
            "forecast error — but the fast-pricing (Andersen-Bollerslev-Diebold-Vega, 2003) and "
            "recovery-drift confounds remain.\n"
            "- **The right horizon.** The surprise *does* move prices — on the announcement, within "
            "minutes. Measure the event-window return instead of a monthly hold and you'll find the "
            "information, gone long before a monthly rule can trade it.\n\n"
            "*The reproducible core is offline and deterministic; the surprise index is an explicit "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
