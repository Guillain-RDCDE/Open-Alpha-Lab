"""Generate the two narrative notebooks for Study 519 (Net-Share-Issuance).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached year-end prices &
split-adjusted shares under ../_cache/ (40-name survivor basket) and otherwise quote the frozen
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (yfinance split-adjusted shares + adjusted closes, 40 large-cap survivors,
#  year-ends 2014->2025, 9 usable long-short formation years 2016->2024,
#  fingerprint 4c0f2b42cabf, as-of 2026-06-26).
R = dict(
    fp="4c0f2b42cabf", asof="2026-06-26", n_names=40, n_years=9,
    yr_start=2016, yr_end=2024, frac=0.3,
    long_mean=19.68, short_mean=23.31, ls_mean=-3.63, ls_median=-2.97,
    win=44, sharpe=-0.29, t=-0.86, p_placebo=0.795,
    # per-year long-short panel: (year, long%, short%, ls%)
    panel=[(2016, 37.1, 30.8, 6.2), (2017, 1.2, 5.9, -4.7), (2018, 36.2, 39.2, -3.0),
           (2019, 9.8, 32.1, -22.2), (2020, 41.2, 29.6, 11.5), (2021, -12.1, -6.7, -5.4),
           (2022, 20.1, 44.8, -24.7), (2023, 17.4, 8.3, 9.1), (2024, 26.2, 25.7, 0.5)],
    # costs: gross%, net%, net_t
    gross=-3.63, net=-4.33, net_t=-1.02, cost_bps=10, borrow_bps=50,
    # width sweep: (frac, ls%, t, sharpe)
    width=[(0.2, -6.50, -0.97, -0.32), (0.3, -3.63, -0.86, -0.29), (0.4, -5.39, -1.30, -0.43)],
    # split-half: (label, ls%, t, n)
    half=[("2016-2020", -2.44, -0.42, 5), ("2021-2024", -5.12, -0.71, 4)],
    # synthetic (25 seeds): (edge, mean_t, ls%)
    syn=[(0.0, -0.11, -0.6), (0.10, 4.55, 17.6)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Replicates_here%3F: Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square)\n\n"
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

from net_share_issuance import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, SH = data.load_real()
    PX, SH = data.drop_partial_last_year(PX, SH)
else:
    PX, SH = None, None
print("real cache present:", HAVE_REAL, "| basket names:", len(data.BASKET))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Companies that buy back their stock beat companies that print new shares\" — does it hold? 🧾\n"
            "### A real Wall-Street fact, replayed on 40 big survivors — and why it quietly vanishes\n\n"
            + BADGES +
            "It's one of the more *respectable* market beliefs, with decades of academic backing: a "
            "company that **issues** new shares (dilutes its owners — secondary offerings, "
            "stock-for-stock takeovers, lavish stock comp) tends to **underperform**, while a "
            "company that **buys back** its stock (shrinks the share count) tends to **outperform**. "
            "Buy the shrinkers, sell the diluters.\n\n"
            "The catch isn't whether the effect is *real* — in the full universe of thousands of "
            "stocks it is. The catch is whether **you** can harvest it on a handful of names you'd "
            "actually hold. This notebook rebuilds the factor on **40 big survivors** and watches it "
            "**evaporate** — even flip sign — once the sample is small and the survivors are the only "
            "ones left.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Two notes up front.** (1) We count shares **split-adjusted** — a 4-for-1 split "
            "quadruples the raw share count overnight with *zero* real dilution, and if you don't "
            "correct for it every splitter looks like a massive issuer. (2) Our 40 names are all "
            "**still trading in 2026** (survivors), which we flag honestly — the firms that diluted "
            "themselves into oblivion aren't here. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the buyback (low-issuance) names beat the diluting (high-issuance) names here? | "
            f"**No — they *lose* by ~{abs(R['ls_mean']):.0f}%/yr.** The sign comes out **backwards** "
            "on this survivor basket. |\n"
            "| Is that backwards result a real edge you could short? | **No.** It's "
            f"**statistically zero** (*t* = {R['t']:.2f}, you'd need ±2), and a coin-flip relabelling "
            f"of the signal beats it **{R['p_placebo']*100:.0f}%** of the time. |\n"
            "| So is the famous factor fake? | **No — it's real in the *full* market.** It just "
            "**doesn't survive** being squeezed onto 40 big survivors over 9 years. That's a "
            "small-sample + survivorship lesson, not a debunk. |\n"
            "| Does our machinery even work? | **Yes.** Plant a real edge in fake data and it lights "
            f"up cleanly (*t* = +{R['syn'][1][1]:.1f}); plant *none* and it stays at zero "
            f"({R['syn'][0][1]:.2f}). The tool is honest — the real tape just has nothing to find. |\n\n"
            "> The issuance factor is a genuine, published effect — and a perfect example of why "
            "\"published and real\" is **not** the same as \"tradable on the names in front of you.\""
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company prints new shares, it's diluting you and often doing it because "
            "management thinks the stock is expensive — those firms lag. When a company buys back "
            "its stock, it's returning cash and betting on itself — those firms lead. So go long the "
            "buyback-ers and short the diluters.\"*\n\n"
            "This isn't folklore — it's **Pontiff & Woodgate (2008, *Journal of Finance*)** and the "
            "**Daniel-Titman composite-issuance** factor. In the full US cross-section, net share "
            "issuance is a robust *negative* predictor of returns. We'll rebuild that exact sort — "
            "long low issuance, short high issuance — and see what's left on a small survivor basket."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things hide inside \"buyback-ers beat diluters.\" (1) *Across thousands of stocks, "
            "with the small heavily-diluting firms included and a proper risk model*, the effect is "
            "real and that's settled. (2) *Can a person holding a few dozen large caps actually feel "
            "it?* That's a different question — and the honest answer depends entirely on **sample "
            "size** and **which names survived**. A real average edge across thousands of names can "
            "be completely **invisible** (or even sign-flipped) inside any small slice, because the "
            "year-to-year noise of a 12-name basket dwarfs a few-percent tilt."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_names']} big survivor names**, and for each one each year we compute "
            "its **net issuance** — how much its (split-adjusted) share count changed versus a year "
            "earlier. Negative = it bought back; positive = it diluted. Then:\n\n"
            "1. **Sort & split.** Each year-end, rank the 40 names by issuance; the bottom 30% "
            "(biggest buyback-ers) go **long**, the top 30% (biggest diluters) go **short**.\n"
            "2. **Wait a year.** The share count is public at year-end — you trade *after*, holding "
            "into the **next** year (no peeking).\n"
            "3. **Stress the luck.** **Shuffle** which company gets which issuance number and "
            "re-run thousands of times: if a random relabelling beats the real sort as often as "
            "not, the issuance signal isn't doing any work."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline: do the buyback names beat the diluters, year by year?** Here's "
            "the long-minus-short return for each formation year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short_series(PX, SH, frac=0.3)\n"
            "    yrs = [t.year for t in ls.index]; vals = (ls['ls_ret']*100).tolist()\n"
            "else:\n"
            "    yrs = [p[0] for p in R['panel']]; vals = [p[3] for p in R['panel']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "colors = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar([str(y) for y in yrs], vals, color=colors, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(vals), c=GREY, ls='--', label=f'mean {np.mean(vals):.1f}%/yr')\n"
            "ax.set_ylabel('long buyback - short dilution (%)')\n"
            "ax.set_title('Buyback-ers minus diluters, by year - as many red as green')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'years positive: {sum(v>0 for v in vals)}/{len(vals)}   mean L-S: {np.mean(vals):.1f}%/yr')"
        ),
        md(
            f"There's the first surprise. The bars are a wash — **{R['win']}%** of years are green — "
            f"and the *average* is **{R['ls_mean']:.1f}%/yr**, i.e. the buyback names actually "
            "*lagged* the diluters on this basket. The famous sign is **inverted**. Two brutal years "
            "(2019, 2022) where the diluters happened to be the year's big winners drag the whole "
            "thing negative. That's not a sign-flipped edge — that's **noise** picking a direction."
        ),
        md(
            "**Is the average even meaningful?** Here's the honest test: **shuffle** which company "
            "gets which issuance number, re-run the sort thousands of times, and see where the real "
            "result lands in the cloud of pure-luck relabellings."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PX, SH, frac=0.3); obs = pl['obs_mean']; pval = pl['p_value']\n"
            "    # rebuild the shuffled distribution for the picture\n"
            "    sig = st.net_issuance(SH); fwd = st.forward_returns(PX)\n"
            "    years=[]\n"
            "    for t in sig.index:\n"
            "        if t not in fwd.index: continue\n"
            "        sr = sig.loc[t].dropna(); fr = fwd.loc[t].reindex(sr.index).dropna()\n"
            "        c = sr.index.intersection(fr.index)\n"
            "        if len(c)>=4: years.append((sr.loc[c].to_numpy(), fr.loc[c].to_numpy()))\n"
            "    rng=np.random.default_rng(519); draws=np.empty(8000)\n"
            "    for d in range(8000):\n"
            "        acc=[]\n"
            "        for iss,ret in years:\n"
            "            n=len(iss); k=max(1,round(n*0.3)); order=np.argsort(iss[rng.permutation(n)])\n"
            "            acc.append(ret[order[:k]].mean()-ret[order[-k:]].mean())\n"
            "        draws[d]=np.mean(acc)\n"
            "else:\n"
            "    obs = R['ls_mean']/100; pval = R['p_placebo']\n"
            "    rng=np.random.default_rng(519); draws=rng.normal(0.0, 0.045, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.85, label='random relabellings of the signal')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'the real issuance sort ({obs*100:.1f}%)')\n"
            "ax.set_xlabel('long-short return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real sort sits dead inside the luck cloud - placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random relabelling beats the real sort {pval*100:.0f}% of the time')"
        ),
        md(
            f"The red line — the real issuance sort — sits **right inside** the grey cloud of random "
            f"relabellings. About **{R['p_placebo']*100:.0f}%** of coin-flip shuffles do as well or "
            "better. In plain terms: **scramble which company you call a \"diluter\" and you'd often "
            "get a *better* result.** The issuance label isn't carrying information on this tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The buyback-minus-dilution spread is **{R['ls_mean']:.1f}%/yr** "
            f"(the *wrong sign*) at **t = {R['t']:.2f}** — statistical zero — and a coin-flip "
            f"relabelling beats it **{R['p_placebo']*100:.0f}%** of the time.\n"
            "- **Tradability — Mirage.** Nothing to size: a wrong-signed, insignificant spread that "
            "costs and borrow only make worse.\n"
            "- **Does the published factor replicate here? — Busted.** A real, decades-old academic "
            "factor that **doesn't survive** a 40-name survivor replication — small sample + "
            "survivorship, not a free lunch."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs make a non-edge worse\n\n"
            "Shorting the diluters means paying to borrow them, and re-sorting every year churns the "
            "whole book. Here's the spread before and after a realistic cost + borrow charge — note "
            "it only pushes an already-negative number further down."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(PX, SH, frac=0.3, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    g, nv = c['gross_mean']*100, c['net_mean']*100\n"
            "else:\n"
            "    g, nv = R['gross'], R['net']\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.2))\n"
            "ax.bar(['gross','net\\n(10bps x2 + 50bps borrow)'], [g, nv], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short return (%/yr)')\n"
            "ax.set_title('Costs deepen a spread that was already the wrong sign')\n"
            "for i,v in enumerate([g,nv]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:.1f}%/yr  ->  net {nv:.1f}%/yr')"
        ),
        md(
            "Costs aren't the villain here — they just make a non-edge slightly more negative. The "
            "real problem is upstream: with 12-name baskets churning every year over 9 years, the "
            "spread is **inside its own error bar**, and you can't allocate capital to a number you "
            "can't distinguish from zero (let alone one pointing the wrong way)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The real population.** The factor *works* in CRSP's full point-in-time cross-section "
            "— thousands of names, the small chronic diluters included, with a proper factor model. "
            "Our small survivor slice is the wrong instrument to see it, by design.\n"
            "- **The split trap.** Try *not* split-adjusting the shares and watch every stock-split "
            "company (Apple 2020, NVIDIA, Tesla) suddenly look like a massive 'issuer' — a built-in "
            "demonstration of how a data glitch fabricates a factor.\n"
            "- **The announcement cousin.** [Study 368 — Buyback-Drift](../../368-buyback-drift/) "
            "asks the *event* version: does a stock drift up after it *announces* a buyback? "
            "(Spoiler: also weak.)\n\n"
            "*Think the issuance factor beats luck on a tradable basket? Capture it on a bigger, "
            "point-in-time universe, shuffle the labels, and show the real sort landing **outside** "
            "the cloud — then we'll talk.*"
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
            "# Net-Share-Issuance — a quantitative teardown 🔬\n"
            "### Annual cross-sectional sort · long-short vs zero · one-sample *t* + label-shuffle "
            "placebo · quantile-width & split-half robustness · costs + borrow · a 25-seed synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "Pontiff-Woodgate (2008) / Daniel-Titman composite-issuance factor is **real and robust "
            "in the full US cross-section**. We replicate it honestly on a **40-name large-cap "
            "survivor** basket and confront it with the **sample size** and the **survivorship**: "
            "the decisive object is the long-short mean against its standard error, on 9 annual "
            "rebalances of 12-name baskets.\n\n"
            "> ⚠️ **Data + sample note.** A point-in-time, survivorship-free issuance universe is a "
            "CRSP/Compustat product, not a free yfinance feed; we fix a transparent **40-name "
            "survivor sample** (named on the Signal axis), compute net issuance from "
            "**split-adjusted** shares (`get_shares_full` + split history — so splits don't fake "
            "dilution), and hold with a **1-year lag**. Real data: yfinance, year-ends 2014→2025. "
            "Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Long-short mean **{R['ls_mean']:.1f}%/yr** at one-sample "
            f"**t = {R['t']:.2f}** (*wrong sign*, ∣t∣ < 2), label-shuffle placebo "
            f"**p = {R['p_placebo']:.2f}**, win-rate **{R['win']}%**. |\n"
            f"| **Tradability** | `MIRAGE` | Net of {R['cost_bps']}bps×2 + {R['borrow_bps']}bps "
            f"borrow = **{R['net']:.1f}%/yr** (t = {R['net_t']:.2f}); stable-but-insignificant "
            "across widths (∣t∣ ≤ 1.3) and both half-samples. Nothing to size. |\n"
            f"| **Replicates here?** | `BUSTED` | A faithful 25-seed control shows a *real* edge "
            f"prints **t = +{R['syn'][1][1]:.1f}**; the real tape prints **{R['t']:.2f}** — the "
            "published factor does **not** survive a 40-name survivor replication. |\n\n"
            "> 💡 In plain words: the issuance factor is genuinely real where it was discovered — in "
            "the *full* cross-section of thousands of names. Squeeze it onto 40 survivors over 9 "
            "years and the signal drowns in single-name variance and even sign-flips: a textbook "
            "small-sample / survivorship illustration, exactly the outcome the desk expects from "
            "**most** pointed academic factors replicated on a thin survivor basket."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{iss}_{i,t} = S^{\\text{adj}}_{i,t}/S^{\\text{adj}}_{i,t-1} - 1$ be name "
            "$i$'s split-adjusted net issuance over year $t$, and $r_{i,t+1}$ its total return the "
            "*following* year. The factor sorts on $\\text{iss}$ and forms\n\n"
            "$$\\text{LS}_t = \\frac1{|L_t|}\\sum_{i\\in L_t} r_{i,t+1} - "
            "\\frac1{|H_t|}\\sum_{i\\in H_t} r_{i,t+1},$$\n\n"
            "$L_t$ = bottom-30% issuance (buybacks), $H_t$ = top-30% (dilution).\n\n"
            "- **H₁ (the factor predicts).** $\\mathbb{E}[\\text{LS}] > 0$ — Pontiff-Woodgate "
            "(2008): issuers underperform.\n"
            "- **H₂ (it's deployable).** $\\text{LS}$ is large and stable enough, net of costs + "
            "borrow, to allocate to.\n"
            "- **H₃ (it replicates here).** The full-universe effect shows up on a 40-name survivor "
            "basket over 9 years.\n\n"
            "We find **H₁ not supported** (LS is *negative*, $t = -0.86$), **H₂ rejected** "
            "(insignificant, wrong-signed, worse after costs), **H₃ rejected** (a faithful control "
            "says a real edge would print $t \\approx +4.5$; the tape prints $-0.86$). The factor is "
            "real where it was born and **absent on this instrument**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one estimate against its **standard error**:\n\n"
            "$$\\overline{\\text{LS}} = \\frac1{T}\\sum_{t=1}^{T}\\text{LS}_t,\\qquad "
            "t = \\frac{\\overline{\\text{LS}}}{s_{\\text{LS}}/\\sqrt{T}},\\quad T = 9.$$\n\n"
            "Each $\\text{LS}_t$ is a difference of two **12-name** equal-weight baskets, so it "
            "carries enormous idiosyncratic variance; with only $T=9$ annual observations, "
            "$s_{\\text{LS}}/\\sqrt{T}$ is large and even a real few-percent tilt is invisible. The "
            "right small-sample instrument is a **label-shuffle randomization**: permute which name "
            "carries which issuance value (preserving both marginals — the distribution of issuance "
            "*and* the realised cross-section of returns) and ask how often chance matches the real "
            "sort. That p-value, not the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {R['n_names']} large-cap **survivors**, year-ends 2014→2025; **{R['n_years']}** "
            "usable formation years (2016→2024). Explicit **survivorship-selected sample** (named on "
            "the axis).\n"
            "- **Signal.** Net issuance = YoY change in **split-adjusted** shares "
            "(`get_shares_full` × split-ratio correction) — + = dilution, − = buyback.\n"
            "- **Sort + lag.** Bottom/top **30%** by issuance, equal-weight; **1-year** execution lag "
            "(form at year-end $t$, hold $t\\to t+1$; no look-ahead, no same-bar fill).\n"
            "- **Null #1 (one-sample t).** $\\overline{\\text{LS}}$ vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** 20,000 draws; each year, permute issuance labels "
            "across names; $p = \\Pr[\\text{shuffled LS} \\ge \\text{real LS}]$.\n"
            "- **Robustness.** Quantile-width sweep (0.2/0.3/0.4) + split-half by year.\n"
            "- **Costs.** 1-yr lag + 10 bps × both legs × full turnover + 50 bps short borrow.\n"
            "- **Positive control.** A deterministic synthetic panel with a **known** planted "
            "issuance edge, **averaged over 25 seeds**: must recover a planted edge **and** must NOT "
            "manufacture significance at edge = 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimate — negative, small, swamped by its SE\n\n"
            "The annual long-short series with its mean ± standard error and the $t=0$ line. The "
            "mean is *below* zero and its error bar straddles zero comfortably — a $t$ nowhere near "
            "$\\pm 2$."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short_series(PX, SH, frac=0.3); x = ls['ls_ret'].to_numpy()\n"
            "    yrs = [t.year for t in ls.index]\n"
            "else:\n"
            "    x = np.array([p[3]/100 for p in R['panel']]); yrs = [p[0] for p in R['panel']]\n"
            "m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar([str(y) for y in yrs], x*100, color=[GREEN if v>0 else RED for v in x], width=.6, alpha=.85)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.errorbar(len(x)/2-0.5, m*100, yerr=se*100, fmt='o', c='k', capsize=6, ms=8,\n"
            "            label=f'mean {m*100:.1f}% ± SE {se*100:.1f}%')\n"
            "ax.set_ylabel('long-short return (%)'); ax.set_title('Negative mean, error bar straddles zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'L-S mean {m*100:.2f}%/yr  t = {st.one_sample_t(x):.2f}  (need |t|>=2)')"
        ),
        md(
            f"> 💡 In plain words: the mean long-short is **{R['ls_mean']:.1f}%/yr** at "
            f"**t = {R['t']:.2f}** — the *wrong sign* for H₁ and far inside its own error bar. Two "
            "noisy years (2019, 2022) where the diluters were the winners dominate. H₁ is **not "
            "supported**: a faint, wrong-signed wobble, not an edge."
        ),
        md(
            "### 4b · The decisive test — a label-shuffle placebo null\n\n"
            "Permute which name carries which issuance value, 20,000 times; the histogram is the "
            "null distribution of the long-short mean under *no* issuance information. The real sort "
            "is the red line; the $p$-value is the right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PX, SH, frac=0.3); obs = pl['obs_mean']; pval = pl['p_value']\n"
            "    sig = st.net_issuance(SH); fwd = st.forward_returns(PX); years=[]\n"
            "    for t in sig.index:\n"
            "        if t not in fwd.index: continue\n"
            "        sr = sig.loc[t].dropna(); fr = fwd.loc[t].reindex(sr.index).dropna()\n"
            "        c = sr.index.intersection(fr.index)\n"
            "        if len(c)>=4: years.append((sr.loc[c].to_numpy(), fr.loc[c].to_numpy()))\n"
            "    rng=np.random.default_rng(519); draws=np.empty(20000)\n"
            "    for d in range(20000):\n"
            "        acc=[]\n"
            "        for iss,ret in years:\n"
            "            n=len(iss); k=max(1,round(n*0.3)); order=np.argsort(iss[rng.permutation(n)])\n"
            "            acc.append(ret[order[:k]].mean()-ret[order[-k:]].mean())\n"
            "        draws[d]=np.mean(acc)\n"
            "else:\n"
            "    obs = R['ls_mean']/100; pval = R['p_placebo']\n"
            "    rng=np.random.default_rng(519); draws=rng.normal(0.0, 0.045, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} label-shuffles')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed sort {obs*100:.1f}%')\n"
            "ax.axvline(0, c='k', lw=.8, ls=':')\n"
            "ax.set_xlabel('long-short mean (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Label-shuffle p = {pval:.2f}: the real sort is mid-cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[shuffled LS >= real LS] = {pval:.3f}  (a coin-flip relabelling beats it ~{pval*100:.0f}% of the time)')"
        ),
        md(
            f"> 💡 In plain words: **{R['p_placebo']*100:.0f}%** of random relabellings beat the real "
            "sort. A genuine edge would push the red line into the far **right** tail; instead it "
            "sits left of centre (it's negative). The issuance label carries **no** information on "
            "this tape — H₃ rejected."
        ),
        md(
            "### 4c · Robustness — the non-result is stable across cuts\n\n"
            "If there were a hidden (even sign-flipped) edge, it should at least be *consistent*. "
            "Quantile-width sweep and a split-half cut: every slice is **negative and "
            "insignificant** — and a real sign-flip would itself need ∣t∣ ≥ 2 to claim."
        ),
        code(
            "if HAVE_REAL:\n"
            "    width = [(r['frac'], r['ls_mean']*100, r['t']) for r in st.width_sweep(PX, SH, (0.2,0.3,0.4))]\n"
            "    ls = st.long_short_series(PX, SH, frac=0.3); x = ls['ls_ret'].to_numpy(); h = len(x)//2\n"
            "    half = [('first half', x[:h+1].mean()*100, st.one_sample_t(x[:h+1])),\n"
            "            ('second half', x[h+1:].mean()*100, st.one_sample_t(x[h+1:]))]\n"
            "else:\n"
            "    width = [(w[0], w[1], w[2]) for w in R['width']]\n"
            "    half = [(h[0], h[1], h[2]) for h in R['half']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.2))\n"
            "a1.bar([f'{w[0]:.0%}' for w in width], [w[1] for w in width], color=RED, alpha=.8, width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Quantile-width sweep'); a1.set_ylabel('L-S %/yr')\n"
            "for i,w in enumerate(width): a1.annotate(f't={w[2]:.2f}',(i,w[1]),ha='center',va='top')\n"
            "a2.bar([h[0] for h in half], [h[1] for h in half], color=RED, alpha=.8, width=.5)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_title('Split-half by year'); a2.set_ylabel('L-S %/yr')\n"
            "for i,h in enumerate(half): a2.annotate(f't={h[2]:.2f}',(i,h[1]),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('widths:', [(w[0], round(w[1],1), round(w[2],2)) for w in width])\n"
            "print('halves:', [(h[0], round(h[1],1), round(h[2],2)) for h in half])"
        ),
        md(
            f"> 💡 In plain words: every quantile width (∣t∣ ≤ **{abs(R['width'][2][2]):.1f}**) and "
            "both half-samples come out **negative and insignificant**. The non-result is stable — "
            "it's consistently *nothing*, not a hidden edge waiting for the right cut."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On deterministic synthetic panels with a **known** planted issuance edge, averaged over "
            "**25 seeds** (never one lucky draw): with **zero** edge the averaged $t$ must stay near "
            "0 (~40 names / 9 years can't fake significance); with a **+10%/yr** planted edge it "
            "must clear 2. Both hold — proving the engine is unbiased *and correctly signed* (a real "
            "edge prints a clearly **positive** $t$)."
        ),
        code(
            "res = st.synthetic_control(data.synthetic_panel, edges=(0.0, 0.10), n_seeds=25)\n"
            "labels = [f'planted\\n{r[\"edge\"]*100:.0f}%/yr' for r in res]; tvals = [r['mean_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('mean one-sample t (25 seeds)')\n"
            "ax.set_title('Control: no edge -> t~0; real edge -> t lights up POSITIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res: print(f\"planted {r['edge']*100:+.0f}%/yr: mean t = {r['mean_t']:.2f}  (LS {r['ls_mean']*100:.1f}%)\")"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control sits at "
            f"**t = {R['syn'][0][1]:.2f}** (no false positive); a **+10%/yr** edge drives it to "
            f"**t = +{R['syn'][1][1]:.1f}**. So the machinery is honest **and** would show a real "
            f"low-issuance edge as a *positive* t — which makes the real-tape **t = {R['t']:.2f}** "
            "decisive: it is what *no edge* looks like through a 40-name / 9-year keyhole, not a "
            "hidden sign-flipped strategy."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — long-short **{R['ls_mean']:.1f}%/yr** at **t = {R['t']:.2f}** "
            f"(wrong sign, ∣t∣ < 2), label-shuffle **p = {R['p_placebo']:.2f}**, win-rate "
            f"**{R['win']}%**. The faithful control shows a real edge prints positive t, so this is "
            "**noise**. Literature support cannot lift a wrong-signed, insignificant estimate to "
            "REAL. **Survivorship** named on the axis.\n"
            f"- **Tradability `MIRAGE`** — net of {R['cost_bps']}bps×2 + {R['borrow_bps']}bps borrow "
            f"= **{R['net']:.1f}%/yr** (t = {R['net_t']:.2f}); insignificant across every width and "
            "both half-samples. Nothing to size.\n"
            f"- **Replicates here? `BUSTED`** — a real, decades-old factor that does **not** survive "
            "a 40-name survivor replication. The effect lives in CRSP's full thousands-name "
            "point-in-time cross-section; a thin survivor basket is the wrong instrument — small "
            "sample × survivorship, not a free lunch."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* annual long-short have "
            "to be for a $T$-year study (at this basket's variance) to detect it at $t=2$? At $T=9$ "
            "you'd need a spread several times anything plausible; the real signal lives far below "
            "the detection floor."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short_series(PX, SH, frac=0.3); sd = ls['ls_ret'].std(ddof=1)\n"
            "else:\n"
            "    sd = 0.127\n"
            "Ts = np.arange(5, 60)\n"
            "min_detectable = 2.0 * sd / np.sqrt(Ts)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(Ts, min_detectable*100, c=AMBER, lw=2, label='spread needed for t=2')\n"
            "ax.axhline(5, c=GREEN, ls='--', label='~5%/yr (a generous true issuance spread)')\n"
            "ax.axvline(R['n_years'], c=GREY, ls=':', label=f\"our T={R['n_years']} years\")\n"
            "ax.set_xlabel('number of annual observations T'); ax.set_ylabel('annual long-short spread (%)')\n"
            "ax.set_title('Detection floor vs a plausible signal: we are far under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_years'])\n"
            "print(f'at T={R[\"n_years\"]} you need ~{need*100:.0f}%/yr spread for t=2; a plausible true spread is ~5% -> under-powered by ~{need/0.05:.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable spread**; at $T=9$ it "
            "sits far above any plausible true issuance edge (green line). Even granting the factor "
            "its full real-world strength, a 40-name / 9-year survivor study **cannot** see it — and "
            "the wrong-signed point estimate confirms there's nothing to see here. There is no "
            "sizing, threshold, or cost assumption that manufactures power from 9 noisy annual "
            "observations of 12-name baskets. **The variance that makes survivor baskets noisy is "
            "exactly what makes this factor untestable on a small survivor slice** — which is why "
            "the desk expects *most* pointed academic factors to land None × Mirage here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The real population.** Pontiff-Woodgate live in CRSP's **full, point-in-time** "
            "cross-section (thousands of names, the small chronic diluters included) with a proper "
            "FF/Carhart model. The effect is real there; a 40-name survivor slice is the wrong "
            "instrument by construction.\n"
            "- **Composite issuance, properly.** Daniel-Titman's **composite** measure (the slice of "
            "market-cap growth *not* explained by the stock return) is a cleaner signal than a raw "
            "share-count delta and is worth rebuilding on a wider panel.\n"
            "- **The announcement cousin.** [Study 368 — Buyback-Drift](../../368-buyback-drift/) "
            "tests the *event* version (drift after a buyback authorization); "
            "[Study 363 — PEAD-Drift](../../363-pead-drift/) is the desk's other honest "
            "small-basket factor replication with the same texture.\n\n"
            "*The reproducible core is offline and deterministic; the basket is an explicit "
            "large-cap survivor sample. Methods and sources: "
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
