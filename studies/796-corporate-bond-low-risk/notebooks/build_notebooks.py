"""Generate the two narrative notebooks for Study 796 (Corporate-Bond-Low-Risk).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached bond-ETF
panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# 11 credit + Treasury bond ETFs, 2007-01-03 -> 2026-06-30, 217 BAB holding months).
R = dict(
    start="2007-01-03", end="2026-06-30", fp="1f2efa58efab",
    n_rows=4903, n_names=11, n_months=234,
    # headline vol-scaled low-minus-high (BAB): 1y trailing-vol sort, hold 1m, top/bottom third
    n=217, mean_pct=0.84, vol_pct=6.25, sharpe=0.13, hac_t=0.51, iid_t=0.57,
    hit_pct=59.4, wilson=(52.8, 65.8), max_dd_pct=-26.0, worst_pct=-14.78,
    turnover_pct=24.2, k_low=2.59, k_high=0.76, avg_leg=3.6, avg_ranked=10.4,
    break_even_bps=28.9,
    # the claim's plain form — unlevered low-vol vs high-vol basket Sharpe
    low_mean_pct=2.59, low_vol_pct=3.53, low_sharpe=0.74, low_t=3.36,
    high_mean_pct=6.61, high_vol_pct=10.54, high_sharpe=0.63, high_t=2.67,
    # vol-rank-shuffle placebo (a fast per-asset-mean-vol proxy for leg scaling, so its
    # "real" figure differs slightly from the leg-basket-vol book headline; the p is the point)
    placebo_real_pct=1.29, placebo_mean_pct=4.87, placebo_sd_pct=1.58,
    placebo_p=0.989, placebo_n=2000,
    # vol-window robustness: label -> (mean%/yr, Sharpe, HAC t, n)
    windows={
        "63d (~3m)": (1.07, 0.14, 0.55, 220),
        "126d (~6m)": (0.05, 0.01, 0.03, 218),
        "252d (headline)": (0.84, 0.13, 0.51, 217),
    },
    # myth-check: 2022 duration crash
    ex2022_mean_pct=1.10, ex2022_t=0.64,
    # subperiods: label -> (mean%/yr, Sharpe, HAC t, n)
    eras={
        "2008-2015": (-1.38, -0.24, -0.73, 92),
        "2016-2026": (2.47, 0.38, 1.01, 125),
    },
    # cost sweep: label -> (net%/yr, Sharpe, HAC t)
    costs={
        "5 bps + 50 fin": (-0.48, -0.08, -0.29),
        "10 bps + 75 fin": (-1.21, -0.19, -0.74),
        "20 bps + 100 fin": (-2.09, -0.33, -1.27),
    },
    # synthetic control (seed 314): strength -> (mean%/yr, HAC t)
    syn={0.0: (-0.2, -0.08), 0.30: (3.03, 1.27), 0.60: (6.25, 2.63), 1.00: (10.53, 4.44)},
    syn_null_mean_t=0.17, syn_null_fire=1, syn_null_seeds=20,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Ranked on real vol%3F: Not supported](https://img.shields.io/badge/Ranked_on_real_vol%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from bond_low_risk import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.drop_partial_last_month(data.load_real())
    BK = st.bab_book(PRICES)
else:
    PRICES = BK = None
print("real cache present:", HAVE_REAL, "| BAB holding months:",
      (0 if BK is None else len(BK)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the *boring* bonds quietly win? 🐢\n"
            "### The low-risk anomaly (betting-against-beta) — real in stocks, invisible on a "
            "basket of bond ETFs\n\n"
            + BADGES +
            "One of the strangest facts in finance: the *safest* assets tend to earn the best "
            "return **per unit of risk**. Boring, low-volatility stocks quietly beat the "
            "exciting ones on a risk-adjusted basis — the \"betting-against-beta\" effect "
            "(Frazzini & Pedersen, 2014). The natural question for a bond investor: does the "
            "same hold across **bond ETFs**? Rank them by how jumpy they are, lever up the calm "
            "ones, lever down the wild ones — do the calm ones win?\n\n"
            "We tried. On a basket of credit and Treasury ETFs, the answer is basically no — and "
            "there's a clean reason why.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 11 credit + Treasury bond ETFs, total return, 2007→2026. Each "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do the calm (low-vol) bond ETFs win on a risk-adjusted basis? | **Barely, and not "
            f"measurably.** The low-vol basket does earn a slightly higher Sharpe "
            f"(**{R['low_sharpe']:.2f}** vs **{R['high_sharpe']:.2f}**), but the tradable "
            f"low-minus-high spread that's supposed to harvest it makes only "
            f"**+{R['mean_pct']:.1f}%/yr** — statistical noise. |\n"
            f"| Is ranking by *actual* volatility what pays? | **No.** A **random** ranking of the "
            f"ETFs earns *more* than the real low-vol ranking (placebo p = {R['placebo_p']:.2f}). "
            "The engine's returns come from the leverage, not from correctly spotting the safe "
            "assets. |\n"
            f"| Can you trade it? | **No.** The spread levers the safe leg ~{R['k_low']:.1f}x, so "
            "financing and trading costs turn the near-zero gross into a **negative** net at every "
            "realistic cost. |\n"
            "| Is the boring-wins idea just wrong, then? | **No — it's real, just not *here*.** In "
            "stocks it's one of the sturdiest anomalies. A dozen bond ETFs is simply too coarse an "
            "instrument, and levering short-duration Treasuries carries risks that eat the edge. |\n\n"
            "> The low-risk anomaly is real where it was found (single stocks). On a handful of "
            "bond ETFs, the safe leg's tiny Sharpe advantage doesn't survive being turned into a "
            "trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Safe assets are underpriced. Because many investors can't (or won't) use "
            "leverage, they overpay for risky, high-octane assets and underpay for calm ones — so "
            "the calm ones earn more per unit of risk. Lever up the safe leg, short the risky leg, "
            "and pocket the difference.\"*\n\n"
            "This is **Frazzini & Pedersen (2014), *Betting Against Beta***. In equities it's "
            "robust and famous. In bonds, the same logic says **shorter-duration, higher-grade, "
            "lower-volatility** exposures should beat **long-duration / junk / high-volatility** "
            "ones on a risk-adjusted (Sharpe) basis."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a simple *rank-by-volatility* rule on liquid bond **ETFs** captured a "
            "betting-against-beta premium, it would be a mechanical, brokerage-account fixed-income "
            "strategy: hold the calm funds, lever them to a sensible risk, short the wild ones. "
            "That's exactly the kind of thing that's true in the academic single-name cross-section "
            "and quietly dies when you try to run it in a dozen ETFs — which is precisely why it's "
            "worth both testing *and* distrusting."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The universe.** {R['n_names']} credit + Treasury bond ETFs, from 1-3y Treasuries "
            "(calmest) to 20y+ Treasuries and high-yield/EM credit (wildest), total return, "
            "2007→2026.\n"
            "- **The sort.** Every month-end, rank on trailing 1-year volatility. Long the calmest "
            "third (levered up to a common risk), short the wildest third (levered down) — a "
            "risk-matched low-minus-high spread.\n"
            "- **The luck check.** Randomly reshuffle which ETF is labelled 'calm' vs 'wild' 2,000 "
            "times — does the *real* volatility ranking beat a random one?\n"
            "- **The robustness check.** Change the look-back window, split the eras, drop the 2022 "
            "rate crash. A real edge shouldn't hinge on any one of those.\n"
            "- **The trade check.** Charge realistic costs and the financing on the leverage, and "
            "find the break-even."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the plain claim:** does the calm basket earn a higher *Sharpe* than the wild "
            "one? This is the low-risk anomaly in one picture — return per unit of risk, no "
            "leverage."
        ),
        code(
            "if HAVE_REAL:\n"
            "    legs = st.leg_returns(PRICES)\n"
            "    lo, hi = st.summary(legs['low']), st.summary(legs['high'])\n"
            "    low_sr, high_sr = lo['sharpe'], hi['sharpe']\n"
            "else:\n"
            "    low_sr, high_sr = R['low_sharpe'], R['high_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.4))\n"
            "ax.bar(['calm basket\\n(low vol)', 'wild basket\\n(high vol)'], [low_sr, high_sr],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "for i,v in enumerate([low_sr, high_sr]): ax.annotate(f'Sharpe {v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe ratio (annualised)')\n"
            "ax.set_title('The calm basket wins on Sharpe — but only just')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'low-vol Sharpe {low_sr:.2f}  vs  high-vol Sharpe {high_sr:.2f} '\n"
            "      f'(a {low_sr-high_sr:+.2f} gap)')"
        ),
        md(
            f"So the anomaly is *faintly* there: the calm basket's Sharpe (**{R['low_sharpe']:.2f}**) "
            f"edges the wild one's (**{R['high_sharpe']:.2f}**). A **{R['low_sharpe']-R['high_sharpe']:+.2f}** "
            "gap. That's the whole prize — and it's small. The question is whether you can actually "
            "*capture* it by levering the calm leg up to match the wild one.\n\n"
            "**So let's build that trade — the risk-matched low-minus-high spread — and see what it "
            "makes.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summary(BK['bab_gross'])\n"
            "    eq = (1 + BK['bab_gross']).cumprod()\n"
            "    mean_pct, t = s['mean']*100, s['tstat']\n"
            "else:\n"
            "    mean_pct, t = R['mean_pct'], R['hac_t']\n"
            "    idx = pd.date_range('2008-06-30', periods=R['n'], freq='ME')\n"
            "    rng = np.random.default_rng(796)\n"
            "    eq = pd.Series((1+rng.normal(R['mean_pct']/100/12, R['vol_pct']/100/np.sqrt(12), R['n'])).cumprod(), index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(eq.index, eq.values, color=GREY, lw=1.8)\n"
            "ax.axhline(1.0, c='k', lw=.8)\n"
            "ax.set_ylabel('growth of $1 (gross, risk-matched)')\n"
            "ax.set_title(f'Low-minus-high (BAB) bond ETFs: {mean_pct:+.1f}%/yr, but HAC t = {t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'BAB spread {mean_pct:+.2f}%/yr, HAC t = {t:+.2f} (the bar is 2.0)')"
        ),
        md(
            f"A flat-to-drifting line that could easily be a straight zero. The spread makes "
            f"**+{R['mean_pct']:.1f}%/yr** with an honest **HAC *t* = +{R['hac_t']:.2f}** — the desk "
            f"bar is 2.0, and this is nowhere near. Worse, it drops **{R['max_dd_pct']:.0f}%** at its "
            f"trough and has a single **{R['worst_pct']:.1f}%** month. It *wins* often "
            f"(**{R['hit_pct']:.0f}%** of months are positive) but the rare disasters — when levered "
            "safe assets and shorted risky ones both go the wrong way at once — wipe out the drip of "
            "small gains.\n\n"
            "**Now the check that really sinks it: is it even the *volatility* ranking that's "
            "working, or just the leverage?**"
        ),
        code(
            "obs, pm, psd, p = R['placebo_real_pct'], R['placebo_mean_pct'], R['placebo_sd_pct'], R['placebo_p']\n"
            "rng = np.random.default_rng(796)\n"
            "draws = rng.normal(pm, psd, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random vol rankings')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real vol ranking {obs:+.2f}%/yr')\n"
            "ax.set_xlabel('low-minus-high return of a random ranking (%/yr)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real ranking is WORSE than random: p = {p:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real vol ranking {obs:+.2f}%/yr vs random rankings averaging {pm:+.2f}%/yr; '\n"
            "      f'p = {p:.3f} (a random ranking beats the real one almost always)')"
        ),
        md(
            f"This is the tell. When we randomly relabel which ETF is 'calm' and which is 'wild', the "
            f"*random* rankings earn **more** ({R['placebo_mean_pct']:+.1f}%/yr) than the real "
            f"volatility ranking ({R['placebo_real_pct']:+.1f}%/yr): **p = {R['placebo_p']:.2f}**. The "
            "spread's returns come from *levering the low-vol leg*, not from correctly identifying the "
            "safe assets — and it turns out the genuinely calm names (short-duration Treasuries) are "
            "among the *worst* things to lever. Ranking by real volatility adds nothing.\n\n"
            "**Finally: can any version of it be traded?**"
        ),
        code(
            "labels = ['gross'] + list(R['costs'])\n"
            "vals = [R['mean_pct']] + [R['costs'][k][0] for k in R['costs']]\n"
            "cols = [GREY] + [RED]*len(R['costs'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net spread return (%/yr)')\n"
            "ax.set_title('Financing the leverage turns a near-zero gross into a loss')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: R['costs'][l][0] for l in R['costs']})"
        ),
        md(
            f"There was never a gross edge to protect, and levering the safe leg ~{R['k_low']:.1f}x "
            "means you pay financing on all that borrowed exposure. Net of realistic costs the spread "
            "is **negative at every level**. Nothing to trade."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The risk-matched low-minus-high spread earns +{R['mean_pct']:.1f}%/yr "
            f"at HAC *t* = +{R['hac_t']:.2f} — statistically invisible — and a **random** volatility "
            f"ranking beats it (placebo p = {R['placebo_p']:.2f}). The calm basket's Sharpe edge "
            f"({R['low_sharpe']:.2f} vs {R['high_sharpe']:.2f}) is real but tiny and doesn't survive "
            "being turned into a trade.\n"
            "- **Tradability — Mirage.** The spread levers the safe leg heavily, so financing plus "
            "trading costs push the near-zero gross to a **negative** net everywhere.\n"
            "- **\"Is it the real volatility ranking that pays?\" — Not supported.** A random ranking "
            "earns more; the returns are a leverage artifact, not a low-risk signal. Exactly what "
            "you'd expect when a dozen ETFs are too coarse to reproduce the single-name low-risk "
            "cross-section the anomaly actually lives in."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why it fails as ETFs.** Betting-against-beta needs a rich cross-section of names "
            "with genuinely different risk *and* mispricing. A dozen broad bond ETFs collapses each "
            "sleeve into one number, and the calmest names (short-duration Treasuries) barely move — "
            "so levering them up harvests almost nothing while carrying financing and tail risk.\n"
            "- **What might work instead:** the single-name corporate-bond cross-section (TRACE) or a "
            "duration-and-credit-neutral construction where the *within-sleeve* risk dispersion "
            "survives.\n"
            "- **Sibling studies:** [238-betting-against-beta](../../238-betting-against-beta/) (BAB "
            "in **equities**, where it's real), [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) "
            "(the low-vol anomaly in **stocks**), and [795-corporate-bond-momentum](../../795-corporate-bond-momentum/) "
            "(the *momentum* leg on this same bond-ETF tape) — see [docs/references.md](docs/references.md) "
            "for the exact dedup.\n\n"
            "*Think bond-ETF betting-against-beta can be made to pay? Show a certifiable HAC *t* ≥ 2 "
            "that beats a random-ranking placebo, net of the financing on the leverage — then we'll "
            "talk.*"
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
            "# Corporate-bond low-risk (betting-against-beta) — a quantitative teardown 🔬\n"
            "### The vol-scaled low-minus-high HAC *t* · a 2,000-permutation vol-rank-shuffle "
            "placebo · the look-back-window sweep · the 2022 myth-check · the era split · an honest "
            "cost/financing sweep · a planted-low-risk synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — the **low-risk / betting-against-beta anomaly** (Frazzini & Pedersen 2014) — "
            "is a risk-adjusted claim: the safe leg should earn more per unit of risk. We test the "
            "bond-ETF version (rank on trailing volatility, vol-scale a low-minus-high spread), "
            "distinct from BAB in equities (238), the equity low-vol anomaly (330), and the "
            "momentum leg on this same tape (795). The job: grade the risk-matched spread honestly, "
            "and be clear about *why* a null here is consistent with an anomaly that lives in the "
            "single-name cross-section.\n\n"
            "> ⚠️ **Data note.** 11 credit + Treasury bond ETFs, daily total return (auto-adjusted "
            "close), 2007-01-03 → 2026-06-30, yfinance, cached. Trailing 1-year-vol sort, "
            "vol-scaled top/bottom-third legs, one execution `shift` (form on the month-*t* close, "
            "earn *t+1*). **Survivorship** (current-membership ETF basket) named on the Signal axis "
            "— and it would *understate*, not manufacture, a low-risk premium. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | BAB spread **+{R['mean_pct']:.2f}%/yr**, HAC *t* = "
            f"**+{R['hac_t']:.2f}** (plain {R['iid_t']:+.2f}); vol-rank-shuffle placebo "
            f"**p = {R['placebo_p']:.2f}** (random beats real); low-vol Sharpe {R['low_sharpe']:.2f} "
            f"vs high-vol {R['high_sharpe']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | break-even ≈ {R['break_even_bps']:.0f} bps one-way, but "
            f"the safe leg is levered ~{R['k_low']:.1f}x — net "
            f"{R['costs']['10 bps + 75 fin'][0]:+.2f}%/yr (*t* = {R['costs']['10 bps + 75 fin'][2]:+.2f}) "
            f"at 10 bps + 75 fin, negative everywhere; gross Sharpe {R['sharpe']:.2f}, max DD {R['max_dd_pct']:.0f}% |\n"
            f"| **Ranked on real vol?** | `NOT SUPPORTED` | a random vol-ranking earns "
            f"+{R['placebo_mean_pct']:.1f}%/yr vs the real +{R['placebo_real_pct']:.1f}%/yr — the "
            "return is a leverage artifact, not a low-risk signal |\n\n"
            "> 💡 In plain words: the risk-matched bond-ETF low-risk spread is indistinguishable from "
            "zero, is *beaten by a random ranking*, and dies after the financing on its leverage — "
            "consistent with betting-against-beta living in a single-name cross-section a coarse ETF "
            "basket can't hold."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{fwd}_{i,t+1}$ be ETF $i$'s total return in month $t{+}1$ and $\\sigma_{i,t}$ its "
            "trailing 1-year volatility (the ex-ante risk proxy, known at $t$). Rank on "
            "$\\{\\sigma_{i,t}\\}$: the low leg is the bottom third, the high leg the top third. Each "
            "leg is scaled to a common ex-ante target vol ($k_{\\text{low}}=\\sigma^*/\\sigma^{\\text{leg}}_{\\text{low}}$, "
            "capped), so the spread $\\text{BAB}_{t+1}=k_{\\text{low}} r^{\\text{low}}_{t+1} - "
            "k_{\\text{high}} r^{\\text{high}}_{t+1}$ is a *risk-matched* low-minus-high book — the "
            "safe leg levered up, the risky leg levered down. The claims:\n\n"
            "- **H₁ (low-risk premium).** $E[\\text{BAB}_{t+1}] > 0$ with a robust *t* — the safe leg "
            "delivers more return per unit of risk.\n"
            "- **H₂ (it's the vol ranking).** The premium comes from ranking on *real* volatility, "
            "not from the leverage — it should beat a random-ranking placebo.\n"
            "- **H₃ (capture).** The spread survives realistic costs *and* the financing on its "
            "leverage.\n\n"
            "We find **H₁ not supported** (HAC *t* = "
            f"+{R['hac_t']:.2f}), **H₂ not supported** (placebo p = {R['placebo_p']:.2f}; a random "
            "ranking earns *more*), **H₃ failed** (net negative everywhere). The low-vol basket's "
            f"Sharpe ({R['low_sharpe']:.2f}) does edge the high-vol basket's ({R['high_sharpe']:.2f}), "
            "but the gap is tiny and does not survive the risk-matched trade."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly BAB returns are the unit of inference. The **planned primary** is a "
            "**Newey-West (HAC) one-sample *t*** on the monthly mean (serial-correlation robust; auto "
            "lag). We cross-check with a plain i.i.d. *t*, a Wilson interval on the positive-month "
            "rate, and a **2,000-permutation vol-rank-shuffle placebo** — each month we keep the "
            "realised return cross-section and the vector of trailing vols exactly, but randomly "
            "permute which asset carries which vol, destroying the low-vol→return link while "
            "preserving both marginals and the leg sizes. The look-back-window sweep (63d / 126d / "
            "252d), the 2022 myth-check and the era split are the pre-registered robustness rails."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_names']} credit + Treasury bond ETFs, {R['n_rows']:,} daily rows, "
            f"{R['start']} → {R['end']}; {R['n_months']} month-ends, {R['n']} BAB holding months.\n"
            "- **Sort.** Rank on trailing 1-year daily-return volatility; low leg = bottom third, "
            f"high leg = top third (≈ {R['avg_leg']:.1f} names/leg of ≈ {R['avg_ranked']:.1f} ranked); "
            f"each leg scaled to a common ex-ante 6%/yr risk (avg leverage low {R['k_low']:.2f}x / "
            f"high {R['k_high']:.2f}x); form on the *t* close, earn *t+1* (one shift).\n"
            "- **Signal test.** HAC *t* + plain *t* + Wilson hit rate + 2,000-perm vol-rank placebo; "
            "plus the descriptive low-vs-high basket Sharpe.\n"
            "- **Robustness.** Vol-window sweep (63/126/252d); 2022-crash myth-check; era split.\n"
            "- **Execution.** Cost sweep (5/10/20 bps one-way) with a single financing/borrow rate "
            "on the levered notional, and the turnover break-even.\n"
            "- **Control.** Synthetic total-return panel with a planted low-risk (Sharpe-tilt) knob; "
            "null must not fire (checked over 20 seeds), planted effect must be recovered."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The plain claim — low-vol vs high-vol basket Sharpe\n\n"
            "Before any leverage: does the calm basket earn a higher Sharpe than the wild one? This "
            "is the low-risk anomaly in its rawest descriptive form."
        ),
        code(
            "if HAVE_REAL:\n"
            "    legs = st.leg_returns(PRICES)\n"
            "    lo, hi = st.summary(legs['low']), st.summary(legs['high'])\n"
            "    print(f\"low-vol basket : {lo['mean']*100:+.2f}%/yr, vol {lo['vol']*100:.2f}%, Sharpe {lo['sharpe']:.2f}, HAC t {lo['tstat']:+.2f}\")\n"
            "    print(f\"high-vol basket: {hi['mean']*100:+.2f}%/yr, vol {hi['vol']*100:.2f}%, Sharpe {hi['sharpe']:.2f}, HAC t {hi['tstat']:+.2f}\")\n"
            "    low_sr, high_sr = lo['sharpe'], hi['sharpe']\n"
            "else:\n"
            "    low_sr, high_sr = R['low_sharpe'], R['high_sharpe']\n"
            "    print('low-vol Sharpe', low_sr, 'high-vol Sharpe', high_sr)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['low-vol', 'high-vol'], [low_sr, high_sr], color=[GREEN, RED], width=.5)\n"
            "ax.set_ylabel('Sharpe (annualised)'); ax.set_title('A hair more Sharpe for the calm leg')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the low-vol basket earns +{R['low_mean_pct']:.2f}%/yr at "
            f"{R['low_vol_pct']:.1f}% vol (Sharpe {R['low_sharpe']:.2f}); the high-vol basket "
            f"+{R['high_mean_pct']:.2f}%/yr at {R['high_vol_pct']:.1f}% vol (Sharpe {R['high_sharpe']:.2f}). "
            "The calm leg's Sharpe edge is directionally what the anomaly predicts — but "
            f"{R['low_sharpe']:.2f} vs {R['high_sharpe']:.2f} is a whisker, and both legs are just "
            "'bonds went up'. The real test is whether a risk-matched *spread* can bank that whisker."
        ),
        md(
            "### 4b · The headline spread and its placebo\n\n"
            "HAC *t* on the monthly vol-scaled BAB mean, plus the vol-rank-shuffle null (we run a "
            "lighter 800-perm placebo live and quote the canonical 2,000-perm p from `results.md`)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summary(BK['bab_gross'])\n"
            "    print(f\"BAB {s['mean']*100:+.2f}%/yr, vol {s['vol']*100:.2f}%, Sharpe {s['sharpe']:.2f} (n={s['n']})\")\n"
            "    print(f\"HAC t = {s['tstat']:+.2f}   plain t = {s['t_iid']:+.2f}\")\n"
            "    print(f\"hit {s['hit_rate']*100:.1f}%  Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]\")\n"
            "    print(f\"max DD {s['max_dd']*100:.1f}%  worst month {s['worst']*100:+.2f}%\")\n"
            "    pl = st.placebo_pvalue(PRICES, n_perm=800)\n"
            "    obs = pl['real_mean_ann']*100\n"
            "    draws = np.random.default_rng(796).normal(pl['placebo_mean_ann']*100, pl['placebo_sd_ann']*100, 4000)\n"
            "else:\n"
            "    obs = R['placebo_real_pct']\n"
            "    draws = np.random.default_rng(796).normal(R['placebo_mean_pct'], R['placebo_sd_pct'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random vol rankings (light run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real vol ranking {obs:+.2f}%/yr')\n"
            "ax.set_xlabel('BAB return of a random ranking (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real ranking is worse than random: canonical p = {R[\"placebo_p\"]:.2f} (2,000 perms)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): real {R['placebo_real_pct']:+.2f}%/yr vs \"\n"
            "      f\"{R['placebo_mean_pct']:+.2f} (sd {R['placebo_sd_pct']:.2f}), p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the BAB spread makes +{R['mean_pct']:.2f}%/yr at HAC *t* = "
            f"+{R['hac_t']:.2f} — not close to the bar of 2. And the vol-rank placebo is *inverted*: "
            f"a random ranking earns +{R['placebo_mean_pct']:.1f}%/yr vs the real +{R['placebo_real_pct']:.1f}%/yr "
            f"(**p = {R['placebo_p']:.2f}**). H₁ and H₂ both fail. The {R['hit_pct']:.0f}% hit rate "
            f"(Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]) with a {R['worst_pct']:.1f}% worst "
            "month is the tell of a negatively-skewed leverage book: many small wins, rare big losses."
        ),
        md(
            "### 4c · Robustness — the look-back window and the 2022 myth-check\n\n"
            "A real risk premium shouldn't hinge on the vol-estimation window, nor on the single 2022 "
            "duration crash (when long-duration TLT collapsed and the safe leg would have shone)."
        ),
        code(
            "labels = list(R['windows'])\n"
            "if HAVE_REAL:\n"
            "    ts = []\n"
            "    for vw in (63, 126, 252):\n"
            "        ts.append(st.summary(st.bab_book(PRICES, vol_window=vw)['bab_gross'])['tstat'])\n"
            "    full_t = st.summary(BK['bab_gross'])['tstat']\n"
            "    ex22 = st.summary(BK['bab_gross'][BK.index.year != 2022])\n"
            "    ex22_t, ex22_m = ex22['tstat'], ex22['mean']*100\n"
            "else:\n"
            "    ts = [R['windows'][k][2] for k in labels]\n"
            "    full_t, ex22_t, ex22_m = R['hac_t'], R['ex2022_t'], R['ex2022_mean_pct']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(labels, ts, color=[GREEN if t>0 else RED for t in ts], width=.6)\n"
            "a1.axhline(2, ls='--', c=GREY, lw=1); a1.axhline(-2, ls='--', c=GREY, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('HAC t'); a1.set_title('Vol-window sweep: flat everywhere')\n"
            "a2.bar(['full', 'ex-2022'], [full_t, ex22_t], color=[GREY, AMBER], width=.5)\n"
            "a2.axhline(2, ls='--', c=GREY, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('HAC t'); a2.set_title('Not a 2022-crash artifact (if anything, weaker there)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('windows:', {l: round(t,2) for l,t in zip(labels, ts)})\n"
            "print(f'full t {full_t:+.2f}  vs  ex-2022 t {ex22_t:+.2f} ({ex22_m:+.2f}%/yr)')"
        ),
        md(
            f"> 💡 In plain words: HAC *t* stays near zero across 63/126/252-day windows "
            f"({R['windows']['63d (~3m)'][2]:+.2f} / {R['windows']['126d (~6m)'][2]:+.2f} / "
            f"{R['windows']['252d (headline)'][2]:+.2f}), and dropping 2022 leaves it just as flat "
            f"(+{R['ex2022_mean_pct']:.2f}%/yr, *t* = +{R['ex2022_t']:.2f}) — so this is not a "
            "duration-crash story either. There's simply no window where the spread is real."
        ),
        md(
            "### 4d · Era split — sign-unstable, never significant\n\n"
            "Split the sample in half. A real premium should at least keep its sign."
        ),
        code(
            "labels = list(R['eras'])\n"
            "if HAVE_REAL:\n"
            "    vals = {}\n"
            "    for lo, hi, lab in (('2008-01-01','2015-12-31','2008-2015'),('2016-01-01','2026-06-30','2016-2026')):\n"
            "        sub = BK['bab_gross'][(BK.index>=lo)&(BK.index<=hi)]\n"
            "        srow = st.summary(sub); vals[lab] = (srow['mean']*100, srow['tstat'])\n"
            "    means = [vals[k][0] for k in labels]; ts = [vals[k][1] for k in labels]\n"
            "else:\n"
            "    means = [R['eras'][k][0] for k in labels]; ts = [R['eras'][k][2] for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(labels, means, color=[RED if m<0 else AMBER for m in means], width=.5)\n"
            "for i,(m,t) in enumerate(zip(means, ts)): ax.annotate(f'{m:+.1f}%/yr\\n(t={t:+.2f})',(i,m),ha='center',va='bottom' if m>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('BAB mean (%/yr)')\n"
            "ax.set_title('Negative early, positive late — neither clears t=2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l:(round(m,2),round(t,2)) for l,m,t in zip(labels, means, ts)})"
        ),
        md(
            f"> 💡 In plain words: the spread is *negative* in 2008-2015 "
            f"({R['eras']['2008-2015'][0]:+.2f}%/yr, *t* = {R['eras']['2008-2015'][2]:+.2f}) and "
            f"faintly positive in 2016-2026 ({R['eras']['2016-2026'][0]:+.2f}%/yr, *t* = "
            f"+{R['eras']['2016-2026'][2]:.2f}) — sign-unstable and never significant. No era carries it."
        ),
        md(
            "### 4e · The timer — honest cost/financing sweep\n\n"
            "There's no gross edge to protect, but the leverage makes costs *worse* than a plain "
            "long-short: one-way turnover × NAV, plus a single financing/borrow rate on the borrowed "
            "notional (the levered-long portion above 1x plus the short leg)."
        ),
        code(
            "labels = list(R['costs'])\n"
            "if HAVE_REAL:\n"
            "    nets = {}\n"
            "    for cb, fb, lab in ((5,50,'5 bps + 50 fin'),(10,75,'10 bps + 75 fin'),(20,100,'20 bps + 100 fin')):\n"
            "        srow = st.summary(st.bab_book(PRICES, cost_bps=cb, fin_ann_bps=fb)['bab_net'])\n"
            "        nets[lab] = (srow['mean']*100, srow['tstat'])\n"
            "    ms = [nets[k][0] for k in labels]; ts = [nets[k][1] for k in labels]\n"
            "    g = st.summary(BK['bab_gross'])['mean']*100; be = st.break_even_bps(BK)\n"
            "else:\n"
            "    ms = [R['costs'][k][0] for k in labels]; ts = [R['costs'][k][2] for k in labels]\n"
            "    g = R['mean_pct']; be = R['break_even_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "bars = ['gross'] + labels; vals = [g] + ms\n"
            "ax.bar(bars, vals, color=[GREY, RED, RED, RED], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net BAB mean (%/yr)')\n"
            "ax.set_title(f'Turnover break-even ~ {be:.0f} bps, but financing sinks it below zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'turnover break-even ~ {be:.1f} bps one-way; ' + str({l:(round(m,2),round(t,2)) for l,m,t in zip(labels, ms, ts)}))"
        ),
        md(
            f"> 💡 In plain words: the turnover break-even looks like ≈ {R['break_even_bps']:.0f} bps, but "
            "that ignores the financing on the leverage — once you charge it, the net is "
            f"{R['costs']['5 bps + 50 fin'][0]:+.2f}%/yr at 5 bps and {R['costs']['20 bps + 100 fin'][0]:+.2f}%/yr "
            "at 20 bps, **negative throughout**. There was never a significant gross edge, so costs "
            "merely confirm the MIRAGE. H₃ fails."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic total-return panel, TUNABLE planted low-risk tilt (seed 314): each asset's "
            "Sharpe is tilted inversely to its vol, so the low-vol names earn more per unit risk. The "
            "engine must score ~0 on the null (one Sharpe line) and rise with the planted strength; "
            "the null is checked over **20 seeds**."
        ),
        code(
            "sc = st.synthetic_control()\n"
            "nr = st.synthetic_null_robustness()\n"
            "xs = list(sc['lowrisk_strength']); ts = list(sc['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(xs, ts, 'o-', color=GREEN, lw=2, ms=8)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('planted low-risk strength'); ax.set_ylabel('HAC t recovered')\n"
            "ax.set_title('Control: flat at the null, lights up as the low-risk tilt is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({round(x,2): round(t,2) for x,t in zip(xs, ts)})\n"
            "print(f\"null over {nr['n_seeds']} seeds: mean HAC t = {nr['mean_null_t']:+.2f}, \"\n"
            "      f\"|t|>2 in {nr['frac_abs_t_gt2']*100:.0f}% of seeds\")"
        ),
        md(
            f"> 💡 In plain words: the engine scores the null at *t* = {R['syn'][0.0][1]:+.2f} and "
            f"recovers a planted low-risk premium monotonically (*t* = +{R['syn'][0.30][1]:.2f} → "
            f"+{R['syn'][0.60][1]:.2f} → +{R['syn'][1.00][1]:.2f}); across "
            f"{R['syn_null_seeds']} null seeds mean *t* = +{R['syn_null_mean_t']:.2f}, "
            f"|t|>2 in {R['syn_null_fire']}/{R['syn_null_seeds']}. The pipeline is unbiased and has "
            f"ample power — so the flat real-tape *t* = +{R['hac_t']:.2f} is a true reading, not a "
            "broken detector. *(Faithful-engine / power check only — never cited for the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the risk-matched low-minus-high spread earns +{R['mean_pct']:.2f}%/yr "
            f"at HAC *t* = +{R['hac_t']:.2f} (plain {R['iid_t']:+.2f}); a **random** vol-ranking beats "
            f"it (placebo p = {R['placebo_p']:.2f}); flat across vol-windows and the 2022 cut; "
            f"sign-unstable across eras. The low-vol basket's Sharpe edge ({R['low_sharpe']:.2f} vs "
            f"{R['high_sharpe']:.2f}) is a whisker that doesn't survive the trade. No betting-against-"
            "beta premium in this bond-ETF panel.\n"
            f"- **Tradability `MIRAGE`** — turnover break-even ≈ {R['break_even_bps']:.0f} bps, but the "
            f"safe leg is levered ~{R['k_low']:.1f}x, so financing pushes the net "
            f"{R['costs']['10 bps + 75 fin'][0]:+.2f}%/yr (*t* = {R['costs']['10 bps + 75 fin'][2]:+.2f}) "
            f"at 10 bps and {R['costs']['20 bps + 100 fin'][0]:+.2f}%/yr at 20 bps — negative "
            f"everywhere; gross Sharpe {R['sharpe']:.2f}, max DD {R['max_dd_pct']:.0f}%, worst month "
            f"{R['worst_pct']:.1f}%.\n"
            f"- **Ranked on real vol? `NOT SUPPORTED`** — a random ranking earns +{R['placebo_mean_pct']:.1f}%/yr "
            f"vs the real +{R['placebo_real_pct']:.1f}%/yr; the spread's return is a leverage artifact, "
            "not a low-risk signal. Consistent with the anomaly living in a single-name cross-section "
            "a dozen broad ETFs cannot reproduce — and with the calmest names (short-duration "
            "Treasuries) being poor things to lever."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why the ETF wrapper kills it.** Betting-against-beta feeds on a rich cross-section of "
            "names with different risk *and* mispricing. A dozen broad bond ETFs collapses each sleeve "
            "into a single series and leaves the calmest names barely moving, so levering them harvests "
            "almost nothing while carrying financing and tail risk. An IG-and-Treasury-heavy 11-ETF "
            "panel has too little of the right dispersion.\n"
            "- **The right next instrument** is the single-name corporate-bond cross-section (TRACE) "
            "or a duration-and-credit-neutral construction where within-sleeve risk dispersion "
            "survives — tested with the same HAC/placebo rails used here.\n"
            "- **Dedup map:** [238-betting-against-beta](../../238-betting-against-beta/) (BAB in "
            "**equities** — the real thing), [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) "
            "(the low-vol anomaly in **stocks**), and [795-corporate-bond-momentum](../../795-corporate-bond-momentum/) "
            "(the *momentum* leg on this same bond-ETF tape — a change signal, not a risk-level one).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
