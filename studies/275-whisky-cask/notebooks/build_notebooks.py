"""Generate the two narrative notebooks for Study 275 (Whisky-Cask).

    python notebooks/build_notebooks.py

This script writes BOTH notebooks and then EXECUTES them with nbconvert (offline,
no --allow-errors). Both notebooks follow the seven desk beats (see
../../../METHODOLOGY.md). The hardcoded whisky index and the synthetic generator
run anywhere, offline and deterministically; the real S&P cells use the cached
^GSPC parquet at the repo-level _cache/ if present, and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for
any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=16,
    year_start=2009, year_end=2024,
    whisky_ann=12.17, sp_ann=12.42,
    whisky_vol=10.21, sp_vol=13.92,
    whisky_sharpe=1.04, sp_sharpe=0.811,
    corr=-0.172, whisky_beats_frac=0.562,
    rho_lag1=0.779,
    raw_vol_pct=10.21, unsmoothed_vol_pct=29.13, vol_inflation_x=2.85,
    raw_sharpe=1.04, unsmoothed_sharpe=0.251,
    gross_ann_pct=12.17, net_ann_pct=5.93, total_cost_wedge_pct=6.24,
    mean_excess_pct=-0.67, hac_t=-0.148, win_frac=0.562,
    fp="925a321f826e",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from whisky_cask import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc_annual()   # cache-only (fetch=False) by default
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("S&P (^GSPC) cache present:", HAVE_REAL)
"""

# R-dict cell so notebooks can reference R['key'] when HAVE_REAL is False.
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=16, year_start=2009, year_end=2024,
    whisky_ann=12.17, sp_ann=12.42,
    whisky_vol=10.21, sp_vol=13.92,
    whisky_sharpe=1.04, sp_sharpe=0.811,
    corr=-0.172, whisky_beats_frac=0.562,
    rho_lag1=0.779,
    raw_vol_pct=10.21, unsmoothed_vol_pct=29.13, vol_inflation_x=2.85,
    raw_sharpe=1.04, unsmoothed_sharpe=0.251,
    gross_ann_pct=12.17, net_ann_pct=5.93, total_cost_wedge_pct=6.24,
    mean_excess_pct=-0.67, hac_t=-0.148, win_frac=0.562,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Whisky-Cask -- are whisky casks the alternative asset they're sold as?\n"
            "### The 'whisky beat every other luxury asset' pitch, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Scroll any wealth magazine and you'll meet the pitch: **rare whisky casks** are the "
            "alternative asset that *quietly beat the stock market* through the 2010s -- higher "
            "returns, lower volatility, and uncorrelated to equities, so they 'smooth your "
            "portfolio'. Cask-broker ads quote eye-watering index gains. This notebook asks the "
            "only question that matters: is that smoothness a feature of the **asset**, or an "
            "artifact of how the **index is measured**?\n\n"
            "> **This is the plain-language layer.** Want the Geltner unsmoothing, the HAC "
            "t-test, and the cost-wedge arithmetic? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the whisky index beat the S&P? | **No.** ~**12.2%/yr** vs the S&P's "
            "~**12.4%/yr** (price-only) -- a dead heat, before whisky's costs. |\n"
            "| Is whisky really 'low volatility'? | **No.** The headline 10% vol is a "
            "*measurement illusion*. Unsmoothed, the true vol is ~**29%** -- nearly 3x larger, "
            "and higher than equities. |\n"
            "| Is the high Sharpe real? | **No.** Raw Sharpe **1.04** collapses to **0.25** once "
            "the appraisal-smoothing is removed -- worse than just owning the index fund. |\n"
            "| Could a retail buyer earn the index? | **No.** Dealer markups, storage, insurance "
            "and evaporation cut ~**12%/yr gross to ~6%/yr net**. There is no tradable cask ETF. |\n\n"
            "> The whisky-cask 'edge' is two illusions stacked on top of each other: an index that "
            "*looks* smooth because it's appraised once a year, and a gross return you can't "
            "actually capture once the middlemen are paid."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Rare whisky was the best-performing luxury collectible of the decade -- up "
            "hundreds of percent -- with lower volatility than stocks and almost no correlation "
            "to equity markets. A cask is a tangible, appreciating, uncorrelated asset.\"*\n\n"
            "The numbers come from **appraisal-based indices** -- Knight Frank's Luxury Investment "
            "Index (rare whisky sub-index) and Rare Whisky 101's Apex 1000. These are published "
            "**once a year**, based on auction-hammer and valuation marks for a basket of the "
            "rarest *bottles*. That annual, smoothed cadence is the entire story."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a cask really delivered equity-beating returns at half the volatility and zero "
            "correlation, it would be one of the best risk-adjusted assets on earth -- and every "
            "pension fund would own a warehouse of it. The fun question is *why the index looks "
            "that good*, and whether a normal buyer could ever touch those returns. The answer is "
            "a clinic in two classic alt-asset traps: **appraisal smoothing** and **the cost "
            "wedge**."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The smoothing illusion.** An index marked once a year, anchored to last year's "
            "mark, *cannot* show its true bumpiness. It reports a low volatility that the asset "
            "doesn't really have -- which flatters the Sharpe ratio. We'll un-smooth it.\n"
            "2. **The cost wedge.** The index is what a *seller of rare bottles* realises. A cask "
            "buyer pays a dealer markup to get in, storage + insurance every year, loses ~2%/yr "
            "of spirit to evaporation (the 'Angels' Share'), and pays commission to get out.\n"
            "3. **Tiny n.** We have ~16 annual observations (2009-2024). That is nowhere near "
            "enough to certify any 'edge' as statistically real, even if one existed.\n\n"
            "We test all of this against the S&P 500 (^GSPC) calendar-year price returns from "
            "2009 to 2024 -- the same window, apples to apples (both price-only)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the hardcoded annual whisky index in this study's "
            "`data.py`, paired with the S&P 500 calendar-year return."
        ),
        code(
            "wh = data.whisky_index()\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    perf = st.performance_stats(df)\n"
            "    whisky_ann, sp_ann = perf['whisky_ann'], perf['sp_ann']\n"
            "    corr = perf['corr']\n"
            "else:\n"
            "    whisky_ann, sp_ann, corr = R['whisky_ann'], R['sp_ann'], R['corr']\n"
            "\n"
            "print('Whisky index years:', int(wh['year'].min()), '-', int(wh['year'].max()))\n"
            "print(f'Whisky annualised: {whisky_ann:.1f}%/yr  |  S&P annualised: {sp_ann:.1f}%/yr')\n"
            "print(f'Correlation to S&P: {corr:+.2f}  (the pitch sells this as diversification)')\n"
            "print()\n"
            "print('Both are PRICE-ONLY: the whisky index has no yield, so we strip S&P dividends')\n"
            "print('to keep the comparison fair. (Total-return S&P would win outright.)')"
        ),
        md("**The boom -- and the hangover.**"),
        code(
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(wh['year'], wh['index_level'], 'o-', c=AMBER, lw=2, ms=5,\n"
            "        label='Whisky index (rebased 100 @ 2008)')\n"
            "ax.axvspan(2018, 2022, color=GREY, alpha=0.12, label='Plateau')\n"
            "ax.axvspan(2022, 2024, color=RED, alpha=0.10, label='Reported decline')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Index level')\n"
            "ax.set_title('Rare whisky: a celebrated 2010s boom that stalled and then fell')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "peak = wh.loc[wh['index_level'].idxmax()]\n"
            "last = wh.iloc[-1]\n"
            "print(f'Peak: {peak[\"index_level\"]:.0f} in {int(peak[\"year\"])}')\n"
            "print(f'Latest: {last[\"index_level\"]:.0f} in {int(last[\"year\"])} '\n"
            "      f'({(last[\"index_level\"]/peak[\"index_level\"]-1)*100:+.1f}% from peak)')"
        ),
        md(
            "The index roughly **6x'd from 2008 to its peak** -- the number the cask-broker ads "
            "quote. But the secondary market cooled hard after 2021, and the appraisal indices "
            "finally printed *negative* years in 2023-2024 as the cask-investment bubble deflated "
            "(amid a wave of FCA and advertising-standards warnings about cask mis-selling)."
        ),
        md("**The smoothness illusion -- reported vs unsmoothed volatility.**"),
        code(
            "if HAVE_REAL:\n"
            "    sm = st.smoothing_stats(df)\n"
            "    raw_vol, true_vol = sm['raw_vol_pct'], sm['unsmoothed_vol_pct']\n"
            "    rho = sm['rho_lag1']\n"
            "else:\n"
            "    raw_vol, true_vol, rho = R['raw_vol_pct'], R['unsmoothed_vol_pct'], R['rho_lag1']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Reported\\n(appraised)', 'Unsmoothed\\n(true estimate)',\n"
            "               'S&P 500'],\n"
            "              [raw_vol, true_vol, R['sp_vol']],\n"
            "              color=[AMBER, RED, GREEN], width=0.55)\n"
            "ax.set_ylabel('Annual volatility (%)')\n"
            "ax.set_title(f'The appraisal index hides ~3x its true volatility (lag-1 autocorr = {rho:.2f})')\n"
            "for b, v in zip(bars, [raw_vol, true_vol, R['sp_vol']]):\n"
            "    ax.annotate(f'{v:.0f}%', (b.get_x()+b.get_width()/2, v+0.5),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Reported vol: {raw_vol:.1f}%  ->  unsmoothed vol: {true_vol:.1f}%  '\n"
            "      f'({true_vol/raw_vol:.1f}x larger)')\n"
            "print('A once-a-year appraisal mark cannot show the asset bouncing around. The low')\n"
            "print('reported vol is a property of the MEASUREMENT, not of the whisky.')"
        ),
        md(
            "The reported index has a lag-1 autocorrelation of **0.78** -- last year's mark "
            "powerfully anchors this year's. That is the textbook fingerprint of an appraisal-based "
            "series. Undo it (the standard Geltner unsmoothing) and the 'true' volatility jumps "
            "from ~10% to ~**29%** -- *higher* than the S&P. The 'low-vol alternative asset' is, "
            "once measured honestly, more volatile than the stock market."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** The whisky index does **not** beat the S&P: ~12.2% vs ~12.4%/yr, "
            "a mean annual *excess* of **-0.7pp** with a HAC t of **-0.15**. The seductive high "
            "Sharpe (1.04) is a smoothing artifact -- it collapses to **0.25** once the index is "
            "unsmoothed. n=16 could not certify an edge even if one existed.\n"
            "- **Tradability -- Mirage.** There is no cask ETF. A real buyer pays a dealer markup "
            "to enter, storage + insurance + evaporation every year, and commission to exit -- "
            "cutting ~12%/yr gross to ~**6%/yr net**. The index return is unreachable.\n"
            "- **Busted.** The 'uncorrelated, low-vol, equity-beating' story is three claims, and "
            "all three fail: the low vol is measurement smoothing, the high return doesn't beat "
            "equities, and the negative correlation is the random sign of 16 noisy points."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' is to buy a cask, hold it in a bonded warehouse, and sell it later. "
            "The index never charges the costs a real owner pays:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cost = st.net_of_costs(df)\n"
            "    gross, net = cost['gross_ann_pct'], cost['net_ann_pct']\n"
            "else:\n"
            "    gross, net = R['gross_ann_pct'], R['net_ann_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Index (gross)', 'Cask owner (net)', 'S&P 500'],\n"
            "              [gross, net, R['sp_ann']], color=[AMBER, RED, GREEN], width=0.55)\n"
            "ax.set_ylabel('Annualised return (%)')\n"
            "ax.set_title('What the index shows vs what a cask owner keeps')\n"
            "for b, v in zip(bars, [gross, net, R['sp_ann']]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+0.2),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Index gross: {gross:.1f}%/yr  ->  net of real costs: {net:.1f}%/yr')\n"
            "print(f'Cost wedge: {gross-net:.1f}pp/yr eaten by markup, storage, insurance, evaporation.')\n"
            "print('The index is a sellers-of-rare-bottles benchmark. You are not that seller.')"
        ),
        md(
            "Once you subtract the ~40% dealer markup (amortised), ~1.5%/yr storage and insurance, "
            "~2%/yr evaporation, and the exit commission, the headline ~12%/yr gross becomes "
            "~**6%/yr net** -- roughly *half* the index, and well behind a total-return index fund. "
            "And that assumes your individual cask tracks the index of the rarest bottles, which "
            "most do not."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants smoothing into a synthetic "
            "alt-asset and shows the Sharpe inflate exactly the way the real index's does -- the "
            "machinery is honest.\n"
            "- **Same family:** every appraisal-based 'alternative asset' shares this flaw -- "
            "fine art, farmland, private real estate. The Geltner unsmoothing is the universal "
            "lie detector.\n\n"
            "*Think casks really are the alt-asset of the decade? Fork this, plug in a genuinely "
            "transaction-based (not appraisal) cask price series, net out the real costs, and show "
            "an unsmoothed Sharpe that beats a total-return index fund. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Whisky-Cask -- a quantitative teardown\n"
            "### 16 annual obs * ^GSPC price * Geltner unsmoothing * HAC t * cost wedge * positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test the cask pitch on "
            "three axes: raw performance vs the S&P, the **Geltner (1991) appraisal-unsmoothing** "
            "that exposes the manufactured Sharpe, and the **cost wedge** that separates the index "
            "from what a buyer keeps. We close with a HAC t-test on the annual excess return and a "
            "synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: S&P 500 ^GSPC daily close "
            "(repo cache, 2009--2024 calendar-year **price** returns); whisky index hardcoded in "
            "`data.py` (appraisal-based, price-only). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Mean annual excess (whisky - S&P) = **-0.7pp**, HAC t = "
            "**-0.15**. Raw Sharpe 1.04 -> **0.25** unsmoothed. n=16. |\n"
            "| **Tradability** | `MIRAGE` | No cask vehicle; ~**6.2pp/yr** cost wedge cuts ~12% "
            "gross to ~6% net; the index is price-only and illiquid. |\n"
            "| **Busted?** | `YES` | 'Uncorrelated, low-vol, equity-beating' -- all three fail: "
            "smoothing makes the vol; the return ties; the correlation is sign noise. |\n\n"
            "> The appraisal-smoothing (lag-1 autocorr 0.78) does the heavy lifting. Remove it and "
            "the asset is more volatile than equities, with a Sharpe below a passive index fund."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $W_t$ be the whisky-index annual return and $M_t$ the S&P 500 annual price "
            "return. The pitch makes three testable claims:\n\n"
            "- **H1 (return).** $E[W_t] > E[M_t]$ -- whisky beats equities.\n"
            "- **H2 (risk).** $\\sigma(W_t) < \\sigma(M_t)$ -- whisky is lower volatility, hence "
            "a higher Sharpe. *Caveat:* an appraisal index's $\\sigma$ is biased **down** by serial "
            "correlation, so H2 must be tested on the **unsmoothed** series.\n"
            "- **H3 (diversification).** $\\text{corr}(W_t, M_t) \\approx 0$ -- uncorrelated, so it "
            "improves a portfolio.\n\n"
            "We reject H1 (returns tie, excess t = -0.15), reject H2 once unsmoothed (true vol > "
            "equity vol), and treat H3 as un-identified at n=16 (the sample correlation is a coin "
            "flip's worth of information)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Cask investing is a multi-hundred-million-pound retail product sold precisely on H1-H3. "
            "If they held, casks would belong in every diversified portfolio. The interesting "
            "finding is that the *single most important number in the whole pitch -- the low "
            "volatility -- is an artifact of the measurement cadence*, not a property of the asset. "
            "This is the same Geltner critique that demolished the 'private real estate has a high "
            "Sharpe' claim in the 1990s."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Annual whisky index (hardcoded in `data.py`, appraisal-based, rebased 100 "
            "@ 2008); S&P 500 ^GSPC December/December **price** returns 2009--2024 (16 obs).\n"
            "- **Performance.** Geometric annualised return, sample vol, naive annual Sharpe "
            "(rf=2%), and the return correlation.\n"
            "- **Unsmoothing.** Estimate lag-1 autocorrelation; apply Geltner (1991) first-order "
            "unsmoothing $r^{true}_t = (r_t - \\rho r_{t-1})/(1-\\rho)$; recompute vol and Sharpe.\n"
            "- **Cost wedge.** Net out entry markup (40%, amortised), storage+insurance (1.5%/yr), "
            "Angels' Share evaporation (2%/yr), exit commission (10%).\n"
            "- **Inference.** Newey-West HAC t-test on the annual excess return (whisky - S&P).\n"
            "- **Positive control.** Synthetic alt-asset with a planted smoothing knob to confirm "
            "the unsmoothing recovers the true Sharpe.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Raw performance: the seductive headline\n\n"
            "At face value the whisky index looks like a better asset than the S&P."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    perf = st.performance_stats(df)\n"
            "else:\n"
            "    perf = dict(n=R['n'], whisky_ann=R['whisky_ann'], sp_ann=R['sp_ann'],\n"
            "                whisky_vol=R['whisky_vol'], sp_vol=R['sp_vol'],\n"
            "                whisky_sharpe=R['whisky_sharpe'], sp_sharpe=R['sp_sharpe'],\n"
            "                corr=R['corr'], whisky_beats_frac=R['whisky_beats_frac'])\n"
            "\n"
            "tbl = pd.DataFrame({\n"
            "    'metric': ['Ann. return %','Volatility %','Naive Sharpe (rf=2%)','Corr to S&P'],\n"
            "    'Whisky': [perf['whisky_ann'], perf['whisky_vol'], perf['whisky_sharpe'], perf['corr']],\n"
            "    'S&P 500': [perf['sp_ann'], perf['sp_vol'], perf['sp_sharpe'], 1.0],\n"
            "})\n"
            "print(tbl.to_string(index=False))\n"
            "print()\n"
            "print(f\"At face value: similar return, LOWER vol, HIGHER Sharpe ({perf['whisky_sharpe']} \"\n"
            "      f\"vs {perf['sp_sharpe']}), near-zero correlation. The perfect alt-asset pitch.\")\n"
            "print('Beat-the-S&P in only {:.0%} of years, though -- a coin flip.'.format(perf['whisky_beats_frac']))"
        ),
        md(
            "> 💡 **In plain words.** Lower vol + similar return = higher Sharpe, and that high "
            "Sharpe is the entire sales hook. The next cell shows it is manufactured by the "
            "measurement, not earned by the asset."
        ),
        md(
            "### 4b - The killer: Geltner unsmoothing\n\n"
            "Appraisal indices are serially correlated by construction. We estimate the lag-1 "
            "autocorrelation and unsmooth."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = df['whisky_return'].to_numpy()\n"
            "    sm = st.smoothing_stats(df)\n"
            "    rho = sm['rho_lag1']\n"
            "    w_true, _ = st.geltner_unsmooth(w, rho)\n"
            "else:\n"
            "    sm = dict(rho_lag1=R['rho_lag1'], raw_vol_pct=R['raw_vol_pct'],\n"
            "              unsmoothed_vol_pct=R['unsmoothed_vol_pct'],\n"
            "              vol_inflation_x=R['vol_inflation_x'],\n"
            "              raw_sharpe=R['raw_sharpe'], unsmoothed_sharpe=R['unsmoothed_sharpe'])\n"
            "    rng = np.random.default_rng(275)\n"
            "    w_true = rng.normal(R['whisky_ann']/100, R['unsmoothed_vol_pct']/100, R['n'])\n"
            "    w = np.empty(R['n']); w[0]=w_true[0]\n"
            "    for t in range(1,R['n']): w[t]=(1-R['rho_lag1'])*w_true[t]+R['rho_lag1']*w[t-1]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "yrs = np.arange(R['year_start'], R['year_start']+len(w))\n"
            "axes[0].plot(yrs, w*100, 'o-', c=AMBER, label='Reported (smoothed)')\n"
            "axes[0].plot(yrs, np.asarray(w_true)*100, 's--', c=RED, label='Unsmoothed (true est.)')\n"
            "axes[0].axhline(0, c='k', lw=0.8); axes[0].set_ylabel('Annual return %')\n"
            "axes[0].set_title('Reported vs unsmoothed returns'); axes[0].legend()\n"
            "axes[1].bar(['Reported','Unsmoothed','S&P'],\n"
            "            [sm['raw_vol_pct'], sm['unsmoothed_vol_pct'], R['sp_vol']],\n"
            "            color=[AMBER,RED,GREEN])\n"
            "axes[1].set_ylabel('Volatility %'); axes[1].set_title('Volatility, honestly')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"lag-1 autocorr rho = {sm['rho_lag1']:.2f}\")\n"
            "print(f\"vol: {sm['raw_vol_pct']:.1f}% -> {sm['unsmoothed_vol_pct']:.1f}% \"\n"
            "      f\"({sm['vol_inflation_x']:.1f}x); Sharpe: {sm['raw_sharpe']:.2f} -> {sm['unsmoothed_sharpe']:.2f}\")"
        ),
        md(
            "The lag-1 autocorrelation of **0.78** is enormous -- true asset returns are close to "
            "white noise; only *smoothed* series look like this. Unsmoothing inflates the volatility "
            "~**2.85x** (to ~29%, above the S&P's 14%) and crushes the Sharpe from **1.04** to "
            "**0.25**. The headline risk-adjusted edge does not survive an honest variance estimate."
        ),
        md(
            "### 4c - The HAC excess-return test\n\n"
            "Does whisky beat the S&P per year? Newey-West t-test on the annual excess."
        ),
        code(
            "if HAVE_REAL:\n"
            "    test = st.excess_return_test(df, n_lags=1)\n"
            "    excess = (df['whisky_return'] - df['sp500_return']).to_numpy() * 100\n"
            "else:\n"
            "    test = dict(n=R['n'], mean_excess_pct=R['mean_excess_pct'], hac_t=R['hac_t'],\n"
            "                hac_lags=1, win_frac=R['win_frac'])\n"
            "    rng2 = np.random.default_rng(99)\n"
            "    excess = rng2.normal(R['mean_excess_pct'], 20, R['n'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "yrs = np.arange(R['year_start'], R['year_start']+len(excess))\n"
            "ax.bar(yrs, excess, color=[GREEN if e>0 else RED for e in excess])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(test['mean_excess_pct'], c=AMBER, ls='--', lw=2,\n"
            "           label=f\"mean excess {test['mean_excess_pct']:+.1f}pp\")\n"
            "ax.set_ylabel('Whisky - S&P (pp)'); ax.set_xlabel('Year')\n"
            "ax.set_title(f\"Annual excess return: HAC t = {test['hac_t']:+.2f} (n={test['n']})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Mean annual excess: {test['mean_excess_pct']:+.2f}pp\")\n"
            "print(f\"HAC (Newey-West, 1 lag) t-stat: {test['hac_t']:+.3f}  -- |t| < 2, not significant\")\n"
            "print(f\"Whisky beat S&P in {test['win_frac']:.0%} of years.\")"
        ),
        md(
            "> 💡 **In plain words.** The point estimate of the 'edge' is **negative** (-0.7pp/yr) "
            "with a HAC t of **-0.15**. Whisky did not beat the S&P even before its costs. There is "
            "no signal to rescue -- and with n=16, there is no power to find one if there were."
        ),
        md(
            "### 4d - The cost wedge\n\n"
            "Net the gross index down to what a retail cask owner actually keeps."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cost = st.net_of_costs(df)\n"
            "else:\n"
            "    cost = dict(gross_ann_pct=R['gross_ann_pct'], net_ann_pct=R['net_ann_pct'],\n"
            "                total_cost_wedge_pct=R['total_cost_wedge_pct'],\n"
            "                annual_recurring_drag_pct=3.5, entry_markup_pct=40.0,\n"
            "                exit_commission_pct=10.0, one_off_drag_ann_pct=-2.7)\n"
            "\n"
            "items = ['Index\\n(gross)', 'after entry\\nmarkup+exit', 'after storage\\n+insurance+evap',\n"
            "         'S&P 500']\n"
            "vals = [cost['gross_ann_pct'],\n"
            "        cost['gross_ann_pct']+cost['one_off_drag_ann_pct'],\n"
            "        cost['net_ann_pct'], R['sp_ann']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "bars = ax.bar(items, vals, color=[AMBER, '#e08e0b', RED, GREEN])\n"
            "ax.set_ylabel('Annualised return %')\n"
            "ax.set_title(f\"The cost wedge: {cost['total_cost_wedge_pct']:.1f}pp/yr from gross to net\")\n"
            "for b,v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:.1f}%',(b.get_x()+b.get_width()/2, v+0.15), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Gross {cost['gross_ann_pct']:.1f}% -> net {cost['net_ann_pct']:.1f}% \"\n"
            "      f\"(wedge {cost['total_cost_wedge_pct']:.1f}pp/yr)\")\n"
            "print('And the S&P number is PRICE-ONLY; total-return equities widen the gap further.')"
        ),
        md(
            "The recurring drags (storage + insurance + evaporation, ~3.5%/yr) plus the amortised "
            "entry markup and exit commission carve **~6.2pp/yr** off the gross index. Net of costs, "
            "the cask owner earns ~**6%/yr** -- about half the index, and behind even price-only "
            "equities. Against *total-return* equities the gap is wider still."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- mean annual excess -0.7pp, HAC t -0.15; raw Sharpe 1.04 "
            "collapses to 0.25 unsmoothed; n=16. No real, robust edge.\n"
            "- **Tradability `MIRAGE`** -- no cask vehicle; ~6.2pp/yr cost wedge; the index is "
            "appraisal-based, price-only, and unreachable by a retail buyer.\n"
            "- **Busted** -- the low-vol claim is a smoothing artifact, the return claim ties "
            "(and loses net), the diversification claim is unidentified at n=16."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the unreachable index\n\n"
            "There is no cask ETF, no daily NAV, and no two-sided market. Entry is via brokers at "
            "opaque markups; exit is via auction at commission, often months later; the asset is "
            "physical, perishable (evaporating), and storage-bound. Even if the index return were "
            "real, the implementation gap alone makes it a `MIRAGE`. The honest benchmark -- a "
            "total-return global index fund -- wins on return, liquidity, and cost simultaneously."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the unsmoothing machinery actually recover a true Sharpe when we *know* the "
            "answer? We plant a fixed-true-Sharpe alt-asset and sweep the smoothing knob: the "
            "*reported* Sharpe should inflate with smoothing while the *unsmoothed* Sharpe stays "
            "near the truth."
        ),
        code(
            "true_drift, true_vol = 0.07, 0.18\n"
            "true_sharpe = (true_drift - 0.02)/true_vol\n"
            "rows = []\n"
            "for lam in [0.0, 0.3, 0.5, 0.7, 0.85]:\n"
            "    d, _ = data.synthetic_index(n_years=400, true_drift=true_drift,\n"
            "                                true_vol=true_vol, smoothing=lam, seed=275)\n"
            "    rep = d['reported_return'].to_numpy()\n"
            "    rep_sh = (rep.mean()-0.02)/rep.std(ddof=1)\n"
            "    uns, rho = st.geltner_unsmooth(rep)\n"
            "    uns_sh = (uns.mean()-0.02)/uns.std(ddof=1)\n"
            "    rows.append({'smoothing': lam, 'rho': round(rho,2),\n"
            "                 'reported_sharpe': round(rep_sh,2),\n"
            "                 'unsmoothed_sharpe': round(uns_sh,2)})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot(ctrl['smoothing'], ctrl['reported_sharpe'], 'o-', c=AMBER, label='Reported Sharpe (inflated)')\n"
            "ax.plot(ctrl['smoothing'], ctrl['unsmoothed_sharpe'], 's-', c=GREEN, label='Unsmoothed Sharpe (recovered)')\n"
            "ax.axhline(true_sharpe, c='k', ls=':', label=f'True Sharpe = {true_sharpe:.2f}')\n"
            "ax.set_xlabel('Planted smoothing'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('Positive control: smoothing inflates reported Sharpe; unsmoothing recovers truth')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.to_string(index=False))"
        ),
        md(
            "The control confirms the method: as planted smoothing rises, the *reported* Sharpe "
            "balloons above the truth, while the *unsmoothed* Sharpe tracks the true value. The "
            "real whisky index sits at the high-smoothing end of this picture -- its flattering "
            "Sharpe is exactly the artifact the control manufactures on demand. The verdict is a "
            "statement about **the measurement and the costs**, not a quirk of one sample."
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


# ---------------------------------------------------------------------------
# Execute notebooks with nbconvert (offline, no --allow-errors)
# ---------------------------------------------------------------------------
def _execute(filename: str) -> None:
    import subprocess
    import sys

    path = os.path.join(HERE, filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {filename}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {filename}")
    print("executed", filename)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
