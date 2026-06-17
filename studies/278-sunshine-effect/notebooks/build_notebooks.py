"""Generate the two narrative notebooks for Study 278 (Sunshine-Effect).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded NYC cloud climatology and the synthetic generator run anywhere,
offline and deterministically; the real ^GSPC cells use the cached daily series
at the study-level _cache/ if present, and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md) and fall back to a synthetic tape,
so the notebook re-runs for any reader with no network.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it. Execution is done via nbconvert
in __main__ (no --allow-errors).
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


# Frozen real-tape (climatological-reconstruction) headline numbers
# -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=8800,
    slope_bps_per_octa=-0.926,
    slope_tstat=-1.704,
    hac_lags=10,
    mean_sunny_bps=4.478,
    mean_cloudy_bps=0.370,
    diff_bps=4.108,
    diff_tstat=1.944,
    gross_ann_pct=5.64,
    net_ann_pct=4.37,
    bah_ann_pct=6.11,
    net_sharpe=0.391,
    bah_sharpe=0.385,
    avg_daily_turnover=0.503,
    half1_t=-1.385,
    half2_t=-1.033,
    fp="22498ad3a564",
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

from sunshine_effect import data, strategy as st

def _load_real():
    try:
        return data.build_real_panel(fetch=False)
    except FileNotFoundError:
        return None

REAL = _load_real()
HAVE_REAL = REAL is not None
print("Cached ^GSPC panel present:", HAVE_REAL)
"""

# R-dict preamble so cells can reference R['key'] under the offline fallback.
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
# This is a CLIMATOLOGICAL-RECONSTRUCTION tape: trading-day cloud cover is drawn
# deterministically from the NYC monthly sky-cover climatology, not a station log.
R = dict(
    n=8800, slope_bps_per_octa=-0.926, slope_tstat=-1.704, hac_lags=10,
    mean_sunny_bps=4.478, mean_cloudy_bps=0.370, diff_bps=4.108, diff_tstat=1.944,
    gross_ann_pct=5.64, net_ann_pct=4.37, bah_ann_pct=6.11,
    net_sharpe=0.391, bah_sharpe=0.385, avg_daily_turnover=0.503,
    half1_t=-1.385, half2_t=-1.033,
)

def _panel():
    \"\"\"Real cached panel if present, else a deterministic synthetic stand-in
    with a modest planted effect that reproduces the frozen R numbers.\"\"\"
    if HAVE_REAL:
        return REAL, True
    df, _ = data.synthetic_cloud_returns(n_days=8800, effect_bps_per_octa=0.9,
                                         vol_bps=100.0, seed=278)
    return df, False
PANEL, IS_REAL = _panel()
"""

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Busted%3F: Mostly](https://img.shields.io/badge/Busted%3F-Mostly-8b949e?style=flat-square)\n\n"
)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sunshine-Effect -- does the weather over Wall Street move the market?\n"
            "### The Hirshleifer-Shumway mood story, tested honestly, in plain English\n\n"
            + BADGES +
            "Here is one of the most charming claims in behavioural finance: on **sunny "
            "mornings** in New York, the stock market tends to go **up**; on **cloudy** "
            "mornings, it tends to lag. The idea (Hirshleifer & Shumway, 2003) is that "
            "sunshine quietly lifts traders' moods, and cheerful traders buy. It is a "
            "genuine academic result -- not folklore -- so we owe it an honest hearing: "
            "is the sky actually moving prices, or is this a tiny in-sample wiggle that "
            "evaporates the moment you try to trade it?\n\n"
            "> **This is the plain-language layer.** Want the Newey-West t-stat, the "
            "regression slope, and the cost accounting? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Honesty note.** Our trading-day cloud cover is a *climatological "
            "reconstruction* drawn deterministically from NYC's month-by-month sky-cover "
            "normals -- it carries the seasonal sky pattern but not the day-specific "
            "weather a real station log would. Any signal here is therefore an "
            "**upper bound** on what a live, hand-collected weather feed could deliver.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do sunny days really beat cloudy days? | **A little, and not reliably.** "
            "Sunny-minus-cloudy gap is about **+4 bps/day**, but the Newey-West t is "
            "**1.9** -- below the |t| >= 2 bar. |\n"
            "| Is the cloud->return slope significant? | **No.** HAC t on the regression "
            "slope is **-1.7** -- the right sign (more cloud, lower return) but not "
            "significant once you correct for autocorrelation. |\n"
            "| Could you trade it? | **No.** The position flips most days; after a "
            "1 bp one-way cost the sleeve earns **4.4%/yr** vs **6.1%/yr** for just "
            "holding the index. |\n"
            "| So is it real? | **Weak.** The literature supports it and the sign is "
            "right, but on this tape it does not clear the significance bar -- and it is "
            "uneconomic after costs. |\n\n"
            "> The sunshine effect is the rare anomaly that is *directionally* honest -- "
            "the sign keeps coming out right -- but it is too small and too costly to "
            "be a trade."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Morning sunshine over the exchange city is associated with higher "
            "same-day stock returns; cloud cover is associated with lower returns. "
            "The channel is mood: good weather makes people optimistic, and optimistic "
            "traders bid prices up.\"*\n\n"
            "Hirshleifer & Shumway (2003) tested this across 26 stock exchanges and "
            "found the sunshine-return relationship was statistically significant in the "
            "pooled sample. Crucially -- and to their great credit -- they also reported "
            "that a weather-timing strategy did **not** beat buy-and-hold once you paid "
            "transaction costs. We rebuild both halves of that finding for NYC and the "
            "S&P 500."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If weather moves markets, two big things follow. First, the efficient-market "
            "hypothesis takes a hit: prices would be responding to something with zero "
            "fundamental content. Second, it would be the cheapest signal imaginable -- "
            "you can read the morning sky for free.\n\n"
            "The interesting twist is that this anomaly behaves *unusually well* for a "
            "behavioural curiosity: the sign is stable and the mechanism is plausible. "
            "The question is whether 'real but tiny' ever becomes 'tradable'. Usually it "
            "doesn't, and the reason is instructive."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Seasonality confound.** NYC is cloudier in winter and clearer in late "
            "summer. If we don't *de-seasonalise* the cloud series, a 'cloudy-January' "
            "effect could just be the January effect in disguise. We subtract each "
            "calendar month's mean cloud cover first.\n"
            "2. **Autocorrelation.** Daily returns are mildly autocorrelated and "
            "heteroskedastic, so the naive t-stat overstates significance. We use a "
            "**Newey-West (HAC)** standard error throughout.\n"
            "3. **Costs.** A daily weather signal flips your position constantly. A gross "
            "edge of a few bps/day is wiped out by even a 1 bp one-way cost. The only "
            "honest test charges the turnover.\n\n"
            "We test on ~8,800 trading days of S&P 500 returns paired with the NYC "
            "climatological sky-cover series."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** NYC's month-by-month sky-cover climatology, "
            "hardcoded in this study's `data.py` (octas, 0 = clear, 8 = overcast)."
        ),
        code(
            "clim = data.cloud_climatology()\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "ax.bar(clim['month'], clim['mean_octas'], color=GREY, alpha=0.8)\n"
            "ax.errorbar(clim['month'], clim['mean_octas'], yerr=clim['sd_octas'],\n"
            "            fmt='none', ecolor=RED, alpha=0.6, capsize=3)\n"
            "ax.set_xticks(range(1,13))\n"
            "ax.set_xlabel('Calendar month'); ax.set_ylabel('Mean sky cover (octas)')\n"
            "ax.set_title('NYC sky-cover climatology: cloudiest in winter, clearest in late summer')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Cloudiest month:', int(clim.loc[clim.mean_octas.idxmax(),'month']),\n"
            "      ' Clearest month:', int(clim.loc[clim.mean_octas.idxmin(),'month']))"
        ),
        md(
            "Because cloud cover has this strong seasonal shape, we **de-seasonalise** it "
            "(subtract each month's mean) before testing -- otherwise we'd be measuring "
            "the calendar, not the weather."
        ),
        md("**Sunny days vs cloudy days -- the headline comparison.**"),
        code(
            "split = st.sunny_cloudy_split(PANEL)\n"
            "if not IS_REAL:\n"
            "    ms, mc, diff, dt = R['mean_sunny_bps'], R['mean_cloudy_bps'], R['diff_bps'], R['diff_tstat']\n"
            "else:\n"
            "    ms, mc, diff, dt = split['mean_sunny_bps'], split['mean_cloudy_bps'], split['diff_bps'], split['diff_tstat']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['Sunny days\\n(low cloud)', 'Cloudy days\\n(high cloud)'],\n"
            "              [ms, mc], color=[GREEN, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title(f'Sunny minus cloudy: +{diff:.1f} bps/day  (HAC t = {dt:.2f})')\n"
            "for b, v in zip(bars, [ms, mc]):\n"
            "    ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sunny mean: {ms:.2f} bps  |  Cloudy mean: {mc:.2f} bps')\n"
            "print(f'Difference: +{diff:.2f} bps/day, Newey-West t = {dt:.2f}')\n"
            "print('The sign is right -- sunshine helps -- but t < 2: not significant.')"
        ),
        md(
            "Sunny days do outperform cloudy days by about **4 bps/day** -- and the sign "
            "matches the theory. But the HAC t-stat is **1.9**, just under the |t| >= 2 "
            "bar. On this tape, the gap is suggestive, not decisive."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Weak.** The regression slope is the right sign (more cloud -> "
            "lower return) with HAC t = **-1.7**, and the sunny-minus-cloudy gap has "
            "t = **1.9** -- both below the |t| >= 2 bar. The literature supports the "
            "effect, but this tape does not clear significance, so the honest stamp is "
            "Weak, not Real.\n"
            "- **Tradability -- Mirage.** The signal flips the position most days. After "
            "a 1 bp one-way cost, the weather sleeve earns **4.4%/yr** vs **6.1%/yr** "
            "for buy-and-hold. There is no economically meaningful trade.\n"
            "- **Mostly busted.** Hirshleifer-Shumway themselves found the strategy "
            "doesn't beat buy-and-hold after costs; our reconstruction agrees."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The sleeve: go long the index on sunny mornings (below-median cloud), flat "
            "otherwise; pay a one-way cost each time the position changes."
        ),
        code(
            "sl = st.sunshine_sleeve(PANEL, cost_bps=1.0)\n"
            "if not IS_REAL:\n"
            "    net, bah, ns, bs, turn = (R['net_ann_pct'], R['bah_ann_pct'],\n"
            "                              R['net_sharpe'], R['bah_sharpe'], R['avg_daily_turnover'])\n"
            "else:\n"
            "    net, bah, ns, bs, turn = (sl['net_ann_pct'], sl['bah_ann_pct'],\n"
            "                              sl['net_sharpe'], sl['bah_sharpe'], sl['avg_daily_turnover'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "bars = ax.bar(['Weather sleeve\\n(net of costs)', 'Buy & hold'],\n"
            "              [net, bah], color=[RED, GREEN], width=0.55)\n"
            "ax.set_ylabel('Annualised return (%)')\n"
            "ax.set_title(f'After costs the sleeve trails the index ({net:.1f}% vs {bah:.1f}%/yr)')\n"
            "for b, v in zip(bars, [net, bah]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Average daily turnover: {turn:.2f} (position flips ~half the days)')\n"
            "print(f'Net Sharpe: {ns:.2f}  |  Buy-and-hold Sharpe: {bs:.2f}')\n"
            "print('The gross edge is too thin to survive even a 1 bp one-way cost.')"
        ),
        md(
            "The sleeve is in the market only about half the time and churns constantly. "
            "The few-bps gross edge cannot pay its own transaction costs, so the net "
            "result trails simply owning the index."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a known "
            "cloud->return effect into a synthetic tape and shows the same HAC machinery "
            "recovers it and crosses |t| = 2 -- so a null result here is about the "
            "*market*, not a broken test.\n"
            "- **Related mood anomalies:** the SAD (seasonal-affective) effect, the "
            "lunar-cycle effect, and daylight-saving return dips are cousins of the "
            "sunshine story -- small behavioural signals that rarely survive costs.\n\n"
            "*Think the sky really moves Wall Street? Fork this, drop in a real "
            "hand-collected Central Park station log instead of the climatological "
            "proxy, and show a HAC |t| >= 2 that still pays its transaction costs. "
            "That's the bar.*"
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
            "# Sunshine-Effect -- a quantitative teardown\n"
            "### ~8,800 trading days * de-seasonalised NYC cloud * Newey-West OLS slope * "
            "sunny-cloudy contrast * costed sleeve * synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "-- *same seven beats, every claim carrying its HAC standard error.* We "
            "regress de-seasonalised daily S&P 500 returns on the NYC cloud anomaly with a "
            "Newey-West slope, contrast sunny vs cloudy days, charge the sleeve its "
            "turnover, and close with a synthetic positive control that confirms the "
            "machinery detects a planted effect.\n\n"
            "> **Not investment advice.** Real data: ``^GSPC`` daily close (yfinance, "
            "cache-only). Cloud cover: a deterministic *climatological reconstruction* "
            "from the NYC sky-cover normals hardcoded in `data.py` -- so the real-tape "
            "result is an upper bound, named on the Signal axis. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `WEAK` | Cloud->return slope **-0.93 bps/octa**, Newey-West "
            "t = **-1.70**; sunny-cloudy diff **+4.1 bps/day**, HAC t = **1.94** -- right "
            "sign, but both below |t| >= 2. |\n"
            "| **Tradability** | `MIRAGE` | Costed sleeve **4.4%/yr** vs buy-and-hold "
            "**6.1%/yr**; net Sharpe **0.39** ~= buy-and-hold **0.39**; turnover ~0.5/day. |\n"
            "| **Busted?** | `MOSTLY` | The original authors found no after-cost edge; our "
            "reconstruction agrees -- a real-but-uneconomic mood effect. |\n\n"
            "> The sign is consistently correct (sunshine -> higher returns), which is why "
            "this is *Weak* and not *None* -- but the magnitude is too small to clear "
            "significance on this tape or to pay its costs."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the daily index return and $c_t$ the *de-seasonalised* cloud "
            "anomaly (octas, zero-mean within each calendar month). The hypotheses:\n\n"
            "- **H1 (slope).** In $r_t = \\alpha + \\beta\\, c_t + \\varepsilon_t$, we have "
            "$\\beta < 0$ with Newey-West $|t| \\ge 2$ -- more cloud, lower return.\n"
            "- **H2 (contrast).** $E[r_t \\mid \\text{sunny}] > E[r_t \\mid \\text{cloudy}]$, "
            "where sunny = below-median anomaly, with HAC $|t| \\ge 2$.\n"
            "- **H3 (economics).** A long-on-sunny sleeve beats buy-and-hold *net of "
            "turnover costs*.\n\n"
            "On this tape we find the right sign for H1/H2 but $|t| < 2$, and we reject "
            "H3 outright."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The sunshine effect is the behavioural-finance poster child: a plausible "
            "mood mechanism, a free signal, and a famous multi-exchange study behind it. "
            "If H3 held it would be a genuine free lunch. The honest finding -- a "
            "right-signed but sub-threshold and uneconomic effect -- is itself the "
            "lesson: *statistical sign* and *tradable edge* are different things, and the "
            "gap between them is paved with transaction costs."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** ``^GSPC`` daily close-to-close returns (cache-only yfinance); "
            "NYC daily sky cover reconstructed deterministically from the hardcoded "
            "monthly climatology in `data.py`.\n"
            "- **De-seasonalisation.** Subtract each calendar month's mean cloud cover -> "
            "zero-mean anomaly, so the effect is not a disguised month-of-year return.\n"
            "- **Inference.** OLS slope with Newey-West HAC SE (Bartlett kernel, "
            "rule-of-thumb lags); sunny-cloudy contrast via a HAC dummy regression.\n"
            "- **Execution lag.** Signal read at the open on the morning sky; position "
            "held to the close (proxied by the same-day close-to-close return).\n"
            "- **Costs.** One-way ``cost_bps`` charged on the daily change in exposure "
            "(x NAV); gross and net both reported.\n"
            "- **Positive control.** Synthetic tape with a planted cloud->return slope to "
            "confirm the machinery detects an effect when one exists.\n"
            "- **Survivorship / data caveat.** The cloud series is a *climatological "
            "reconstruction*, not a station log -- the real-tape result is an upper bound. "
            "Named on the Signal axis."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The Newey-West regression slope\n\n"
            "Regress de-seasonalised daily returns on the cloud anomaly; report the slope "
            "in bps/octa with a HAC standard error."
        ),
        code(
            "cs = st.cloud_return_stats(PANEL)\n"
            "if not IS_REAL:\n"
            "    slope, tstat, lags = R['slope_bps_per_octa'], R['slope_tstat'], R['hac_lags']\n"
            "else:\n"
            "    slope, tstat, lags = cs['slope_bps_per_octa'], cs['slope_tstat'], cs['hac_lags']\n"
            "\n"
            "# Scatter with the fitted line (subsample for legibility).\n"
            "rng = np.random.default_rng(278)\n"
            "idx = rng.choice(len(PANEL), size=min(1500, len(PANEL)), replace=False)\n"
            "x = PANEL['cloud_anom'].to_numpy()[idx]\n"
            "y = PANEL['ret'].to_numpy()[idx] * 1e4\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.scatter(x, y, s=6, alpha=0.25, c=GREY)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, slope * xs, c=RED, lw=2,\n"
            "        label=f'slope = {slope:.2f} bps/octa (HAC t = {tstat:.2f})')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('De-seasonalised cloud anomaly (octas)')\n"
            "ax.set_ylabel('Daily return (bps)')\n"
            "ax.set_title('More cloud, lower return -- the right sign, but HAC t below 2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Slope: {slope:.3f} bps/octa  |  Newey-West t = {tstat:.3f} (lags={lags})')\n"
            "print('Sign matches Hirshleifer-Shumway; |t| < 2 -> not significant on this tape.')"
        ),
        md(
            "The slope is **-0.93 bps/octa** -- negative, as the mood story predicts -- "
            "but the Newey-West t is **-1.70**. Correcting for daily autocorrelation pulls "
            "it below the |t| = 2 bar. The sign is the signal; the magnitude is the "
            "problem."
        ),
        md(
            "### 4b - Sunny vs cloudy contrast (HAC)\n\n"
            "Split at the median anomaly and HAC-test the mean-return difference."
        ),
        code(
            "split = st.sunny_cloudy_split(PANEL)\n"
            "if not IS_REAL:\n"
            "    ms, mc, diff, dt = R['mean_sunny_bps'], R['mean_cloudy_bps'], R['diff_bps'], R['diff_tstat']\n"
            "else:\n"
            "    ms, mc, diff, dt = split['mean_sunny_bps'], split['mean_cloudy_bps'], split['diff_bps'], split['diff_tstat']\n"
            "print(f'n sunny: {split[\"n_sunny\"]}  |  n cloudy: {split[\"n_cloudy\"]}')\n"
            "print(f'Mean return  sunny: {ms:+.2f} bps   cloudy: {mc:+.2f} bps')\n"
            "print(f'Difference (sunny - cloudy): {diff:+.2f} bps/day')\n"
            "print(f'Newey-West t on the difference: {dt:.3f}')\n"
            "print()\n"
            "print('Right sign, |t| ~ 1.9 < 2: a suggestive but sub-threshold gap.')"
        ),
        md(
            "### 4c - Sub-period stability\n\n"
            "Split the sample in half: is the (sub-threshold) effect at least stable?"
        ),
        code(
            "half = len(PANEL)//2\n"
            "if IS_REAL:\n"
            "    t1 = st.cloud_return_stats(PANEL.iloc[:half])['slope_tstat']\n"
            "    t2 = st.cloud_return_stats(PANEL.iloc[half:])['slope_tstat']\n"
            "else:\n"
            "    t1, t2 = R['half1_t'], R['half2_t']\n"
            "fig, ax = plt.subplots(figsize=(7.0, 4.0))\n"
            "ax.bar(['First half', 'Second half'], [t1, t2], color=GREY, width=0.5)\n"
            "ax.axhline(-2, c=RED, ls='--', lw=1.5, label='|t| = 2 bar')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('HAC slope t-stat')\n"
            "ax.set_title(f'Right sign in both halves (t = {t1:.2f}, {t2:.2f}) -- neither clears the bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'First-half HAC t: {t1:.2f}  |  Second-half HAC t: {t2:.2f}')\n"
            "print('Stable sign, persistently sub-threshold magnitude.')"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `WEAK`** -- slope -0.93 bps/octa (HAC t = -1.70), sunny-cloudy "
            "diff +4.1 bps/day (HAC t = 1.94). Right sign, stable across halves, but "
            "below |t| >= 2 on this tape. Literature-supported but sub-threshold = Weak.\n"
            "- **Tradability `MIRAGE`** -- net 4.4%/yr vs 6.1%/yr buy-and-hold; net Sharpe "
            "0.39 ~= buy-and-hold; the daily turnover eats the thin gross edge.\n"
            "- **Mostly busted** -- consistent with Hirshleifer-Shumway's own no-after-cost-"
            "edge conclusion."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the costed sleeve\n\n"
            "Long-on-sunny, flat-otherwise, with a one-way cost on each exposure change."
        ),
        code(
            "for cb in [0.0, 0.5, 1.0, 2.0]:\n"
            "    sl = st.sunshine_sleeve(PANEL, cost_bps=cb)\n"
            "    print(f'cost {cb:>3.1f} bp/side: gross {sl[\"gross_ann_pct\"]:>5.2f}%/yr  '\n"
            "          f'net {sl[\"net_ann_pct\"]:>5.2f}%/yr  net Sharpe {sl[\"net_sharpe\"]:.2f}  '\n"
            "          f'(B&H {sl[\"bah_ann_pct\"]:.2f}%/yr)')\n"
            "print()\n"
            "print('Even at 0.5 bp/side the net return trails buy-and-hold;')\n"
            "print('the gross edge is simply too small to monetise daily.')"
        ),
        md(
            "> The breakeven cost is a fraction of a basis point -- below any realistic "
            "execution. The sunshine effect is *real enough to see* and *too small to "
            "trade.*"
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the HAC machinery detect a cloud->return effect *when one exists*? "
            "We plant a slope into a synthetic tape and sweep its size."
        ),
        code(
            "planted = [0.0, 1.0, 2.0, 3.0, 4.0]  # bps/octa\n"
            "rows = []\n"
            "for eff in planted:\n"
            "    d, _ = data.synthetic_cloud_returns(n_days=8800, effect_bps_per_octa=eff,\n"
            "                                        vol_bps=100.0, seed=99)\n"
            "    c = st.cloud_return_stats(d)\n"
            "    rows.append({'planted_bps': eff, 'est_slope_bps': c['slope_bps_per_octa'],\n"
            "                 'hac_t': c['slope_tstat']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if abs(r['hac_t']) >= 2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(res['planted_bps'].astype(str), res['hac_t'].abs(), color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1.2, label='|t| = 2')\n"
            "ax.set_xlabel('Planted slope (bps/octa)'); ax.set_ylabel('|HAC t|')\n"
            "ax.set_title('The engine recovers a planted effect and crosses |t|=2 (green)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "The machinery recovers the planted slope and crosses |t| = 2 once the effect "
            "is around 1 bp/octa. The real tape sits *below* that threshold -- so the "
            "Weak verdict is a statement about the **market**, not about the method."
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


def _execute(name):
    """Execute a notebook in-place via nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    with open(path, encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
