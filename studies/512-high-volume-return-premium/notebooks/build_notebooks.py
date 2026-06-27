"""Generate the two narrative notebooks for Study 512 (High-Volume-Return-Premium).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached yfinance parquets
under ../_cache/ if present, otherwise they quote the frozen headline numbers in ``R``.

``R`` is the ONE dict of real numbers and it mirrors docs/results.md exactly (as-of 2026-06-26).
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_weeks=622, year_start=2014, year_end=2025, n_tickers=40,
    n_days=3016,
    fp_returns="f8b840c95180", fp_volume="1235e9c48831",
    # gross horizon=1 long-short
    ls_mean=-5.47, ls_vol=14.08, ls_sharpe=-0.388, ls_t=-1.343, ls_hac=-1.328,
    ls_hit=49.2, ls_dd=-61.3,
    long_leg=19.47, short_leg=24.94, turnover=64.5,
    # net of costs
    net_mean=-11.17, net_t=-2.743, net_hac=-2.713,
    cost_bps=5.0, borrow_bps=50.0,
    # placebo
    placebo_p=0.120, placebo_p_seeds=(0.125, 0.100, 0.117),
    real_weekly=-0.1051, null_weekly=-0.0036, null_std=0.0672,
    # horizon sweep (weeks -> (mean%/yr, t_os, sharpe))
    horizon={1: (-5.47, -1.343, -0.388), 2: (-9.15, -1.613, -0.467),
             4: (-19.31, -2.301, -0.667), 8: (-33.06, -2.703, -0.786)},
    # synthetic control (premium -> (ls_mean%/yr, t_os))
    control={-0.06: (-11.26, -2.904), -0.03: (-5.25, -1.355), 0.0: (0.75, 0.194),
             0.03: (6.76, 1.744), 0.06: (12.77, 3.294), 0.10: (20.78, 5.361)},
)


# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from high_volume_return_premium import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "hv_returns.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "hv_volume.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    ret, vol = data.fetch_panel()
    ret = data.drop_partial_last_week(ret); vol = vol.reindex(ret.index)
    ls   = st.long_short(ret, vol, horizon=1, cost_bps=0.0)
    ls_n = st.long_short(ret, vol, horizon=1, cost_bps=5.0, borrow_ann_bps=50.0)
    s_ls   = st.summary(ls["ls_gross"])
    s_net  = st.summary(ls_n["ls_net"])
    s_long = st.summary(ls["long_ret"])
    s_short= st.summary(ls["short_ret"])
    print(f"N weeks: {s_ls['n']} | LS mean: {s_ls['mean']*100:+.2f}%/yr"
          f" | one-sample t: {s_ls['t_os']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# High-Volume-Return-Premium -- do heavily-traded stocks keep climbing?\n"
            "### Gervais-Kaniel-Mingelgrin (2001): the visibility shock, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survivorship--biased%3F: Named](https://img.shields.io/badge/Survivorship--biased%3F-Named-8b949e?style=flat-square)\n\n"
            "In 2001, Simon Gervais, Ron Kaniel and Dan Mingelgrin published a curious finding: "
            "a stock that trades on **unusually high volume** over a day or a week tends to "
            "*appreciate* over the next few weeks -- and a stock that trades on unusually **low** "
            "volume tends to *lag*. They called it the **high-volume return premium** and "
            "explained it as a visibility shock: a heavy-volume day puts the stock on more "
            "investors' radar, the investor base widens, and the price drifts up.\n\n"
            "We test that exact sort -- rank by abnormal volume, long the loud names, short the "
            "quiet ones -- on a basket of large-cap survivors, naming the survivorship bias and "
            "the one execution lag up front.\n\n"
            "> **This is the plain-language layer.** Want the abnormal-volume construction, the "
            "label-shuffle null, and the horizon sweep? See the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the loud names out-run the quiet ones? | **No.** The long-short earns "
            f"**{R['ls_mean']:+.2f}%/yr** (one-sample *t* = **{R['ls_t']:+.2f}**) -- the "
            "*wrong* sign for the premium. |\n"
            "| Is the sort even real? | **No.** A within-week label-shuffle places the real "
            f"book at **p = {R['placebo_p']:.2f}** -- indistinguishable from random labels. |\n"
            "| Could you trade it? | **No.** Net of costs it loses "
            f"**{R['net_mean']:+.1f}%/yr** on ~{R['turnover']:.0f}%/week turnover. |\n"
            "| Survivorship problem? | **Yes, and we name it.** The blow-up names that trade "
            "on the biggest volume spikes were deleted from the basket. |\n\n"
            "> The premium is documented on old, broad NYSE tapes. On 12 years of 40 mega-cap "
            "survivors it is the wrong sign, statistically a coin, and badly negative net."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Stocks experiencing unusually high (low) trading volume over a day or a week "
            "tend to appreciate (depreciate) over the course of the following month. We call "
            "this the high-volume return premium ... consistent with ... the trade affecting "
            "the stock's visibility.\"*\n\n"
            "-- Gervais, Kaniel & Mingelgrin (2001), *Journal of Finance*\n\n"
            "The mechanism leans on Merton (1987): a stock known to more investors carries a "
            "higher price. A volume spike is an attention event -- it recruits new holders, and "
            "their buying lifts the price over the following weeks. The trade writes itself: "
            "buy this week's loud names, sell the quiet ones."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, it is a clean, cheap-to-compute signal: you only need volume, which every "
            "feed carries. It would say attention itself is priced at short horizons -- a "
            "behavioural anomaly with a tidy economic story. It has spawned a literature on "
            "attention and retail trading (Barber-Odean 2008) and cross-country tests "
            "(Kaniel-Ozoguz-Starks 2012).\n\n"
            "Our questions:\n"
            "1. **Does it survive on a modern, liquid, much-watched large-cap basket?**\n"
            "2. **Is the sort distinguishable from random labels?**\n"
            "3. **What does the survivorship bias do to a volume-spike sort?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Abnormal, not raw, volume.** We divide each week's average daily volume by "
            "the stock's own trailing 8-week mean -- a stock is 'high volume' relative to "
            "*itself*, computed on strictly past data.\n"
            "2. **One execution lag.** The signal is known at Friday's close; we trade the "
            "**next** week's return. No same-bar fill, no look-ahead.\n"
            "3. **A label-shuffle placebo.** We permute the volume labels across names within "
            "each week and re-run. If the real book is no better than random labels, the sort "
            "is noise -- and we say so."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the engine work when the premium is planted?**"
        ),
        code(
            "ctrl = st.sweep_synthetic()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.5 else (RED if m < -0.5 else GREY) for m in ctrl['ls_mean_ann_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['ls_mean_ann_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted high-volume premium'); ax.set_ylabel('recovered LS mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the premium when it is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: a planted premium of zero returns a coin (t≈0.2), and the "
            "recovered long-short is monotone in the planted premium. So the real-tape verdict "
            "reflects **the market**, not a broken detector."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    ls_m, ls_t = s_ls['mean']*100, s_ls['t_os']\n"
            "    lo_m, hi_m = s_short['mean']*100, s_long['mean']*100\n"
            "    net_m = s_net['mean']*100\n"
            "else:\n"
            "    ls_m, ls_t = R['ls_mean'], R['ls_t']\n"
            "    lo_m, hi_m = R['short_leg'], R['long_leg']\n"
            "    net_m = R['net_mean']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "labels = ['Low-vol\\n(short leg)', 'High-vol\\n(long leg)']\n"
            "axes[0].bar(labels, [lo_m, hi_m], color=[GREY, AMBER], width=0.5)\n"
            "axes[0].set_ylabel('forward 1-wk return (%/yr)')\n"
            "axes[0].set_title('Both quintiles rise in a bull sample -- quiet ones edge ahead')\n"
            "axes[1].bar(['gross', 'net'], [ls_m, net_m], color=[RED, RED], width=0.5)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('long-short mean (%/yr)')\n"
            "axes[1].set_title(f'Long-short is the WRONG sign | t={ls_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long-short: {ls_m:+.2f}%/yr gross | {net_m:+.2f}%/yr net | t={ls_t:+.2f}')"
        ),
        md(
            f"The loud-minus-quiet book earns **{R['ls_mean']:+.2f}%/yr** at one-sample *t* = "
            f"**{R['ls_t']:+.2f}** -- the wrong sign and well inside |t|<2. Net of costs it is "
            f"**{R['net_mean']:+.1f}%/yr**. There is no high-volume premium to harvest here."
        ),
        md("**Is the sort distinguishable from random labels?**"),
        code(
            "if HAVE_REAL:\n"
            "    rm = float(ls['ls_gross'].mean())\n"
            "    pb = st.placebo_pvalue(ret, vol, real_mean=rm, n_shuffles=150, horizon=1)\n"
            "    p, nm, ns = pb['p'], pb['null_mean']*100, pb['null_std']*100\n"
            "    real_wk = rm*100\n"
            "else:\n"
            "    p, nm, ns, real_wk = R['placebo_p'], R['null_weekly'], R['null_std'], R['real_weekly']\n"
            "print(f'Placebo p-value: {p:.3f}  (real weekly mean {real_wk:+.4f}% vs '\n"
            "      f'null {nm:+.4f}% +/- {ns:.4f}%)')\n"
            "if p > 0.10:\n"
            "    print('-> the real book is INDISTINGUISHABLE from random labels. None.')\n"
            "else:\n"
            "    print('-> the real book beats the shuffled null.')"
        ),
        md(
            f"Placebo **p = {R['placebo_p']:.2f}** (and {R['placebo_p_seeds'][0]:.2f} / "
            f"{R['placebo_p_seeds'][1]:.2f} / {R['placebo_p_seeds'][2]:.2f} across seeds). "
            "The sort carries no information that random labels did not."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** One-sample *t* = {R['ls_t']:+.2f} (wrong sign), placebo "
            f"p = {R['placebo_p']:.2f}. The premium does not replicate on modern large-caps. "
            "None, not Weak -- there is no positive effect even to partially support, and the "
            "literature describes a far broader, older, less-watched universe.\n"
            f"- **Tradability -- MIRAGE.** Gross Sharpe {R['ls_sharpe']:+.2f}, net "
            f"{R['net_mean']:+.1f}%/yr, ~{R['turnover']:.0f}%/week turnover. Nothing to trade.\n"
            "- **Survivorship -- Named.** The biggest volume spikes belong to blow-ups and "
            "takeovers -- the names deleted from a survivor basket. Results are upper bounds, "
            "and the upper bound is already negative."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No -- on three counts:\n\n"
            f"1. **Wrong sign.** The book loses money gross ({R['ls_mean']:+.2f}%/yr) before a "
            "cent of cost.\n"
            f"2. **Brutal turnover.** A weekly quintile rebalance churns ~{R['turnover']:.0f}% "
            "of the long basket every week; at 5 bps/leg that compounds into a heavy drag, "
            f"taking the net to {R['net_mean']:+.1f}%/yr.\n"
            "3. **Capacity & crowding.** Even if it had worked, a one-week-horizon mega-cap "
            "signal would be the first thing arbitraged away."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader, point-in-time universe.** GKM used the full NYSE cross-section. The "
            "premium is documented as strongest where investor-recognition frictions bite -- "
            "small and mid-caps, *not* the most-watched mega-caps in our basket "
            "(Kaniel-Ozoguz-Starks 2012).\n"
            "- **Include delisted names.** A point-in-time membership list with failed names "
            "would let the natural volume-spike short candidates back in.\n"
            "- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the "
            "nearest risk-sort neighbour.\n"
            "- **[Study 418 -- Money-Flow-Index](../../418-money-flow-index/)**: the nearest "
            "volume-indicator neighbour.\n\n"
            "*Think the high-volume premium survives net of costs on a point-in-time universe "
            "with delisted names? Fork this, widen to small/mid-caps, and show t > 2 net. "
            "That is the bar.*"
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
            "# High-Volume-Return-Premium -- a quantitative teardown\n"
            "### abnormal-volume sort * weekly long-short * HAC inference * label-shuffle null\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survivorship--biased%3F: Named](https://img.shields.io/badge/Survivorship--biased%3F-Named-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. We test Gervais-Kaniel-Mingelgrin (2001): rank "
            "by abnormal weekly volume, long the top quintile, short the bottom quintile, hold "
            "the next week (one execution lag).\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted close + raw "
            f"volume, {R['n_tickers']} large-caps, {R['year_start']}-{R['year_end']}, as-of "
            "2026-06-26. Methods in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current large-caps projected "
            "backwards. All results are upper-bound estimates."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | LS **{R['ls_mean']:+.2f}%/yr**, one-sample *t* = "
            f"**{R['ls_t']:+.3f}** (wrong sign), placebo **p = {R['placebo_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['ls_sharpe']:+.3f}**, net "
            f"**{R['net_mean']:+.1f}%/yr**, turnover **~{R['turnover']:.0f}%/wk**. |\n"
            "| **Survivorship-biased?** | `NAMED` | 40 current large-caps projected "
            "backwards; upper-bound estimate. |\n\n"
            "> The GKM premium is documented on broad, older NYSE tapes. On 12 years of 40 "
            "much-watched mega-cap survivors it is the wrong sign and a statistical coin."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $V_{i,t}$ = name $i$'s average daily volume in week $t$ and "
            "$\\bar V_{i,t} = \\frac1{8}\\sum_{k=1}^{8} V_{i,t-k}$ its trailing norm. Define "
            "abnormal volume $a_{i,t} = V_{i,t}/\\bar V_{i,t} - 1$. GKM (2001) assert:\n\n"
            "- **H1 (signal).** Names in the top quintile of $a_{i,t}$ earn higher forward "
            "returns than the bottom quintile: $\\mathbb{E}[r^{\\text{high}}_{t+1} "
            "- r^{\\text{low}}_{t+1}] > 0$.\n"
            "- **H2 (visibility).** The driver is an attention/recognition shock, not a "
            "liquidity or risk premium.\n"
            "- **H3 (tradable).** The drift persists over a short horizon and survives costs.\n\n"
            "On our tape we **reject H1** (the long-short is the wrong sign, t=-1.34, placebo "
            "p=0.12) and therefore never reach H2/H3 -- there is no premium to attribute or trade."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "The high-volume premium is a cornerstone of the *attention-is-priced* literature. "
            "If it fails on liquid large-caps, that is itself informative: the recognition "
            "channel needs a stock that *can* gain visibility, and a mega-cap everyone already "
            "watches has little recognition headroom (Kaniel-Ozoguz-Starks 2012). The null here "
            "is consistent with the mechanism switching off precisely where attention is "
            "saturated."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Weekly abnormal volume $a_{i,t}=V_{i,t}/\\bar V_{i,t}-1$, trailing "
            "8-week mean, strictly past (`.shift(1)`).\n"
            "- **Ranking.** Each Friday, sort the cross-section by $a$; top 20% = high volume, "
            "bottom 20% = low volume.\n"
            "- **Book.** Long high-vol, short low-vol, equal-weight, dollar-neutral.\n"
            "- **Execution lag.** Enter the **next** week's compounded return (no same-bar fill).\n"
            "- **Inference.** Plain one-sample *t* (the bar) and Newey-West HAC *t* on the "
            "weekly long-short.\n"
            "- **Null.** Within-week label-shuffle placebo, 150-200 draws, seed-robust.\n"
            "- **Costs.** 5 bps/leg one-way + 50 bps/yr borrow on the short leg.\n"
            f"- **Universe caveat.** {R['n_tickers']} large-cap survivors (yfinance); "
            "survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector\n\n"
            "Sweep the planted premium from negative to positive on a synthetic panel. The "
            "recovered long-short mean should be monotone in the premium and zero at zero."
        ),
        code(
            "ctrl = st.sweep_synthetic()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['ls_mean_ann_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['ls_mean_ann_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['ls_mean_ann_%'], ctrl['t_os'])):\n"
            "    ax.text(i, m + (0.4 if m >= 0 else -1.2), f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted premium'); ax.set_ylabel('recovered LS mean (%/yr)')\n"
            "ax.set_title('Positive control: LS mean is monotone in the planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "### 4b -- Abnormal-volume distribution on the real tape\n\n"
            "What does the cross-section of abnormal volume look like? We expect a "
            "right-skewed distribution (spikes are large and rare)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    wk_ret, wk_vol = st.weekly_frames(ret, vol)\n"
            "    abn = st.abnormal_volume(wk_vol).iloc[-1].dropna()\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(abn, bins=20, color=AMBER, edgecolor='white', alpha=0.85)\n"
            "    ax.axvline(0, c=GREY, lw=1.5, ls=':')\n"
            "    ax.axvline(abn.median(), c=RED, lw=2, ls='--', label=f'median={abn.median():+.2f}')\n"
            "    ax.set_xlabel('abnormal volume (this week / trailing 8-wk - 1)')\n"
            "    ax.set_ylabel('count'); ax.set_title('Abnormal-volume cross-section (latest week)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'median {abn.median():+.3f} | range {abn.min():+.2f} .. {abn.max():+.2f}')\n"
            "else:\n"
            "    print('Frozen: see docs/results.md for the headline numbers.')"
        ),
        md("### 4c -- The quintile legs and the long-short"),
        code(
            "if HAVE_REAL:\n"
            "    ls_m, ls_t, ls_s = s_ls['mean']*100, s_ls['t_os'], s_ls['sharpe']\n"
            "    lo_m, hi_m = s_short['mean']*100, s_long['mean']*100\n"
            "    net_m, net_t = s_net['mean']*100, s_net['t_os']\n"
            "else:\n"
            "    ls_m, ls_t, ls_s = R['ls_mean'], R['ls_t'], R['ls_sharpe']\n"
            "    lo_m, hi_m = R['short_leg'], R['long_leg']\n"
            "    net_m, net_t = R['net_mean'], R['net_t']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].bar(['Low-vol\\n(short)', 'High-vol\\n(long)'], [lo_m, hi_m],\n"
            "            color=[GREY, AMBER], width=0.5)\n"
            "axes[0].set_ylabel('forward 1-wk return (%/yr)')\n"
            "axes[0].set_title('Quintile legs: quiet names edge ahead')\n"
            "axes[1].bar(['gross', 'net'], [ls_m, net_m], color=[RED, RED], width=0.5)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('long-short mean (%/yr)')\n"
            "axes[1].set_title(f'LS gross t={ls_t:+.2f} | net t={net_t:+.2f} (wrong sign)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LS gross {ls_m:+.2f}%/yr (Sharpe {ls_s:+.3f}, t {ls_t:+.3f}) | '\n"
            "      f'net {net_m:+.2f}%/yr (t {net_t:+.3f})')"
        ),
        md(
            "### 4d -- The label-shuffle null distribution\n\n"
            "Permute the volume labels across names within each week, 200 times, and build the "
            "null distribution of the mean long-short. Where does the real mean fall?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rm = float(ls['ls_gross'].mean())\n"
            "    abn_mat, fwd_mat = st._signal_and_forward(ret, vol, 1, st.TRAIL_WEEKS, st.MIN_NAMES)\n"
            "    rng = np.random.default_rng(512); draws = []\n"
            "    for _ in range(200):\n"
            "        sh = abn_mat.copy()\n"
            "        for w in range(sh.shape[0]):\n"
            "            v = ~np.isnan(sh[w]); vals = sh[w, v]; rng.shuffle(vals); sh[w, v] = vals\n"
            "        draws.append(np.nanmean(st._long_short_means(sh, fwd_mat, st.QUINTILE)))\n"
            "    draws = np.array(draws)*100; rm100 = rm*100\n"
            "    p = float((np.abs(draws) >= abs(rm100)).mean())\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(draws, bins=30, color=GREY, edgecolor='white', alpha=0.8)\n"
            "    ax.axvline(rm100, c=RED, lw=2.5, label=f'real mean {rm100:+.4f}% (p={p:.2f})')\n"
            "    ax.axvline(0, c='k', lw=1, ls=':')\n"
            "    ax.set_xlabel('shuffled weekly LS mean (%)'); ax.set_ylabel('count')\n"
            "    ax.set_title('Label-shuffle null: real book sits inside the noise')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'placebo p = {p:.3f}')\n"
            "else:\n"
            "    print(f'Frozen placebo p = {R[\"placebo_p\"]:.3f} '\n"
            "          f'(seeds {R[\"placebo_p_seeds\"]})')"
        ),
        md(
            "### 4e -- Horizon robustness\n\n"
            "Does a longer hold rescue it? The GKM story says the drift plays out over a few "
            "weeks; if the sign is wrong it should *worsen* with horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in (1, 2, 4, 8):\n"
            "        dh = st.long_short(ret, vol, horizon=h)\n"
            "        s = st.summary(dh['ls_gross'])\n"
            "        rows.append({'horizon_wk': h, 'mean_%/yr': s['mean']*100,\n"
            "                     't_os': s['t_os'], 'sharpe': s['sharpe']})\n"
            "    hz = pd.DataFrame(rows)\n"
            "else:\n"
            "    hz = pd.DataFrame([{'horizon_wk': h, 'mean_%/yr': v[0], 't_os': v[1], 'sharpe': v[2]}\n"
            "                       for h, v in R['horizon'].items()])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hz['horizon_wk'], hz['mean_%/yr'], 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('holding horizon (weeks)'); ax.set_ylabel('LS mean (%/yr)')\n"
            "ax.set_title('Longer holds make the (negative) drift LARGER -- opposite of GKM')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(hz.round(3).to_string(index=False))"
        ),
        md(
            "### 4f -- Equity curve and turnover\n\n"
            "The cumulative long-short and the weekly turnover that drives the cost drag."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + ls['ls_gross']).cumprod()\n"
            "    dd = (eq/eq.cummax() - 1)*100\n"
            "    turn = st.avg_turnover(ret, vol, horizon=1)\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    a1.plot(ls.index, eq.values, c=RED, lw=1.5); a1.axhline(1, c='k', lw=1, ls='--')\n"
            "    a1.set_ylabel('cumulative wealth'); a1.set_title('Long-short equity curve (grinds down)')\n"
            "    a2.fill_between(ls.index, dd.values, 0, color=RED, alpha=0.35)\n"
            "    a2.set_ylabel('drawdown (%)'); a2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'avg one-sided long-leg turnover: {turn*100:.1f}%/week | max DD {dd.min():.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: turnover ~{R[\"turnover\"]:.1f}%/wk | max DD {R[\"ls_dd\"]:.1f}%')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- LS {R['ls_mean']:+.2f}%/yr, one-sample *t* = "
            f"{R['ls_t']:+.3f} (wrong sign), placebo p = {R['placebo_p']:.2f} (seed-robust "
            f"{R['placebo_p_seeds']}). The premium does not replicate on modern large-caps.\n"
            f"- **Tradability `MIRAGE`** -- Sharpe {R['ls_sharpe']:+.3f}, net "
            f"{R['net_mean']:+.1f}%/yr, ~{R['turnover']:.0f}%/wk turnover, max DD "
            f"{R['ls_dd']:.1f}%. Even a right-signed book would drown in churn.\n"
            "- **Survivorship `NAMED`** -- the biggest volume spikes belong to the delisted "
            "names a survivor basket removes. Magnitude is an upper bound (already negative)."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "No. Beyond the wrong sign:\n\n"
            f"1. **Turnover.** ~{R['turnover']:.0f}% of the long basket changes every week. At "
            "5 bps/leg the cost line alone is several percent a year.\n"
            "2. **Short borrow.** Shorting the quiet names is cheap but not free; charged at "
            "50 bps/yr here.\n"
            "3. **Decay & crowding.** A 1-week mega-cap signal is the first thing competed away; "
            "McLean-Pontiff (2016) document ~32% post-publication attenuation on average, and "
            "GKM is a 2001 paper."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Small/mid-caps.** The recognition channel needs visibility headroom; the "
            "premium is documented as strongest away from the most-watched names.\n"
            "- **Point-in-time universe with delisted names.** Re-admit the blow-up / takeover "
            "names that trade on the biggest spikes -- the natural shorts.\n"
            "- **Daily-formation variant.** GKM also test day-formation; a daily sort changes "
            "the turnover/cost trade-off.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)** and "
            "**[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the "
            "same cross-sectional-sort infrastructure on risk signals.\n"
            "- **[Study 418 -- Money-Flow-Index](../../418-money-flow-index/)**: the nearest "
            "volume-indicator neighbour.\n\n"
            "*Think the high-volume premium survives net of costs on a point-in-time universe "
            "with delisted names? Fork this, widen to small/mid-caps, and show t > 2 net. "
            "That is the bar.*"
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
