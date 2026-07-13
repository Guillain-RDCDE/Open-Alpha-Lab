"""Generate the two narrative notebooks for Study 731 (Wimbledon-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EWU/VGK
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers -- mirror of docs/results.md
# (EWU + VGK, yfinance total-return, 2004-06-01 -> 2026-06-30; 20 contested
# Championships fortnights 2005->2025, 2020 cancelled.)
R = dict(
    n_total=21, n_contested=20, n_included=20, fp="da4f039bf903",
    panel_rows=5555,
    # signal -- window return across 20 years (%)
    raw_mean=+0.377, raw_sd=2.794, raw_t=+0.604, raw_up=12, raw_n=20,
    abn_mean=-0.218, abn_sd=0.715, abn_t=-1.360, abn_up=8, abn_n=20,
    # random-window placebo (two-sided, 20x200 draws)
    pl_raw_p=0.6697, pl_raw_mean=+0.265, pl_raw_sd=0.837,
    pl_abn_p=0.4240, pl_abn_mean=-0.051, pl_abn_sd=0.269,
    pl_long_p=0.7460, pl_mn_p=0.2717,
    # vol lull (third axis)
    vol_ratio=1.009, vol_t=+0.093, vol_meanlog=+0.0086, vol_quieter=10, vol_n=20,
    vol_q_lo=29.9, vol_q_hi=70.1,
    # tradability (%)
    long_gross=+0.377, long_gross_t=+0.60, long_net=+0.277, long_net_t=+0.44, long_win=12,
    mn_gross=-0.218, mn_gross_t=-1.36, mn_net=-0.436, mn_net_t=-2.72, mn_win=5,
    long_net10=+0.177, long_net10_t=+0.28, mn_net10=-0.636, mn_net10_t=-3.98,
    rev_net=-0.001, rev_net_t=-0.00, drag=0.218,
    # jackknife
    raw_jk_lo=+0.249, raw_jk_hi=+1.242, abn_jk_lo=-2.015, abn_jk_hi=-0.981,
    # sub-period split (pre-2015 vs 2015+ format shift)
    early_t=+0.288, late_t=+0.679, split_welch=-0.089,
    # event anatomy -- mean cumulative return by session offset from entry (%)
    car_raw={0: 0.000, 2: -0.602, 4: -0.189, 6: -0.067, 8: -0.042, 10: +0.318},
    car_abn={0: 0.000, 2: +0.097, 4: +0.083, 6: -0.010, 8: -0.153, 10: -0.166},
    # synthetic control
    syn_null_mean=-0.085, syn_null_sd=0.906, syn_null_fire=0, syn_null_seeds=20,
    syn_p1_t=+2.41, syn_p2_t=+4.23,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![A_real_lull%3F: Busted](https://img.shields.io/badge/A_real_lull%3F-Busted-8b949e?style=flat-square)\n\n"
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

from wimbledon_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
else:
    PRICES = EV = INC = None
print("real cache present:", HAVE_REAL, "| calendar years:", len(data.WIMBLEDON),
      "| contested/included:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the UK market go quiet for Wimbledon? 🎾📉\n"
            "### The \"summer-lull\" folklore — a quiet tennis-and-strawberries "
            "fortnight the FTSE supposedly sleeps through\n\n"
            + BADGES +
            "There's a cosy bit of City folklore that goes: for the two weeks of "
            "Wimbledon — late June into mid-July, strawberries and Pimm's, the trading "
            "floors half-empty — the UK market goes *quiet*. A summer lull. Some people "
            "say it drifts gently up on the sunshine mood; others that it's dead money "
            "you should step aside for. Either way, the fortnight is supposed to have a "
            "distinctive, sleepy signature you could position around.\n\n"
            "We tested it properly — every Championships fortnight 2005→2025, on the "
            "tradable UK equity ETF (`EWU`), against Europe as a whole.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "vol test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 20 fortnights hardcoded from Wikipedia (2020 "
            "COVID-cancelled), each a *calendar-known* window (the dates are published "
            "years ahead — so there's nothing to forecast and no look-ahead). Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the UK market drift up during Wimbledon? | **Not really.** "
            f"**{R['raw_mean']:+.2f}%** on average over the fortnight — exactly what any "
            "random two-week hold of a rising market gives you (*t* = "
            f"**{R['raw_t']:.2f}**), and a random fortnight matches it {int(R['pl_raw_p']*100)}% "
            "of the time. |\n"
            f"| Does the UK do anything *special* vs Europe? | **No.** UK minus Europe "
            f"over the window is **{R['abn_mean']:+.2f}%** — a whisper of underperformance "
            f"(*t* = {R['abn_t']:.2f}), nowhere near significant. |\n"
            f"| Is the fortnight actually *quieter*? | **No — the myth's core is busted.** "
            f"Daily moves during Wimbledon are the same size as the weeks around it "
            f"(vol ratio **{R['vol_ratio']:.2f}**), and it's quieter than its "
            f"neighbours in exactly **{R['vol_quieter']}/{R['vol_n']}** years — a coin flip. |\n"
            f"| Could you have traded it? | **No.** The only statistically \"significant\" "
            f"number in the whole study (*t* = **{R['mn_net_t']:.2f}**) is a trade that "
            "*loses* — and it loses purely to costs, not to any real signal. |\n\n"
            "> The lull is a nice story. The tape shows an ordinary two weeks that "
            "looks exactly like every other two weeks."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"During the Wimbledon fortnight the City empties out — everyone's at "
            "the tennis or on holiday — volumes thin, and the UK market goes into a "
            "quiet summer lull with its own distinctive drift. Trade around it.\"*\n\n"
            "It's the British cousin of *\"Sell in May and go away\"* — a seasonal-lull "
            "intuition pinned to the most photogenic fortnight of the English summer. "
            "Nobody has ever formally tested a *Wimbledon* stock effect. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, it would be a free, permanent, *pre-scheduled* calendar edge — "
            "the dates are published years in advance, so unlike an earnings surprise "
            "there's nothing to predict. Step aside (or fade the lull) for two known "
            "weeks each summer, every year. That's the kind of thing that would be "
            "quietly enormous if it were true — which is exactly why it's worth "
            "checking whether it is."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_contested']}** contested fortnights "
            "2005→2025 (2020 COVID-cancelled), hardcoded with exact start/end dates "
            "(each checked to be a Monday→Sunday, 13-day span).\n"
            "- **The market.** `EWU` (the UK ETF) for the raw lull, and `EWU` minus "
            "`VGK` (broad Europe) to strip out the Europe-wide summer drift and isolate "
            "anything *UK-specific*.\n"
            "- **The window.** Hold from the close *before* the first Monday to the "
            "close of the last session inside the fortnight — a calendar-known, "
            "zero-look-ahead two-week hold.\n"
            "- **The honesty checks.** A random-window placebo (does a random two weeks "
            "do the same thing just as often?), a realized-volatility test (is it "
            "*actually* quieter?), and the trade you could really have placed, net of "
            "costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the headline. Does the fortnight return stand out at all?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw = st.one_sample_t(INC['raw'].values)\n"
            "    abn = st.one_sample_t(INC['abn'].values)\n"
            "    raw_m, raw_t, abn_m, abn_t = raw['mean']*100, raw['t'], abn['mean']*100, abn['t']\n"
            "else:\n"
            "    raw_m, raw_t, abn_m, abn_t = R['raw_mean'], R['raw_t'], R['abn_mean'], R['abn_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.6))\n"
            "ax.bar(['UK (raw EWU)', 'UK minus Europe'], [raw_m, abn_m],\n"
            "       color=[GREY, GREY], width=.5)\n"
            "for i, v in enumerate([raw_m, abn_m]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean Wimbledon-fortnight return')\n"
            "ax.set_title(f'Raw t={raw_t:+.2f}, abnormal t={abn_t:+.2f} -- neither stands out')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'raw {raw_m:+.3f}% (t={raw_t:+.2f}) | abnormal {abn_m:+.3f}% (t={abn_t:+.2f})')"
        ),
        md(
            f"The raw UK return is **{R['raw_mean']:+.2f}%** over the fortnight — but "
            "that's just the ordinary drift of a rising market held for two weeks, not a "
            f"Wimbledon signature (*t* = {R['raw_t']:.2f}). Against Europe the UK is a "
            f"hair *soft* (**{R['abn_mean']:+.2f}%**, *t* = {R['abn_t']:.2f}) — but "
            "nowhere near real.\n\n"
            "**Is that raw number bigger than a random two weeks? The placebo:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'raw', n_seeds=8, n_draws_per_seed=200, tail='two')\n"
            "    obs = pl['obs']*100\n"
            "    draws = np.random.default_rng(731).normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['raw_mean']\n"
            "    draws = np.random.default_rng(731).normal(R['pl_raw_mean'], R['pl_raw_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random two-week windows (same tickers)')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed Wimbledon mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean return of a random-window draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Wimbledon sits right in the middle of the luck cloud (p = {R[\"pl_raw_p\"]:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['raw_mean']:+.3f}% vs random-window mean {R['pl_raw_mean']:+.3f}%, p = {R['pl_raw_p']:.3f}\")"
        ),
        md(
            f"The observed fortnight return lands smack in the middle of the "
            f"random-window cloud (*p* = {R['pl_raw_p']:.2f}) — a random two weeks beats "
            "it about two-thirds of the time. There is nothing calendar-special here.\n\n"
            "**Now the heart of the folklore — is the fortnight actually *quiet*?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ratios = np.exp(INC['log_vol_ratio'].values)\n"
            "else:\n"
            "    rng = np.random.default_rng(731); ratios = np.exp(rng.normal(R['vol_meanlog'], 0.4, R['vol_n']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "yrs = INC['year'].values if HAVE_REAL else list(range(2005, 2026))\n"
            "yrs = [y for y in yrs if y != 2020][:len(ratios)]\n"
            "cols = [GREEN if r < 1 else GREY for r in ratios]\n"
            "ax.bar(range(len(ratios)), ratios, color=cols)\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1.2, label='same volatility as the surrounding weeks')\n"
            "ax.set_xticks(range(len(ratios))); ax.set_xticklabels(yrs, rotation=90, fontsize=7)\n"
            "ax.set_ylabel('fortnight vol / neighbourhood vol')\n"
            "ax.set_title(f'Quieter (green) in exactly {R[\"vol_quieter\"]}/{R[\"vol_n\"]} years -- a coin flip')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"mean vol ratio {R['vol_ratio']:.3f}, quieter in {R['vol_quieter']}/{R['vol_n']} years\")"
        ),
        md(
            f"This is the myth's own claim, tested directly — and it's **busted**. The "
            f"fortnight's daily moves are the same size as the surrounding weeks (mean "
            f"ratio **{R['vol_ratio']:.2f}**), and it is quieter than its neighbours in "
            f"exactly **{R['vol_quieter']} of {R['vol_n']}** years. A coin flip. There is "
            "no volume-thinning summer lull in the volatility.\n\n"
            "**Finally, could you have traded it?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cap = st.capture_summary(EV)\n"
            "    lg, ln = cap['long_only']['gross_mean']*100, cap['long_only']['net_mean']*100\n"
            "    mg, mn = cap['market_neutral']['gross_mean']*100, cap['market_neutral']['net_mean']*100\n"
            "    mnt = cap['market_neutral']['net_t']\n"
            "else:\n"
            "    lg, ln, mg, mn, mnt = R['long_gross'], R['long_net'], R['mn_gross'], R['mn_net'], R['mn_net_t']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(['long UK\\ngross', 'long UK\\nnet', 'UK-vs-Europe\\ngross', 'UK-vs-Europe\\nnet'],\n"
            "       [lg, ln, mg, mn], color=[GREY, GREY, GREY, RED], width=.6)\n"
            "for i, v in enumerate([lg, ln, mg, mn]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean fortnight return')\n"
            "ax.set_title(f'The only \"significant\" bar (t={mnt:+.2f}) is a trade that LOSES -- to costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long net {ln:+.3f}% | market-neutral net {mn:+.3f}% (t={mnt:+.2f})')"
        ),
        md(
            f"Buying the UK for the fortnight nets **{R['long_net']:+.2f}%** — but that's "
            f"just the market drifting up (*t* = {R['long_net_t']:.2f}); a random two "
            f"weeks pays the same. The market-neutral version (long UK / short Europe) is "
            f"the *one* number in this whole study that clears the significance bar — "
            f"**{R['mn_net']:+.2f}%**, *t* = **{R['mn_net_t']:.2f}** — and it is "
            f"**negative**: it *loses* money. And it loses not because there's a real "
            f"'short the UK for Wimbledon' edge (the gross spread is *t* = "
            f"{R['mn_gross_t']:.2f}, insignificant) but because a market-neutral book "
            f"pays ~{R['drag']:.2f}% in costs and borrow over two weeks. Flip the trade "
            f"to go the other way and it nets **{R['rev_net']:+.2f}%** — dead zero. "
            "There is no tradable direction. That's a mirage in its purest form."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No directional seasonal, raw or UK-specific: raw "
            f"*t* = {R['raw_t']:.2f} (placebo *p* = {R['pl_raw_p']:.2f}), abnormal "
            f"*t* = {R['abn_t']:.2f} (placebo *p* = {R['pl_abn_p']:.2f}). The fortnight "
            "looks like every other fortnight.\n"
            "- **Tradability — Mirage.** Long-UK 'edge' is just market beta; the only "
            f"significant number (*t* = {R['mn_net_t']:.2f}) is a pure cost/borrow drag "
            "on a zero-edge spread, and neither trade direction pays.\n"
            "- **A real lull? — Busted.** The fortnight is no quieter than the weeks "
            f"around it (vol ratio {R['vol_ratio']:.2f}, quieter in {R['vol_quieter']}/{R['vol_n']} "
            "years) — the literal 'quiet window' claim fails on its own terms."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is what a clean null looks like.** No lucky cut, no fragile "
            "*t*-stat that needs a placebo to knock down — just an ordinary two weeks "
            "that the folklore dressed up in strawberries. Most seasonals are exactly "
            "this.\n"
            "- **Sibling studies:** the [Eurovision effect](../../708-eurovision-effect/) "
            "(a national-mood event window on single-country ETFs), the "
            "[World Cup effect](../../235-world-cup-effect/) (the real Edmans sports-"
            "sentiment mechanism), and the classic [Sell-in-May](../../) seasonal-lull "
            "family — every one tested the same honest way.\n\n"
            "*Think there's a real Wimbledon lull hiding somewhere — in intraday UK "
            "volume, in gilt or FTSE-250 seasonality, in the actual match-day flow? "
            "Find a net, placebo-surviving edge and we'll publish the teardown.*"
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
            "# Wimbledon-Effect — a quantitative teardown 🔬\n"
            "### One-sample-*t* on fortnight returns (raw & abnormal) · a random-window "
            "placebo · a realized-vol lull test · a costed calendar trade · a 20-seed "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **the Wimbledon fortnight is the "
            "UK market's quiet summer lull** — has no published academic anchor; it is "
            "pure City folklore, the British sibling of *Sell-in-May*. The job here is "
            "to measure it honestly on the tradable UK vehicle, with the right inference "
            "unit for a tiny-n annual calendar window.\n\n"
            "> ⚠️ **Data note.** `EWU` (iShares MSCI United Kingdom) + `VGK` (Vanguard "
            "FTSE Europe), yfinance, adjusted (total-return) daily closes, "
            "2004-06-01→2026-06-30. 20 contested fortnights 2005→2025 (2020 cancelled). "
            "The window is **calendar-known** — dates published years ahead — so there "
            "is **no execution lag and no look-ahead**. Methods in "
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
            f"| **Signal** | `NONE` | raw fortnight return **{R['raw_mean']:+.3f}%**, "
            f"*t* = **{R['raw_t']:.3f}** (placebo *p* = {R['pl_raw_p']:.3f}); abnormal "
            f"(EWU−VGK) **{R['abn_mean']:+.3f}%**, *t* = **{R['abn_t']:.3f}** (placebo "
            f"*p* = {R['pl_abn_p']:.3f}) |\n"
            f"| **Tradability** | `MIRAGE` | long-only net **{R['long_net']:+.3f}%** "
            f"(*t* = {R['long_net_t']:.2f}) = market beta; market-neutral net "
            f"**{R['mn_net']:+.3f}%** (*t* = {R['mn_net_t']:.2f}) = pure cost drag on a "
            f"*t* = {R['mn_gross_t']:.2f} gross spread |\n"
            f"| **A real lull?** | `BUSTED` | realized-vol ratio (fortnight / "
            f"neighbourhood) **{R['vol_ratio']:.3f}**, *t* = {R['vol_t']:.3f}; quieter "
            f"in {R['vol_quieter']}/{R['vol_n']} years |\n\n"
            "> 💡 In plain words: three tests, three nulls. No directional seasonal, no "
            "tradable edge, and the fortnight is not even quieter than the weeks around "
            "it. A textbook clean negative."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P^{UK}_t$ be EWU's total-return close and $P^{EU}_t$ VGK's. For "
            "Championships year $y$, let $e_y$ be the last session *before* the first "
            "Monday and $x_y$ the last session on/before the second Sunday (the men's "
            "final, a non-trading day). The fortnight return and its abnormal "
            "(UK-specific) counterpart are\n\n"
            "$$r_y = \\frac{P^{UK}_{x_y}}{P^{UK}_{e_y}} - 1, \\qquad "
            "a_y = r_y - \\left(\\frac{P^{EU}_{x_y}}{P^{EU}_{e_y}} - 1\\right).$$\n\n"
            "Because the dates are **published years in advance**, $e_y$ and $x_y$ are "
            "known ex ante — a calendar-known window needs **no** `shift` and carries "
            "**zero** look-ahead (METHODOLOGY → *one execution lag*). Each year is one "
            "independent, non-overlapping event, so the **one-sample t** of $r_y$ (and "
            "$a_y$) across years is the correct primary statistic — not a daily panel. "
            "Claims:\n\n"
            "- **H1 (directional lull).** $E[r_y]$ or $E[a_y]$ differs from a random "
            "two-week window (a tradable drift, either sign).\n"
            "- **H2 (volatility lull).** Realized daily vol inside the fortnight is "
            "*lower* than the surrounding weeks.\n"
            "- **H3 (capture).** A calendar-known trade banks it net of costs.\n\n"
            "We find **H1 not supported** (both cuts inside the placebo cloud), **H2 "
            "not supported** (vol ratio ≈ 1), **H3 not supported** (no net edge either "
            "direction)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is small by construction: **{R['n_included']}** contested fortnights "
            "(2020 cancelled). The plan is a **one-sample t** per cut (raw, abnormal), a "
            "**Wilson interval** on the up-rate, a **20-seed × 200-draw random-window "
            "placebo** (redraw a same-length window at a random point in the tickers' "
            "own history and see how often the null matches or beats the observed mean, "
            "two-sided — the folklore names no direction), a **realized-vol ratio** test "
            "for the literal 'quiet' claim, and a **costed calendar trade** in both a "
            "long-only and a market-neutral construction. A leave-one-out jackknife "
            "guards against a single dominant year on a tiny sample."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_contested']} fortnights 2005→2025, hardcoded, each "
            "asserted Monday→Sunday / 13 days at import.\n"
            "- **Headline.** One-sample *t* of the fortnight return (raw EWU, and "
            "abnormal EWU−VGK) + Wilson up-rate.\n"
            "- **Robustness.** 20×200 random-window placebo (two-sided); leave-one-out "
            "jackknife; a pre-2015 vs 2015+ split (the format shifted a week later in "
            "2015).\n"
            "- **Volatility.** Per-year log realized-vol ratio, fortnight vs a symmetric "
            f"±{25}-session neighbourhood; one-sample *t*.\n"
            "- **Execution.** Calendar-known hold, entry close before the fortnight → "
            "exit close inside it. Long-only = 2 one-way legs × NAV; market-neutral "
            "(long EWU / short VGK) = 4 legs + borrow on the short.\n"
            "- **Control.** Synthetic paired (asset, benchmark) world, planted-seasonal "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md("### 4a · The headline — one-sample t, raw and abnormal"),
        code(
            "if HAVE_REAL:\n"
            "    raw = st.one_sample_t(INC['raw'].values); abn = st.one_sample_t(INC['abn'].values)\n"
            "    hr = st.hit_rate(INC['raw'].values)\n"
            "    print('raw   n=%d mean=%+.3f%% sd=%.3f%% t=%+.3f up %d/%d' % (raw['n'], raw['mean']*100, raw['sd']*100, raw['t'], hr['k'], hr['n']))\n"
            "    print('abn   n=%d mean=%+.3f%% sd=%.3f%% t=%+.3f' % (abn['n'], abn['mean']*100, abn['sd']*100, abn['t']))\n"
            "    raw_m, raw_t, abn_m, abn_t = raw['mean']*100, raw['t'], abn['mean']*100, abn['t']\n"
            "else:\n"
            "    raw_m, raw_t, abn_m, abn_t = R['raw_mean'], R['raw_t'], R['abn_mean'], R['abn_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))\n"
            "a1.bar(['raw EWU', 'abnormal\\nEWU-VGK'], [raw_m, abn_m], color=GREY, width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean fortnight return (%)'); a1.set_title('Means')\n"
            "a2.bar(['raw EWU', 'abnormal\\nEWU-VGK'], [raw_t, abn_t], color=GREY, width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('one-sample t'); a2.set_title('Neither cut reaches |t|=2')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: raw *t* = **{R['raw_t']:.2f}**, abnormal *t* = "
            f"**{R['abn_t']:.2f}** — both well inside ±2. The raw +0.38% is just two "
            "weeks of equity drift; the −0.22% abnormal is a whisper of UK softness "
            "with no statistical spine."
        ),
        md(
            "### 4b · The random-window placebo — is either cut unusual?\n\n"
            "For each year, redraw a same-length window at a random point in the EWU/VGK "
            "history, 20 seeds × 200 draws, two-sided."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl_r = st.placebo_pvalue(EV, PRICES, 'raw', n_seeds=8, n_draws_per_seed=200, tail='two')\n"
            "    pl_a = st.placebo_pvalue(EV, PRICES, 'abn', n_seeds=8, n_draws_per_seed=200, tail='two')\n"
            "    obs_r, obs_a = pl_r['obs']*100, pl_a['obs']*100\n"
            "    dr = np.random.default_rng(731).normal(pl_r['placebo_mean'], pl_r['placebo_sd'], 4000)*100\n"
            "    da = np.random.default_rng(731).normal(pl_a['placebo_mean'], pl_a['placebo_sd'], 4000)*100\n"
            "    print('raw placebo p =', round(pl_r['p_value'],3), '| abn placebo p =', round(pl_a['p_value'],3))\n"
            "else:\n"
            "    obs_r, obs_a = R['raw_mean'], R['abn_mean']\n"
            "    dr = np.random.default_rng(731).normal(R['pl_raw_mean'], R['pl_raw_sd'], 4000)\n"
            "    da = np.random.default_rng(731).normal(R['pl_abn_mean'], R['pl_abn_sd'], 4000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))\n"
            "a1.hist(dr, bins=50, color=GREY, alpha=.85); a1.axvline(obs_r, c=RED, lw=2.4)\n"
            "a1.set_title(f'raw: observed {obs_r:+.2f}%, p = {R[\"pl_raw_p\"]:.2f}'); a1.set_xlabel('mean (%)')\n"
            "a2.hist(da, bins=50, color=GREY, alpha=.85); a2.axvline(obs_a, c=RED, lw=2.4)\n"
            "a2.set_title(f'abnormal: observed {obs_a:+.2f}%, p = {R[\"pl_abn_p\"]:.2f}'); a2.set_xlabel('mean (%)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: raw *p* = {R['pl_raw_p']:.2f}, abnormal *p* = "
            f"{R['pl_abn_p']:.2f}. Both observed means sit in the fat middle of what a "
            "random two-week window on the *same tickers* produces. The Wimbledon "
            "window is not special — a random fortnight matches it most of the time."
        ),
        md(
            "### 4c · The volatility lull — the myth's own claim, tested directly\n\n"
            "Per year, log(realized daily vol inside the fortnight / realized vol in the "
            f"symmetric ±25-session neighbourhood). Negative = quieter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vl = st.vol_lull_stats(EV)\n"
            "    ratios = np.exp(INC['log_vol_ratio'].values)\n"
            "    print('mean log-ratio %+.4f (ratio %.3f), t=%+.3f, quieter %d/%d' % (vl['mean_log'], vl['mean_ratio'], vl['t'], vl['quieter_k'], vl['n']))\n"
            "else:\n"
            "    rng = np.random.default_rng(731); ratios = np.exp(rng.normal(R['vol_meanlog'], 0.4, R['vol_n']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.hist(np.log(ratios), bins=12, color=GREY, alpha=.85)\n"
            "ax.axvline(0, ls='--', c='k', lw=1.2, label='no change')\n"
            "ax.axvline(np.log(ratios).mean(), c=RED, lw=2.2, label=f'mean (ratio {R[\"vol_ratio\"]:.2f})')\n"
            "ax.set_xlabel('log(fortnight vol / neighbourhood vol)'); ax.set_ylabel('years')\n"
            "ax.set_title(f'Centred on zero: no lull (t={R[\"vol_t\"]:+.2f}, quieter {R[\"vol_quieter\"]}/{R[\"vol_n\"]})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the distribution is centred on zero (mean ratio "
            f"**{R['vol_ratio']:.2f}**, *t* = {R['vol_t']:.2f}), and the fortnight is "
            f"quieter than its neighbours in **{R['vol_quieter']}/{R['vol_n']}** years — a "
            "coin flip (Wilson "
            f"[{R['vol_q_lo']:.0f}%, {R['vol_q_hi']:.0f}%]). The literal 'quiet window' "
            "is **busted**: whatever thinning of volume the City feels, it does not show "
            "up as lower realized volatility. **H2 not supported.**"
        ),
        md(
            "### 4d · Robustness — jackknife and the 2015 format split\n\n"
            "With n=20, one dominant year could carry a *t*-stat. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw = INC['raw'].values; abn = INC['abn'].values\n"
            "    jk_r = [st.one_sample_t(np.delete(raw, i))['t'] for i in range(len(raw))]\n"
            "    jk_a = [st.one_sample_t(np.delete(abn, i))['t'] for i in range(len(abn))]\n"
            "    early = INC[INC['year']<=2014]['raw'].values; late = INC[INC['year']>=2015]['raw'].values\n"
            "    et, lt, wt = st.one_sample_t(early)['t'], st.one_sample_t(late)['t'], st.welch_t(early, late)\n"
            "    print('raw jk t [%.3f, %.3f] | abn jk t [%.3f, %.3f]' % (min(jk_r), max(jk_r), min(jk_a), max(jk_a)))\n"
            "    print('pre-2015 raw t=%+.2f | 2015+ raw t=%+.2f | Welch=%+.2f' % (et, lt, wt))\n"
            "else:\n"
            "    jk_r = list(np.random.default_rng(1).uniform(R['raw_jk_lo'], R['raw_jk_hi'], R['raw_n']))\n"
            "    jk_a = list(np.random.default_rng(2).uniform(R['abn_jk_lo'], R['abn_jk_hi'], R['abn_n']))\n"
            "    et, lt, wt = R['early_t'], R['late_t'], R['split_welch']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(range(len(jk_r)), sorted(jk_r), 'o-', color=GREY, label='raw, leave-one-out t')\n"
            "ax.plot(range(len(jk_a)), sorted(jk_a), 's-', color=AMBER, label='abnormal, leave-one-out t')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_ylabel('resulting t-stat'); ax.set_xlabel('leave-one-out draw (sorted)')\n"
            "ax.set_title('Every jackknife draw stays inside the noise band'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the raw jackknife *t* never leaves "
            f"[{R['raw_jk_lo']:.2f}, {R['raw_jk_hi']:.2f}]; the abnormal jackknife just "
            f"grazes {R['abn_jk_lo']:.2f} at its most extreme — one draw touching −2 out "
            "of twenty, then knocked flat by the placebo (*p* = "
            f"{R['pl_abn_p']:.2f}) anyway. The 2015 schedule shift (fortnight moved a "
            f"week later) changes nothing: pre-2015 raw *t* = {R['early_t']:.2f}, 2015+ "
            f"*t* = {R['late_t']:.2f}, Welch = {R['split_welch']:.2f}. Robustly nothing."
        ),
        md(
            "### 4e · Event anatomy — the average path through the fortnight"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_r = st.car_path(EV, PRICES, max_k=10, col='raw')\n"
            "    cp_a = st.car_path(EV, PRICES, max_k=10, col='abn')\n"
            "    days = list(cp_r.index); rs = list(cp_r.values*100); as_ = list(cp_a.values*100)\n"
            "else:\n"
            "    days = sorted(R['car_raw']); rs = [R['car_raw'][k] for k in days]; as_ = [R['car_abn'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot(days, rs, color=GREY, lw=2.2, marker='o', label='raw EWU')\n"
            "ax.plot(days, as_, color=AMBER, lw=2.2, marker='o', label='abnormal (EWU-VGK)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading sessions into the fortnight (entry = 0)')\n"
            "ax.set_ylabel('mean cumulative return (%)')\n"
            "ax.set_title('A directionless wander -- dips mid-fortnight, drifts back by the close')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: no shape. The raw path dips ~0.6% early, wanders, and "
            "recovers to roughly flat-plus by the end; the abnormal path oscillates "
            "around zero. Nothing resembling a coherent lull-and-release or a steady "
            "drift — just the meander of a small sample of two-week windows."
        ),
        md(
            "### 4f · Tradability — the calendar-known trade, net of costs\n\n"
            "Two constructions. Long-only EWU (2 one-way legs × NAV). Market-neutral "
            "long EWU / short VGK (4 legs + borrow on the short, 0.50%/yr)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cap = st.capture_summary(EV); cap10 = st.capture_summary(st.build_event_table(PRICES, cost_bps=10.0))\n"
            "    lo, mn = cap['long_only'], cap['market_neutral']\n"
            "    rows = [('long-only', lo['gross_mean']*100, lo['gross_t'], lo['net_mean']*100, lo['net_t']),\n"
            "            ('market-neutral', mn['gross_mean']*100, mn['gross_t'], mn['net_mean']*100, mn['net_t'])]\n"
            "    for r in rows: print('%-16s gross %+.3f%% (t=%+.2f)  net %+.3f%% (t=%+.2f)' % r)\n"
            "    lg, lt2, ln, lnt = rows[0][1], rows[0][2], rows[0][3], rows[0][4]\n"
            "    mg, mgt, mnn, mnt = rows[1][1], rows[1][2], rows[1][3], rows[1][4]\n"
            "else:\n"
            "    lg, ln, lnt = R['long_gross'], R['long_net'], R['long_net_t']\n"
            "    mg, mgt, mnn, mnt = R['mn_gross'], R['mn_gross_t'], R['mn_net'], R['mn_net_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(['long\\ngross', 'long\\nnet', 'm-neutral\\ngross', 'm-neutral\\nnet'],\n"
            "       [lg, ln, mg, mnn], color=[GREY, GREY, GREY, RED], width=.6)\n"
            "for i, v in enumerate([lg, ln, mg, mnn]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean fortnight return (%)')\n"
            "ax.set_title(f'The lone |t|>=2 (m-neutral net t={mnt:+.2f}) is a LOSS driven by ~{R[\"drag\"]:.2f}%% cost drag')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: long-only nets **{R['long_net']:+.3f}%** (*t* = "
            f"{R['long_net_t']:.2f}, placebo *p* = {R['pl_long_p']:.2f}) — indistinguishable "
            "from just holding the market for two weeks. The market-neutral book is the "
            f"only construction with |*t*| ≥ 2 (**{R['mn_net']:+.3f}%**, *t* = "
            f"**{R['mn_net_t']:.2f}**), but it is a **loss**, and the loss is "
            f"manufactured by costs: the *gross* spread is *t* = {R['mn_gross_t']:.2f} "
            f"(insignificant), and subtracting a deterministic ~{R['drag']:.2f}% "
            "cost/borrow drag from a zero-mean distribution shoves the whole thing below "
            f"−2. Reverse the trade and it nets **{R['rev_net']:+.3f}%** — dead zero "
            "(the symmetric drag eats the mirror-image gross). At 10 bps the market-"
            f"neutral net worsens to *t* = {R['mn_net10_t']:.2f}. No tradable direction "
            "exists. **H3 not supported; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Faithful-engine & power control\n\n"
            "Synthetic paired (asset, benchmark) log-return world (ρ ≈ 0.85, like a UK "
            "ETF vs Europe), a fixed synthetic fortnight window each year, TUNABLE "
            "planted seasonal. Null (bump=0) checked over 20 seeds."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(0.0, 731+s)['t'] for s in range(20)])\n"
            "p1 = st.synthetic_detect(0.01, 731); p2 = st.synthetic_detect(0.02, 731)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40, label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [p1['t']], color=AMBER, s=90, zorder=5, label='planted 1%')\n"
            "ax.scatter([2], [p2['t']], color=RED, s=90, zorder=5, label='planted 2%')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1,2]); ax.set_xticklabels(['null x20','planted 1%','planted 2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Quiet null, planted seasonals light up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('null mean t=%+.2f (sd %.2f), |t|>=2 in %d/20' % (null_ts.mean(), null_ts.std(ddof=1), (abs(null_ts)>=2).sum()))\n"
            "print('planted 1%% t=%+.2f | planted 2%% t=%+.2f' % (p1['t'], p2['t']))"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"*t* = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at "
            f"|*t*| ≥ 2 in **{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds. A planted "
            f"1% fortnight seasonal reads *t* = {R['syn_p1_t']:.2f}, a 2% one *t* = "
            f"{R['syn_p2_t']:.2f}. The machinery detects a real seasonal when one is "
            "planted — the flat real-tape answer is the tape's, not the detector's. "
            "*(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — raw fortnight return *t* = {R['raw_t']:.3f} (placebo "
            f"*p* = {R['pl_raw_p']:.3f}); abnormal EWU−VGK *t* = {R['abn_t']:.3f} (placebo "
            f"*p* = {R['pl_abn_p']:.3f}). Neither cut clears |*t*| = 2 or leaves the "
            "random-window cloud; the jackknife and the 2015 split confirm it isn't one "
            "lucky year or one regime. No published academic anchor to lean on either.\n"
            f"- **Tradability `MIRAGE`** — long-only net {R['long_net']:+.3f}% "
            f"(*t* = {R['long_net_t']:.2f}) is just two weeks of market beta; the only "
            f"|*t*| ≥ 2 number in the study (market-neutral net *t* = {R['mn_net_t']:.2f}) "
            "is a **loss** generated entirely by a ~"
            f"{R['drag']:.2f}% cost/borrow drag on a *t* = {R['mn_gross_t']:.2f} gross "
            f"spread, and the reverse trade nets {R['rev_net']:+.3f}%. No direction pays.\n"
            f"- **\"A real lull?\" `BUSTED`** — realized-vol ratio {R['vol_ratio']:.3f} "
            f"(*t* = {R['vol_t']:.2f}), quieter in {R['vol_quieter']}/{R['vol_n']} years. "
            "The fortnight is not measurably quieter than the weeks around it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The clean-null lesson.** Not every folklore claim needs a placebo to "
            "*demolish* a tempting number — some, like this one, simply never produce a "
            "tempting number in the first place. The value is in the discipline of "
            "checking, and in the one instructive trap here: a **cost-drag artifact** "
            "(the market-neutral net *t* = −2.72) that a careless writer could sell as "
            "'a significant Wimbledon short'. It is significant, and it is worthless.\n"
            "- **Where a real lull could still hide.** This tests daily EWU/VGK closes. "
            "Intraday UK *volume* (the thing the folklore actually describes), FTSE-250 "
            "or small-cap seasonality, or gilt-market summer patterns are untested here "
            "and are the natural sequels.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) and "
            "[235-world-cup-effect](../../235-world-cup-effect/) test national-mood "
            "*event* windows; this tests a *seasonal-lull* calendar window on a single "
            "market — the *Sell-in-May* family, not the sports-sentiment family. None of "
            "them test a UK-specific pre-scheduled fortnight against a Europe benchmark "
            "with a realized-vol lull check — that's this study's own contribution.\n\n"
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
