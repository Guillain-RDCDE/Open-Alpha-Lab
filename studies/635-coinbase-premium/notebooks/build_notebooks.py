"""Generate the two narrative notebooks for Study 635 (Coinbase Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
Coinbase+Binance daily closes under ../_cache/ and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Coinbase Exchange
# BTC-USD candles vs Binance BTCUSDT klines, 2017-08-17 -> 2026-06-30, 3,240 UTC days).
R = dict(
    start="2017-08-17", end="2026-06-30", asof="2026-06-30", n_days=3240, years=8.9,
    fingerprint="73bf2b138404",
    prem_mean=1.54, prem_sd=64.00, prem_min=-535.2, prem_max=1335.3, ac1=0.710,
    contemp_corr=0.077, contemp_t=1.91,
    # yearly: (year, mean bps, sd bps, share>0 %)
    yearly=[(2017, 107.37, 198.66, 76), (2018, -33.84, 110.90, 42),
            (2019, -4.45, 57.85, 54), (2020, 5.95, 14.16, 72),
            (2021, 5.94, 6.71, 88), (2022, -1.08, 7.04, 61),
            (2023, 3.19, 9.60, 75), (2024, -0.02, 6.82, 52),
            (2025, 0.43, 4.78, 60), (2026, -5.62, 6.28, 20)],
    neg2026=(145, 181),
    # predictive: (h, beta bps/sd, HAC t)
    pred_raw=[(1, 12.47, 1.42), (2, 23.08, 1.76), (3, 37.97, 2.09),
              (5, 53.36, 1.48), (7, 61.37, 1.16)],
    pred_wins=[(1, 20.81, 2.30), (2, 32.98, 2.22), (3, 46.99, 2.10),
               (5, 68.38, 1.71), (7, 90.72, 1.63)],
    pred_chg=[(1, 2.39, 0.22), (2, -3.22, -0.26), (3, -2.04, -0.17),
              (5, 12.19, 1.12), (7, 6.50, 0.62)],
    winsor_lo=-215.8, winsor_hi=178.7,
    split_rich=17.77, split_cheap=0.18, split_welch=1.40, n_rich=1533, n_cheap=1676,
    # overlay: (one-way bps, net bps/d, net HAC t, Sharpe, net ann %, alpha bps/d, alpha HAC t)
    overlay=[(5.0, 7.38, 1.67, 0.56, 30.9, 3.55, 1.13),
             (10.0, 6.34, 1.43, 0.48, 26.1, 2.52, 0.80),
             (25.0, 3.24, 0.73, 0.24, 12.6, -0.58, -0.18)],
    gross_bps=8.41, exposure=47.3, turnover=76,
    bh_bps=8.07, bh_sharpe=0.43, bh_ann=34.3,
    # z-window robustness at 10 bps: (window, net bps/d, net t, alpha bps/d, alpha t)
    zrobust=[(20, 10.63, 2.52, 6.68, 2.15), (63, 6.34, 1.43, 2.52, 0.80),
             (126, 8.54, 1.96, 4.58, 1.48)],
    rand_mean=-1.46, rand_sd=2.98, rand_t=-0.33, rand_seeds=20,
    # eras: (label, n, prem mean, prem sd, beta bps/sd, pred t, alpha bps/d, alpha t)
    eras=[("2017-2019", 867, 0.85, 122.83, 19.89, 1.21, 6.02, 0.86),
          ("2020-2021", 731, 5.95, 11.08, 10.43, 0.56, -6.04, -0.79),
          ("2022", 365, -1.08, 7.04, 4.95, 0.29, 7.14, 0.69),
          ("2023+", 1277, 0.23, 7.68, 29.15, 2.87, 1.23, 0.34),
          ("2024+ (ETF era)", 912, -0.95, 6.40, 16.90, 1.80, 4.24, 0.97)],
    slope_pre=0.175, slope_post=3.793, slope_ratio=22, slope_diff_t=2.72,
    # synthetic: (planted lead, beta bps/sd, HAC t)
    syn=[(0.0, 4.75, 0.75), (0.003, 34.05, 5.35)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Dead_since_the_ETFs%3F: Busted](https://img.shields.io/badge/Dead_since_the_ETFs%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from coinbase_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    DF = data.load_real()                          # pinned as-of; cache-first
    PREM = st.premium_bps(DF)
    RET = st.log_returns(DF)
else:
    DF = PREM = RET = None
print("real cb/bn cache present:", HAVE_REAL,
      "| days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When Coinbase pays more for Bitcoin than Binance… does it *mean* something? 🪙\n"
            "### The \"Coinbase premium\" — the famous footprint of the US institutional bid, tested honestly\n\n"
            + BADGES +
            "Same Bitcoin, two shops. On **Coinbase** (the big US exchange) it might cost $60,024; on "
            "**Binance** (the global one) $59,961 at the exact same second. That tiny gap — a few "
            "hundredths of a percent — is the **Coinbase premium**, and during the 2020-21 bull run it "
            "became crypto's favourite tea leaf: *\"Coinbase is trading rich — the American institutions "
            "are buying — get in before the wave.\"*\n\n"
            "It's a lovely story with a real mechanism behind it (someone buying billions in actual "
            "dollars really does push the dollar venue first). The question a desk has to ask is "
            "colder: **does today's premium tell you anything about tomorrow's price?**\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the winsorized robustness and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **One caveat up front.** The Binance side is priced in **USDT** (Tether), not dollars. "
            "When Tether itself wobbles, the \"premium\" moves with zero Coinbase flow — that's most of "
            "the wild ±2-5% swings of 2017-18. We name it and test around it."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Was the 2020-21 \"institutional bid\" real? | **Yes, as a footprint.** Coinbase sat rich on "
            "72-88% of days through 2020-21 — the folklore's *shape* is genuinely on the tape. |\n"
            "| Does the premium **predict** the next move? | **Barely.** The lean is always in the right "
            "direction, but at normal statistical standards it's a **whisper, not a signal** — about "
            "1.4 sigma where the desk demands 2. |\n"
            "| Can you trade it? | **No.** A rule that holds BTC only when Coinbase is rich earns less "
            "than just holding BTC, and its \"timing skill\" is indistinguishable from a coin-flip twin "
            "once you charge fees. |\n"
            "| Did the 2024 ETFs kill it? | **Surprise: no.** The premium *shrank* ~10-20× — but per unit "
            "of premium it now says **22× more** than it used to. Smaller footprint, sharper meaning. |\n\n"
            "> The honest verdict: a **real coincident indicator** — it tells you who is buying *right "
            "now* — dressed up by folklore as a crystal ball it never was."
        ),

        md(
            "## 1 · The claim\n\n"
            "> *\"US institutions clear their Bitcoin in actual dollars, mostly on Coinbase. When they "
            "buy size, Coinbase prints **above** Binance. Watch the premium and you front-run the "
            "wave.\"*\n\n"
            "This is the **Coinbase Premium Gap**, popularised by the analytics firm CryptoQuant during "
            "the Saylor/Tesla accumulation winter of 2020-21, and quoted in market wraps ever since. We "
            "rebuild it from the two exchanges' own public price feeds — **8.9 years, 3,240 days**, "
            "every day both venues printed a close at the same UTC instant.\n\n"
            "*(Cousin study, different object: [294-coinbase-rank](../../294-coinbase-rank/README.md) "
            "tested the Coinbase **app's App-Store ranking** — retail FOMO. This is the **exchange "
            "price gap** — the institutional footprint.)*"
        ),

        code(
            "# The premium, 2017 -> 2026 (daily, bps). The eye needs two scales: the wild early years\n"
            "# and the compressed modern era -- so we show both.\n"
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=False,\n"
            "                             gridspec_kw={'height_ratios': [1, 1]})\n"
            "    ax = axes[0]\n"
            "    ax.plot(PREM.index, PREM.values, color=GREY, lw=0.7)\n"
            "    ax.axhline(0, color='k', lw=0.8)\n"
            "    ax.set_title('Coinbase premium vs Binance, daily (bps) — full history')\n"
            "    ax.set_ylabel('bps')\n"
            "    ax = axes[1]\n"
            "    modern = PREM[PREM.index >= '2020-01-01']\n"
            "    ax.plot(modern.index, modern.values, color=GREY, lw=0.7)\n"
            "    roll = modern.rolling(30).mean()\n"
            "    ax.plot(roll.index, roll.values, color=RED, lw=1.8, label='30-day mean')\n"
            "    ax.axhline(0, color='k', lw=0.8)\n"
            "    ax.axvspan(pd.Timestamp('2020-10-01'), pd.Timestamp('2021-12-31'),\n"
            "               color=GREEN, alpha=0.10, label='the \"institutional bid\" era')\n"
            "    ax.set_title('Zoom 2020+ — the era the folklore celebrates (note the 10-20\\u00d7 smaller scale)')\n"
            "    ax.set_ylabel('bps')\n"
            "    ax.legend(loc='upper right')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing - see docs/results.md:', R['prem_mean'], 'bps mean,', R['prem_sd'], 'bps sd')\n"
        ),

        md(
            "## 2 · The regime story is *true*\n\n"
            "Year by year, the premium really does tell the story everyone remembers:\n\n"
            "- **2017-18** — chaos: swings of ±2-5%. Mostly **Tether noise**, not institutions (the "
            "Binance leg is USDT).\n"
            "- **2020-21** — the **institutional bid**: Coinbase rich on **72-88% of days**. Saylor, "
            "Tesla, the corporate-treasury wave — the footprint is unmistakable.\n"
            "- **2022+** — compression: arbitrage desks matured and the gap shrank to single-digit bps.\n"
            "- **2026 H1** — the mirror image: Coinbase **cheap on 145 of 181 days** (mean −5.6 bps). "
            "The US bid, on this tell, is *absent*.\n\n"
            "So the premium is a genuine **coincident** gauge. The folklore's money claim is stronger "
            "though: that it **leads**."
        ),

        code(
            "# Yearly mean premium -- the regime narrative in one bar chart (green rich, red cheap).\n"
            "yrs = [str(y) for y, m, s, p in R['yearly']]\n"
            "means = [m for y, m, s, p in R['yearly']]\n"
            "colors = [GREEN if m > 0 else RED for m in means]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "ax.bar(yrs, means, color=colors, width=0.62)\n"
            "ax.axhline(0, color='k', lw=0.8)\n"
            "for i, (y, m, s, p) in enumerate(R['yearly']):\n"
            "    va = 'bottom' if m >= 0 else 'top'\n"
            "    off = 2 if m >= 0 else -2\n"
            "    ax.annotate(f'{m:+.0f}', (i, m + off), ha='center', va=va, fontsize=9, color='dimgray')\n"
            "ax.set_title('Mean Coinbase premium by year (bps) — rich in the 2020-21 bid, cheap in 2026')\n"
            "ax.set_ylabel('bps (log-free, raw mean)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('2026 H1: Coinbase cheap on %d of %d days' % R['neg2026'])\n"
        ),

        md(
            "## 3 · Does it *lead*? The uncomfortable middle\n\n"
            "Take every day since 2017, note the premium at the close, and ask what Bitcoin did **the "
            "next day** (and over the next week). The answer is maddeningly *almost*:\n\n"
            "- Days when Coinbase was **rich** (vs its recent norm) were followed by **+17.8 bps** the "
            "next day on average; **cheap** days by just **+0.2 bps**. Sounds huge!\n"
            "- But Bitcoin's daily noise is ~±370 bps. Against that roar, the gap is only about **1.4 "
            "standard errors** — the desk's bar for \"real\" is 2. A gap this size shows up from pure "
            "luck roughly one backtest in six.\n"
            "- Cleaning the Tether-tantrum days sharpens it to ~2.3σ, and 2023+ alone reads 2.9σ — but "
            "when a signal only clears the bar *after* you choose the cleaning, the choice is doing "
            "some of the work.\n\n"
            "> 🔬 **For the quants:** HAC *t* = +1.42 raw at h = 1 (winsorized +2.30; h = 3 raw +2.09; "
            "1-day *change* predicts nothing). Full table in the quants notebook."
        ),

        code(
            "# Next-day BTC return after rich vs cheap premium days (trailing 63d z-score sign).\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "vals = [R['split_rich'], R['split_cheap']]\n"
            "labs = ['Coinbase RICH days\\n(z > 0)', 'Coinbase CHEAP days\\n(z \\u2264 0)']\n"
            "bars = ax.bar(labs, vals, color=[GREEN, GREY], width=0.5)\n"
            "for b, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:+.1f} bps', (b.get_x() + b.get_width()/2, v + 0.4),\n"
            "                ha='center', fontsize=11)\n"
            "ax.axhline(0, color='k', lw=0.8)\n"
            "ax.set_ylabel('mean next-day BTC return (bps)')\n"
            "ax.set_title('Fat-looking gap… at Welch t = %.2f — below the 2\\u03c3 bar' % R['split_welch'])\n"
            "plt.tight_layout(); plt.show()\n"
        ),

        md(
            "## 4 · Trying to trade it anyway\n\n"
            "We built the obvious strategy: **hold BTC only when Coinbase is rich** (signal read at "
            "tonight's close, position taken for tomorrow — one clean day of lag, no peeking), pay "
            "realistic exchange fees on every switch (~76 switches/yr).\n\n"
            "| fees (one-way) | strategy, per day | just holding BTC | the *timing* itself worth |\n"
            "|---|--:|--:|--:|\n"
            "| 5 bps | +7.4 bps | +8.1 bps | +3.6 bps (1.1σ) |\n"
            "| 10 bps | +6.3 bps | +8.1 bps | +2.5 bps (0.8σ) |\n"
            "| 25 bps | +3.2 bps | +8.1 bps | **−0.6 bps** (−0.2σ) |\n\n"
            "It **underperforms buy & hold** at every fee level, and the pure timing skill (after "
            "matching for how often it's in the market) never gets near significance — a **20-seed "
            "coin-flip twin** with the same market time earns ~0, so the machinery isn't hiding "
            "anything. The premium is a mirage as a trading signal.\n\n"
            "> 💡 It gets worse: the *one* z-score window that looks tradable (20-day, 2.15σ) is "
            "flanked by 0.8σ and 1.5σ at 63 and 126 days. When only one knob setting works, you found "
            "the knob, not a signal."
        ),

        md(
            "## 5 · The ETF twist — smaller, but *sharper*\n\n"
            "Since January 2024, US institutions can buy Bitcoin through **spot ETFs** (IBIT & friends) "
            "— they no longer need Coinbase to clear dollars. So the premium should be dead, right?\n\n"
            "**Wrong — and it's the best result on this tape.** The premium's *size* collapsed (sd "
            "~6-8 bps now vs 60-200 in the early years), but **per basis point it now predicts 22× "
            "more** than it did before 2023 — and that sharpening is itself statistically solid "
            "(2.7σ). The one era that clears the desk's bar on its own is… **2023 onwards** (2.9σ).\n\n"
            "The economics make sense: with mature arbitrage, any *residual* gap that survives is more "
            "likely to be real net flow rather than noise. The footprint shrank; its ink got darker.\n\n"
            "> 🔬 **For the quants:** pre-2023 slope +0.175 bps/bp vs 2023+ +3.793 bps/bp, HAC-SE "
            "difference *t* = +2.72. The 2024+ ETF-era slice alone is *t* = +1.80 — sharper but thin."
        ),

        md(
            "## 6 · Verdict\n\n"
            "| Axis | Stamp | In one line |\n|---|---|---|\n"
            "| Signal | **WEAK** 🟡 | Right direction everywhere, ~1.4σ where 2 is required; only "
            "spec-assisted slices clear the bar |\n"
            "| Tradability | **MIRAGE** 🔴 | Underperforms holding; timing skill ≈ coin-flip after "
            "fees; the good-looking window is a knob artefact |\n"
            "| \"Dead since the ETFs?\" | **BUSTED** ⚪ | Compressed 10-20× but 22× sharper per bp — "
            "shrunk, not dead |\n\n"
            "**Take-home:** the Coinbase premium is a *thermometer*, not a *forecast*. It faithfully "
            "shows who is bidding **today** — the 2020-21 institutional wave, the 2026 US absence — "
            "but the market has already moved by the time it prints. Folklore promoted a coincident "
            "gauge into a leading one; the tape declines the promotion.\n\n"
            "*Reproduce every number: `python examples/verify.py` — or go one level deeper in "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Coinbase Premium — the quant teardown 🪙\n\n"
            + BADGES +
            "**Claim under test:** when US institutions buy, Coinbase BTC-USD trades rich to Binance "
            "BTCUSDT, and the premium **leads** BTC returns (the CryptoQuant \"Coinbase Premium Gap\" "
            "folklore of the 2020-21 bull).\n\n"
            "**Tape:** Coinbase Exchange `/products/BTC-USD/candles` vs Binance `/api/v3/klines` "
            f"BTCUSDT, UTC daily closes on the same 00:00 boundary, {R['start']} → {R['end']} "
            f"({R['n_days']:,} joint days, {R['years']} yrs). As-of **{R['asof']}**, fingerprint "
            f"`{R['fingerprint']}`. Cache-first; `data.fetch_daily(fetch=True)` touches the network "
            "exactly once.\n\n"
            "**Named caveat:** the Binance leg is **USDT** — the series mixes the Coinbase USD bid "
            "with the Tether peg. We carry a 1%/99% winsorized robustness leg for exactly that. "
            "No survivorship (one asset, two venues, both alive throughout).\n\n"
            "> 💡 **In plain words:** we price the same coin in two shops at the same instant, and ask "
            "whether the gap between the tickets tells you anything about tomorrow."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · The premium series\n\n"
            f"Mean **{R['prem_mean']:+.2f} bps**, sd **{R['prem_sd']:.2f}**, range "
            f"**{R['prem_min']:+.0f} → {R['prem_max']:+.0f} bps**, AC(1) **{R['ac1']:.2f}** "
            "(persistent — hence HAC everywhere). Contemporaneously the premium co-moves with same-day "
            f"returns: corr **{R['contemp_corr']:+.3f}** (HAC *t* {R['contemp_t']:+.2f}) — a genuine "
            "*coincident* tell.\n\n"
            "> 💡 **In plain words:** the gap is tiny on the average day (a hundredth of a percent) "
            "but sticky — a rich day is usually followed by another rich day."
        ),

        code(
            "# Recompute the series stats live from the cache (must match R / docs/results.md).\n"
            "if HAVE_REAL:\n"
            "    print('days %d  %s -> %s  fingerprint %s' % (len(DF), DF.index.min().date(),\n"
            "          DF.index.max().date(), data.fingerprint(DF)))\n"
            "    print('premium: mean %+.2f  sd %.2f  min %+.1f  max %+.1f  AC1 %.3f' % (\n"
            "        PREM.mean(), PREM.std(ddof=1), PREM.min(), PREM.max(), PREM.autocorr(1)))\n"
            "    yt = st.yearly_table(PREM)\n"
            "    print(yt.round(2))\n"
            "else:\n"
            "    print('cache missing; frozen yearly table:', R['yearly'])\n"
        ),

        code(
            "# The series + the two regimes that matter (institutional bid, ETF era).\n"
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "    clipped = PREM.clip(-150, 150)\n"
            "    ax.plot(clipped.index, clipped.values, color=GREY, lw=0.6,\n"
            "            label='daily premium (clipped \\u00b1150 for display)')\n"
            "    ax.plot(PREM.rolling(30).mean().index, PREM.rolling(30).mean().values,\n"
            "            color=RED, lw=1.6, label='30d mean')\n"
            "    ax.axhline(0, color='k', lw=0.8)\n"
            "    ax.axvspan(pd.Timestamp('2020-10-01'), pd.Timestamp('2021-12-31'), color=GREEN,\n"
            "               alpha=0.10, label='2020-21 institutional bid')\n"
            "    ax.axvspan(pd.Timestamp('2024-01-11'), PREM.index.max(), color=AMBER,\n"
            "               alpha=0.10, label='spot-ETF era')\n"
            "    ax.set_ylabel('bps'); ax.legend(loc='lower left', ncols=2)\n"
            "    ax.set_title('Coinbase premium (bps): rich through the institutional bid, compressed and often negative since')\n"
            "    plt.tight_layout(); plt.show()\n"
        ),

        md(
            "## 2 · Predictive regressions — the claim as stated\n\n"
            "Forward *h*-day Coinbase log return on today's premium **level**; Newey-West *t* with "
            "lags = *h* + 2 (covers the forward-window overlap). Beta per +1 sd of the signal. The "
            "**raw level is the claim**; the winsorized leg (clip at the 1%/99% quantiles, "
            f"{R['winsor_lo']:+.1f}/{R['winsor_hi']:+.1f} bps) strips the USDT-depeg tails; the 1-day "
            "**change** tests the \"premium *expansion* is the trigger\" variant.\n\n"
            "> 💡 **In plain words:** we let today's gap try to predict tomorrow (and up to a week "
            "out), scoring it with a statistic that can't be fooled by the gap's stickiness."
        ),

        code(
            "# Live recompute (must match the frozen R table).\n"
            "if HAVE_REAL:\n"
            "    for lab, kw in [('LEVEL raw', dict(signal='level')),\n"
            "                    ('LEVEL winsorized 1%/99%', dict(signal='level', winsor=0.01)),\n"
            "                    ('1-DAY CHANGE', dict(signal='change'))]:\n"
            "        t = st.predictive_table(DF, **kw)\n"
            "        print(lab); print(t.round(3).to_string()); print()\n"
        ),

        code(
            "# t-stat profile across horizons, raw vs winsorized, with the desk bar at t = 2.\n"
            "hs = [h for h, b, t in R['pred_raw']]\n"
            "t_raw = [t for h, b, t in R['pred_raw']]\n"
            "t_win = [t for h, b, t in R['pred_wins']]\n"
            "x = np.arange(len(hs)); w = 0.36\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.bar(x - w/2, t_raw, w, color=GREY, label='raw level (the claim)')\n"
            "ax.bar(x + w/2, t_win, w, color=AMBER, label='winsorized 1%/99%')\n"
            "ax.axhline(2.0, color=RED, lw=1.4, ls='--', label='desk bar: t = 2')\n"
            "ax.axhline(0, color='k', lw=0.8)\n"
            "ax.set_xticks(x, [f'{h}d' for h in hs]); ax.set_xlabel('forward horizon')\n"
            "ax.set_ylabel('HAC t (NW, lags = h+2)')\n"
            "ax.set_title('Premium level \\u2192 forward return: always positive, rarely certified')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
        ),

        md(
            f"Raw h = 1 sits at **+1.42** — the headline claim does not clear the bar on 8.9 years of "
            "tape. Winsorizing lifts h = 1-3 to **2.10-2.30** and raw h = 3 touches **2.09**: the "
            "whisper firms exactly when you clean the peg noise, but those legs inherit the cleaning "
            "choice. The 1-day change predicts nothing (max |*t*| = 1.12). Group split, same story: "
            f"rich days (z63 > 0) → **{R['split_rich']:+.2f} bps** next day vs cheap days "
            f"**{R['split_cheap']:+.2f}** — Welch *t* only **{R['split_welch']:+.2f}** against "
            "crypto's ±370 bps daily noise.\n\n"
            "> 💡 **In plain words:** a 17-bps-a-day edge sounds enormous, but Bitcoin's daily swings "
            "are twenty times that. You need years more data — or a cleaner signal — to tell it from "
            "luck."
        ),

        md(
            "## 3 · Tradability — cost-charged overlay + the honest twins\n\n"
            "Long BTC for day *t+1* iff trailing 63-day z(premium) > 0 at close *t*. **Exactly one "
            f"execution lag**; {R['exposure']:.1f}% exposure; ~{R['turnover']} position changes/yr; "
            "one-way bps × traded NAV per change; long-only, no borrow; rf = 0 on both Sharpe legs "
            "(labeled). Timing alpha = net − exposure × B&H, the day-by-day surplus over a static "
            "position with the same market time."
        ),

        code(
            "# Live overlay recompute + the 20-seed exposure-matched random twin.\n"
            "if HAVE_REAL:\n"
            "    for cb in (5.0, 10.0, 25.0):\n"
            "        ov = st.overlay(DF, cost_bps=cb)\n"
            "        at = st.timing_alpha_t(DF, cost_bps=cb)\n"
            "        print('cost %4.1f bps: net %+5.2f bps/d (HAC t %+5.2f, Sharpe %.2f) | '\n"
            "              'alpha %+5.2f bps/d (t %+5.2f)' % (cb, ov['net_bps_d'], ov['net_t'],\n"
            "              ov['net_sharpe'], ov['alpha_bps_d'], at))\n"
            "    ov = st.overlay(DF, cost_bps=10.0)\n"
            "    print('B&H %+5.2f bps/d (Sharpe %.2f)' % (ov['bh_bps_d'], ov['bh_sharpe']))\n"
            "    rb = st.random_baseline(DF, exposure=ov['exposure'], cost_bps=10.0, n_seeds=20)\n"
            "    print('random twin (%d seeds): %+0.2f +/- %.2f bps/d, mean HAC t %+0.2f' % (\n"
            "          rb['n_seeds'], rb['mean_bps_d'], rb['sd_bps_d'], rb['mean_hac_t']))\n"
            "    for w_ in (20, 63, 126):\n"
            "        o2 = st.overlay(DF, z_window=w_, cost_bps=10.0)\n"
            "        print('z%3d: alpha %+5.2f bps/d (t %+5.2f)' % (w_, o2['alpha_bps_d'],\n"
            "              st.timing_alpha_t(DF, z_window=w_, cost_bps=10.0)))\n"
        ),

        code(
            "# Cumulative log-equity: overlay (net, 10 bps) vs buy & hold vs exposure-matched B&H.\n"
            "if HAVE_REAL:\n"
            "    ov = st.overlay(DF, cost_bps=10.0)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "    ax.plot(ov['bh'].cumsum().index, ov['bh'].cumsum().values, color=GREY, lw=1.6,\n"
            "            label='buy & hold')\n"
            "    ax.plot(ov['net'].cumsum().index, ov['net'].cumsum().values, color=AMBER, lw=1.6,\n"
            "            label='premium overlay (net, 10 bps)')\n"
            "    matched = (ov['exposure'] * ov['bh']).cumsum()\n"
            "    ax.plot(matched.index, matched.values, color=RED, lw=1.2, ls='--',\n"
            "            label='exposure-matched B&H (the fair bar)')\n"
            "    ax.set_ylabel('cumulative log return')\n"
            "    ax.set_title('The overlay hugs its exposure-matched twin — timing adds ~nothing certifiable')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
        ),

        md(
            f"Net of 10 bps the overlay earns **{R['overlay'][1][1]:+.2f} bps/d** (Sharpe "
            f"{R['overlay'][1][3]:.2f}) vs B&H **{R['bh_bps']:+.2f}** (Sharpe {R['bh_sharpe']:.2f}) — "
            f"it *underperforms* holding. Timing alpha: **{R['overlay'][1][6]:+.2f}σ** at 10 bps, "
            f"**{R['overlay'][2][6]:+.2f}σ** at 25 bps. Window robustness: z20/z63/z126 alpha *t* = "
            f"**{R['zrobust'][0][4]:+.2f} / {R['zrobust'][1][4]:+.2f} / {R['zrobust'][2][4]:+.2f}** — "
            "one knob setting clears, its neighbours don't: spec search, not edge. The "
            f"{R['rand_seeds']}-seed random twin: {R['rand_mean']:+.2f} ± {R['rand_sd']:.2f} bps/d "
            f"(mean HAC *t* {R['rand_t']:+.2f}) — the machinery pays nothing for luck.\n\n"
            "> 💡 **In plain words:** after fees, the \"institutional radar\" strategy makes *less* "
            "than doing nothing, and its skill is statistically a coin flip."
        ),

        md(
            "## 4 · Eras + the third axis — did the ETFs kill it?\n\n"
            "Pre-registered myth-check: since 2024 the US institutional bid clears through **spot "
            "ETFs**, not Coinbase — so the premium signal should be **dead** post-2022. Era cards, "
            "then an HAC-SE test of the pre-2023 vs 2023+ slope *difference* (the split is the "
            "pre-registered one, not snooped)."
        ),

        code(
            "# Live era recompute + slope-difference test.\n"
            "if HAVE_REAL:\n"
            "    eras = [('2017-2019', None, '2019-12-31'), ('2020-2021', '2020-01-01', '2021-12-31'),\n"
            "            ('2022', '2022-01-01', '2022-12-31'), ('2023+', '2023-01-01', None),\n"
            "            ('2024+ (ETF era)', '2024-01-01', None)]\n"
            "    for name, s, e in eras:\n"
            "        es = st.era_summary(st.slice_df(DF, s, e))\n"
            "        print('%-16s n=%4d prem %+7.2f (sd %6.2f)  beta %+7.2f bps/sd  t %+5.2f  '\n"
            "              'alpha %+6.2f (t %+5.2f)' % (name, es['n'], es['prem_mean_bps'],\n"
            "              es['prem_sd_bps'], es['beta_bps_per_sd'], es['pred_t'],\n"
            "              es['alpha_bps_d'], es['alpha_t']))\n"
            "    pre = st.slice_df(DF, None, '2022-12-31'); post = st.slice_df(DF, '2023-01-01', None)\n"
            "    def _slope(d):\n"
            "        p = st.premium_bps(d); y = st.forward_return(st.log_returns(d), 1)\n"
            "        r = st.hac_slope(p.to_numpy(), y.to_numpy(), lags=3)\n"
            "        return r['beta'], r['beta'] / r['t']\n"
            "    b1, se1 = _slope(pre); b2, se2 = _slope(post)\n"
            "    print('slope pre %+0.3f bps/bp vs post %+0.3f bps/bp (%0.0fx)  diff t = %+0.2f' % (\n"
            "          b1*1e4, b2*1e4, b2/b1, (b2-b1)/np.hypot(se1, se2)))\n"
        ),

        code(
            "# Era profile: predictive t (bars, desk bar at 2) -- the 2023+ surprise.\n"
            "labs = [e[0] for e in R['eras']]\n"
            "ts = [e[5] for e in R['eras']]\n"
            "colors = [GREEN if t >= 2 else GREY for t in ts]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar(labs, ts, color=colors, width=0.55)\n"
            "ax.axhline(2.0, color=RED, lw=1.4, ls='--', label='desk bar: t = 2')\n"
            "ax.axhline(0, color='k', lw=0.8)\n"
            "for i, t in enumerate(ts):\n"
            "    ax.annotate(f'{t:+.2f}', (i, t + 0.06), ha='center', fontsize=10)\n"
            "ax.set_ylabel('h = 1 predictive HAC t')\n"
            "ax.set_title('The only era that certifies itself is 2023+ — NOT the celebrated 2020-21 bid')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
        ),

        md(
            f"**Busted.** The premium compressed ~10-20× (sd {R['eras'][0][3]:.0f} → "
            f"{R['eras'][4][3]:.1f} bps) and 2026 H1 is negative on {R['neg2026'][0]}/{R['neg2026'][1]} "
            f"days — but per bp the slope is **{R['slope_ratio']}× larger** in 2023+ "
            f"({R['slope_post']:+.3f} vs {R['slope_pre']:+.3f} bps/bp, difference *t* = "
            f"**{R['slope_diff_t']:+.2f}**), and 2023+ is the only slice clearing the bar alone "
            f"(*t* = +2.87). The 2024+ ETF-era sub-sample reads +1.80 — sharper but too thin to "
            "re-certify on its own. And the era the folklore celebrates, 2020-21, predicts *nothing* "
            "(*t* = +0.56): the premium was rich, but knowing so earned you nothing extra.\n\n"
            "> 💡 **In plain words:** the ETFs shrank the footprint but made each remaining basis "
            "point more meaningful — the exact opposite of \"dead\". Still too faint to trade."
        ),

        md(
            "## 5 · Synthetic control — machinery proof *(never market evidence)*\n\n"
            "A deterministic two-venue world: shared random-walk BTC, AR(1) premium (φ = 0.9, sd 25 "
            "bps), and a plantable **lead knob** adding `lead × z(premium)` to the *next* day's "
            "return. Null (lead = 0) must stay quiet; a planted lead must light up."
        ),

        code(
            "# No network; seeds fixed.\n"
            "for lead in (0.0, 0.003):\n"
            "    syn = data.synthetic_world(lead=lead)\n"
            "    pt = st.predictive_table(syn, horizons=(1,), signal='level')\n"
            "    print('planted lead=%.3f: h=1 beta %+7.2f bps/sd  HAC t %+6.2f' % (\n"
            "          lead, pt['beta_bps_per_sd'].iloc[0], pt['hac_t'].iloc[0]))\n"
        ),

        md(
            f"Null: *t* = **{R['syn'][0][2]:+.2f}** (quiet). Planted lead: *t* = "
            f"**{R['syn'][1][2]:+.2f}** (lights up). The real-tape *t*'s are read off an honest "
            "instrument — which makes the +1.42 all the more damning.\n\n"
            "## 6 · Verdict\n\n"
            "- **Signal — WEAK.** Direction positive at every horizon and era; raw headline HAC *t* "
            "= **+1.42**; only spec-assisted legs (winsorized 2.10-2.30, h = 3 at 2.09, 2023+ at "
            "2.87) clear the bar; the celebrated 2020-21 era predicts nothing (+0.56).\n"
            "- **Tradability — MIRAGE.** Timing alpha +0.80σ at 10 bps, negative at 25 bps, "
            "window-fragile (2.15/0.80/1.48 across z20/z63/z126); underperforms buy & hold at every "
            "cost; 20-seed random twin ≈ 0.\n"
            "- **\"Dead since the ETFs?\" — BUSTED.** Compressed 10-20×, yet 22× sharper per bp "
            "(difference *t* = +2.72); 2023+ is the only self-certifying slice. Shrunk and "
            "sharpened — not dead, and still not tradable.\n\n"
            "*Every number: `python examples/verify.py` · frozen in `docs/results.md` (as-of "
            f"{R['asof']}, fingerprint `{R['fingerprint']}`). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for fname, builder in [("01_for_the_curious.ipynb", build_curious),
                           ("02_for_the_quants.ipynb", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)
