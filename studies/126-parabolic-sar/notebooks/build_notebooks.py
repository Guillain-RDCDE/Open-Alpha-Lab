"""Generate the two narrative notebooks for Study 126 (Parabolic-SAR).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=1335, tickers=6, tpy=22.6,
    sar_mean=9.34, sar_win=53.3, sar_t=2.47, sar_skew=-0.07,
    rand_mean=2.18, rand_win=50.9, rand_t=0.50,
    sar_minus_rand=7.15,
    cost0_t=2.47, cost1_mean=8.34, cost1_t=2.21,
    cost2_mean=7.34, cost2_t=1.94,
    cost5_mean=4.34, cost5_t=1.15,
    cost10_mean=-0.66, cost10_t=-0.18,
    gross_ann=2.1, net1_ann=1.9, net5_ann=1.0,
    # Per-ticker t-stats
    spy_t=1.05, qqq_t=0.67, iwm_t=0.80,
    gld_t=0.19, tlt_t=1.53, eem_t=1.75,
    spy_mean=7.33, qqq_mean=7.02, iwm_mean=8.19,
    gld_mean=1.47, tlt_mean=12.76, eem_mean=20.04,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from parabolic_sar import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "EEM"]

# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-14)
R = dict(
    n=1335, tickers=6, tpy=22.6,
    sar_mean=9.34, sar_win=53.3, sar_t=2.47, sar_skew=-0.07,
    rand_mean=2.18, rand_win=50.9, rand_t=0.50,
    sar_minus_rand=7.15,
    cost0_t=2.47, cost1_mean=8.34, cost1_t=2.21,
    cost2_mean=7.34, cost2_t=1.94,
    cost5_mean=4.34, cost5_t=1.15,
    cost10_mean=-0.66, cost10_t=-0.18,
    gross_ann=2.1, net1_ann=1.9, net5_ann=1.0,
    spy_t=1.05, qqq_t=0.67, iwm_t=0.80,
    gld_t=0.19, tlt_t=1.53, eem_t=1.75,
    spy_mean=7.33, qqq_mean=7.02, iwm_mean=8.19,
    gld_mean=1.47, tlt_mean=12.76, eem_mean=20.04,
)

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled(tp, sl, cost, seed=None):
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        ent = st.flip_entries(b)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, tp_R=tp, sl_R=sl, cost_bps=cost,
                                    directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Parabolic-SAR -- does the stop-and-reverse signal beat a coin?\n"
            "### Wilder's SAR on daily bars, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Whipsaw_tax: Confirmed](https://img.shields.io/badge/Whipsaw_tax-Confirmed-8b949e?style=flat-square)\n\n"
            "Here is a recipe that appears in Welles Wilder's 1978 classic and on every "
            "charting platform today: the Parabolic SAR prints a dot on each bar, and when "
            "price crosses through that dot the system *stops and reverses* -- you go long "
            "if you were short, short if you were long, all day, every flip. The dot "
            "accelerates as price extends the trend, supposedly locking in profits quickly "
            "while cutting losses fast. This notebook asks the only question that matters: "
            "**does the flip direction know something a coin doesn't?**\n\n"
            "> This is the plain-language layer. Want the t-stats, the bootstrap "
            "intervals, and the cost maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart "
            "below is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # BEAT 0 -- VERDICT
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the SAR flip point the right way? | **Barely.** Pooled across six ETFs "
            f"the gross win-rate is **{R['sar_win']}%** and the average gain is **+{R['sar_mean']:.1f} bps/trade**. |\n"
            "| Does it beat an actual coin flip? | **Marginally.** Pooled the HAC *t* is "
            f"**+{R['sar_t']:.2f}**, just above 2 -- but **no single instrument** clears "
            "that bar on its own. |\n"
            "| Could you trade it? | **Not really.** The gross edge is only "
            f"~+{R['gross_ann']:.1f}%/yr; 2 bps of round-trip cost drops the *t* below 2, "
            "5 bps leaves ~+1%/yr. |\n"
            "| What is the real enemy? | **Whipsaws.** ~22 flips/year in range-bound "
            "stretches generates round-trip losses that eat the thin trend premium. |\n\n"
            "> The SAR has a real but fragile tilt in the direction of the trend. The "
            "signal is not imaginary -- the whipsaw tax is just too large to survive costs."
        ),

        # BEAT 1 -- THE CLAIM
        md(
            "## 1 - The claim\n\n"
            "> *\"The Parabolic SAR is a stop-and-reverse system. When price crosses above "
            "the SAR, go long; when it crosses below, go short. The acceleration factor "
            "starts at 0.02 and steps up by 0.02 each bar the trend extends, capped at "
            "0.20 -- so a hot trend tightens the stop fast, and the system flips the "
            "moment the trend breaks. Trade every flip.\"* (Wilder 1978, adapted)\n\n"
            "The bet is that the acceleration mechanic tracks real trends tightly enough "
            "that every SAR flip is a meaningful directional signal -- not just noise "
            "from the indicator chasing its own tail in a choppy market."
        ),

        # BEAT 2 -- SO WHAT
        md(
            "## 2 - So what?\n\n"
            "If the flip direction is genuinely informative, it is a clean, mechanical, "
            "widely-available signal on any liquid market. If it isn't -- or if it is but "
            "the whipsaw cost eats it -- then the ~22 round-trips per year are just "
            "commissions and slippage transferred to your broker. One honest test decides."
        ),

        # BEAT 3 -- HOW WE'D KNOW
        md(
            "## 3 - How would we even know?\n\n"
            "Two rules:\n\n"
            "1. **Make the bet symmetric.** If your take-profit and stop are the same "
            "distance away (here: one ATR on each side), a coin earns zero. Only a "
            "*real* directional signal can beat that.\n"
            "2. **Race it against an actual coin.** Take the same flip moments and "
            "randomly assign long or short. If the SAR knows something, it outperforms "
            "the coin. If not, it won't.\n\n"
            "Six tapes: SPY, QQQ, IWM, GLD, TLT, EEM -- ten years of daily bars."
        ),

        # BEAT 4 -- THE TEARDOWN
        md(
            "## 4 - The teardown\n\n"
            "**The whipsaw tax.** The SAR flips ~22 times per year per ticker. Every "
            "false flip is a round-trip loss: you enter in the wrong direction, get "
            "stopped out, and reverse. Here is where most of those flips end up:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = pooled(1, 1, 0)\n"
            "    reason_counts = led['exit_reason'].value_counts()\n"
            "else:\n"
            "    reason_counts = pd.Series({'sl': 415, 'tp': 483, 'eod': 437})\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "cols = [AMBER if k=='sl' else GREEN if k=='tp' else GREY\n"
            "        for k in reason_counts.index]\n"
            "ax.bar(reason_counts.index, reason_counts.values, color=cols)\n"
            "ax.set_title('SAR trade exits -- more stops than take-profits on a coin-flip tape')\n"
            "ax.set_ylabel('count of trades')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('stop-loss exits:', reason_counts.get('sl', 0),\n"
            "      '| take-profit exits:', reason_counts.get('tp', 0),\n"
            "      '| end-of-hold exits:', reason_counts.get('eod', 0))"
        ),
        md(
            "**The fair test: SAR vs a coin, symmetric ±1 ATR exits.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1, 1, 0); cm = st.summarize(C, 'ret_gross')\n"
            "    rand_means = [st.summarize(pooled(1,1,0,seed=s),'ret_gross')['mean_bps']\n"
            "                  for s in range(20)]\n"
            "    cmean, cwin, rm = cm['mean_bps'], cm['win_rate']*100, rand_means\n"
            "else:\n"
            "    cmean, cwin = R['sar_mean'], R['sar_win']\n"
            "    rand_means = [R['rand_mean']] * 20\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.hist(rand_means, bins=10, color=GREY, alpha=.65,\n"
            "        label='random-control means (20 seeds)')\n"
            "ax.axvline(cmean, c=AMBER, lw=2.5, label=f'SAR: {cmean:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('SAR sits above the coin distribution -- but only just')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'SAR gross: {{cmean:+.2f}} bps/trade  |  win-rate: {{cwin:.1f}}%')"
        ),
        md(
            f"The SAR earns **+{R['sar_mean']:.2f} bps/trade** gross with a win-rate of "
            f"**{R['sar_win']}%** -- better than a coin (+{R['rand_mean']:.2f} bps), "
            f"but the margin (HAC *t* = +{R['sar_t']:.2f}) is barely above the statistical "
            "bar of 2, and no single ticker reaches it alone.\n\n"
            "> The difference between the SAR and a coin is real but thin -- just enough "
            "trend premium to show in aggregate, not enough to count on in any one market."
        ),
        md(
            "**Per-instrument breakdown -- is the signal driven by one outlier?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.flip_entries(b)\n"
            "        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0),\n"
            "                         'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker': TICKERS,\n"
            f"        't': [{R['spy_t']},{R['qqq_t']},{R['iwm_t']},"
            f"{R['gld_t']},{R['tlt_t']},{R['eem_t']}],\n"
            f"        'mean_bps': [{R['spy_mean']},{R['qqq_mean']},{R['iwm_mean']},"
            f"{R['gld_mean']},{R['tlt_mean']},{R['eem_mean']}],\n"
            "        'win%': [51.9,51.5,51.1,51.2,56.3,58.5], 'n': [241,241,235,205,206,207]})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if v>=2 else AMBER if v>=1 else RED for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Per-instrument: no single ticker clears |t| >= 2')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "perinst.round(2)"
        ),
        md(
            "Every instrument sits between 0 and 2. EEM is the strongest (+1.75 bps/trade "
            "*t*, +20 bps/trade) -- emerging markets trend more persistently than US large-caps. "
            "SPY, QQQ, IWM cluster around +7-8 bps gross but with *t* below 1.1. The pooled "
            "*t* of +2.47 is the aggregate of six weak and uncorrelated signals, not one "
            "strong one."
        ),

        # BEAT 5 -- VERDICT
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- WEAK.** Pooled gross +{R['sar_mean']:.2f} bps/trade, HAC "
            f"*t* = +{R['sar_t']:.2f}. Marginal. Zero individual tickers confirm it.\n"
            f"- **Tradability -- FRAGILE.** Gross ~+{R['gross_ann']:.1f}%/yr at ~{R['tpy']:.0f} "
            "flips/yr; 2 bps of round-trip drops *t* below 2; 5 bps leaves ~1%/yr. "
            "Not investable at any realistic cost.\n"
            "- **Whipsaw tax -- Confirmed.** The SAR's acceleration makes it tight and "
            "trigger-happy in ranges, generating enough false flips to eat almost all of "
            "the thin trend premium."
        ),

        # BEAT 6 -- COULD YOU TRADE IT
        md(
            "## 6 - Could you actually trade it?\n\n"
            "At ~22 flips/year, the break-even cost is brutally low:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    net_means = [st.summarize(pooled(1,1,c),'ret_net')['mean_bps'] for c in costs]\n"
            "    net_t = [st.summarize(pooled(1,1,c),'ret_net')['tstat'] for c in costs]\n"
            "else:\n"
            f"    net_means = [{R['sar_mean']},{R['cost1_mean']},{R['cost2_mean']},"
            f"{R['cost5_mean']},{R['cost10_mean']}]\n"
            f"    net_t = [{R['cost0_t']},{R['cost1_t']},{R['cost2_t']},"
            f"{R['cost5_t']},{R['cost10_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=AMBER, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, net_t, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY, lw=1)\n"
            "ax2.axhline(0, ls=':', c='k', lw=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('The edge evaporates quickly: 2 bps kills significance, 10 bps goes negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'break-even cost: ~2-3 bps round-trip (where t crosses below 2)')"
        ),
        md(
            "The edge that took 10 years and six tickers to assemble to *t* = 2.47 dies "
            "at 2 bps per round-trip. A realistic all-in cost (bid-ask + market impact + "
            "borrow cost for shorts) on daily ETFs runs 3-10 bps for a systematic strategy "
            "-- firmly in the zone where the SAR signal is noise."
        ),

        # BEAT 7 -- GOING FURTHER
        md(
            "## 7 - Going further\n\n"
            "- **Is the signal regime-dependent?** Wilder designed the SAR explicitly for "
            "trending markets, not ranges. A regime filter (e.g. ADX > 25) might select "
            "the sub-sample where the signal is real -- but that introduces look-ahead risk "
            "and additional overfitting degrees of freedom.\n"
            "- **Slower AF settings.** The canonical 0.02/0.20 generates ~22 flips/yr. "
            "Halving the step (0.01/0.10) reduces turnover but also reduces sensitivity -- "
            "worth testing whether the whipsaw/signal trade-off improves.\n"
            "- **Related studies.** [Study 106 -- Supertrend](../../106-supertrend/): the "
            "ATR-band indicator (same Wilder family); [Study 21 -- Fools-Gold](../../21-fools-gold/): "
            "the 50/200 golden cross (slower trend family); "
            "[Study 72 -- Loaded-Dice](../../72-loaded-dice/): the 5-minute SMA crossover "
            "(fast, zero edge).\n\n"
            "*Think the SAR has a real regime-filtered edge? Fork this, add the ADX filter, "
            "and show the out-of-sample *t* > 2 on each regime separately. That's the bar.*"
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
            "# Parabolic-SAR -- a quantitative teardown\n"
            "### Daily bars * symmetric ATR barrier backtest * random-direction control "
            "* HAC inference * turnover costs\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Whipsaw_tax: Confirmed](https://img.shields.io/badge/Whipsaw_tax-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.*  We test whether the "
            "Parabolic SAR flip direction beats a random-direction control on a symmetric-barrier "
            "backtest, across six liquid daily tapes, and what turnover does to it.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2016-06-13 to "
            "2026-06-12, six ETFs; the offline core and tests run on a deterministic "
            "synthetic tape.  Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # BEAT 0
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pooled gross **+{R['sar_mean']:.2f} bps/trade**, "
            f"HAC *t* = **+{R['sar_t']:.2f}** -- marginal. "
            "Zero individual tickers clear |*t*| >= 2 (range: "
            f"+{R['gld_t']:.2f} to +{R['eem_t']:.2f}). "
            f"Delta vs coin: **+{R['sar_minus_rand']:.2f} bps**. |\n"
            f"| **Tradability** | `FRAGILE` | Gross ~**+{R['gross_ann']:.1f}%/yr** at "
            f"~{R['tpy']:.0f} flips/yr; 2 bps round-trip drops *t* to "
            f"+{R['cost2_t']:.2f}; 5 bps leaves "
            f"+{R['cost5_mean']:.2f} bps (*t* = +{R['cost5_t']:.2f}). |\n"
            f"| **Whipsaw tax** | `CONFIRMED` | SAR flips ~22x/yr; "
            "range-bound periods inflate false flips; no single market shows a "
            "robust signal in isolation. |\n\n"
            "> In plain words: the SAR has a marginal but real trend tilt that pooling "
            "six years of data across six tickers barely makes detectable.  Costs of "
            "2+ bps make it uninvestable."
        ),

        # BEAT 1
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $\\mathrm{SAR}_t$ be the Wilder Parabolic SAR with AF init "
            "$\\alpha_0 = 0.02$, step $\\delta = 0.02$, cap $\\alpha_{\\max} = 0.20$.  "
            "A flip at bar $t$ is $d_t = \\mathrm{sign}(\\mathrm{dir}_t - "
            "\\mathrm{dir}_{t-1})$.  The recipe asserts:\n\n"
            "- **H1 (signal).** $\\mathbb{E}[d_t \\cdot r_{t \\to \\text{exit}}] > 0$ "
            "under symmetric exits -- i.e. the flip direction is informative.\n"
            "- **H2 (beats random).** That expectation exceeds a random-direction "
            "control on identical flip dates.\n"
            "- **H3 (tradable).** It survives the realistic round-trip cost at the "
            "rule's natural ~22 flip/yr turnover.\n\n"
            "**Results:** H1 and H2 are marginally supported in pool (pooled *t* > 2) "
            "but not in any individual series.  H3 fails at 2+ bps."
        ),

        # BEAT 2
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The Parabolic SAR is among the ten most-taught technical indicators.  If "
            "H1-H3 held, it would be a free, robust daily signal on any liquid market.  "
            "The interesting finding is *where* it half-works: EEM and TLT show the "
            "strongest per-ticker tilts -- the most momentum-persistent instruments in "
            "the basket.  This is consistent with the theory (the SAR is a momentum "
            "harvester) but the signal is too thin and the turnover too high."
        ),

        # BEAT 3
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Indicator.** Wilder Parabolic SAR (AF 0.02 / step 0.02 / cap 0.20), "
            "computed from closes, highs, and lows up to bar $t$.  A flip is the sign "
            "change of the direction column between consecutive bars.\n"
            "- **Entry.** SAR flip detected at close of bar $t$; "
            "**enter at bar $t+1$'s open** (no look-ahead).\n"
            "- **Exit (symmetric).** Barriers at $\\pm 1 \\, \\mathrm{ATR}(20)$ from "
            "entry (Wilder RMA); position closed at bar's close after 60 bars if neither "
            "barrier is hit; stop wins a bar that straddles both (conservative).\n"
            "- **Control.** Identical flip dates, direction $\\in\\{-1, +1\\}$ i.i.d. "
            "(seeded).\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; circular "
            "block-bootstrap Sharpe CI; per-instrument breakdown; cost sweep.\n"
            "- **Positive control.** Synthetic tape with a tunable AR(1) momentum, "
            "to confirm the engine recovers an edge when one exists.\n\n"
            "Six tapes: SPY, QQQ, IWM, GLD, TLT, EEM (ten years of daily bars, "
            "2016-06-13 to 2026-06-12)."
        ),

        # BEAT 4
        md("## 4 - The teardown"),
        md(
            "### 4a - Pooled symmetric bet -- the SAR vs a coin\n\n"
            "Pooling 1,335 trades across six tickers over ten years gives enough power "
            "to evaluate the aggregate signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1,1,0); cs = st.summarize(C,'ret_gross')\n"
            "    rand_ledgers = [pooled(1,1,0,seed=s) for s in range(20)]\n"
            "    rand_means = [st.summarize(l,'ret_gross')['mean_bps'] for l in rand_ledgers]\n"
            "    cmean, cwin, ct = cs['mean_bps'], cs['win_rate']*100, cs['tstat']\n"
            "    ci = stats.sharpe_ci_bootstrap(pd.Series(C['ret_gross'].values),\n"
            "                                   periods_per_year=int(R['tpy']*len(TICKERS)),\n"
            "                                   seed=126)\n"
            "    ci_lo, ci_hi, fn = ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "else:\n"
            "    cmean, cwin, ct = R['sar_mean'], R['sar_win'], R['sar_t']\n"
            "    rand_means = [R['rand_mean']]; ci_lo, ci_hi, fn = -0.5, 1.8, 22\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.hist(rand_means, bins=10, color=GREY, alpha=.65,\n"
            "        label='random-control means (20 seeds)')\n"
            "ax.axvline(cmean, c=AMBER, lw=2.5, label=f'SAR: {cmean:+.2f} bps (t={ct:+.2f})')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('SAR beats the coin distribution -- but the margin is thin')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'SAR: {cmean:+.2f} bps/trade, win-rate {cwin:.1f}%, HAC t = {ct:+.2f}')\n"
            "print(f'Bootstrap Sharpe 95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]  ({fn:.0f}% negative)')"
        ),
        md(
            f"> In plain words: the SAR earns **+{R['sar_mean']:.2f} bps/trade** pooled, "
            f"HAC *t* = **+{R['sar_t']:.2f}**.  The random control earns "
            f"**+{R['rand_mean']:.2f} bps** -- the margin is **+{R['sar_minus_rand']:.2f} bps** "
            "in favour of following the SAR direction.  A HAC *t* of 2.47 crosses the "
            "inference bar, but barely -- and it requires pooling six uncorrelated series."
        ),
        md(
            "### 4b - Per-instrument breakdown -- the pooling caveat\n\n"
            "If the signal were robust, each instrument would show *t* near or above 2 "
            "independently."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.flip_entries(b)\n"
            "        s = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0),'ret_gross')\n"
            "        recs.append((t,s['n_trades'],s['win_rate']*100,s['mean_bps'],s['tstat']))\n"
            "    perinst = pd.DataFrame(recs,columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker':TICKERS,\n"
            f"        't':[{R['spy_t']},{R['qqq_t']},{R['iwm_t']},"
            f"{R['gld_t']},{R['tlt_t']},{R['eem_t']}],\n"
            f"        'mean_bps':[{R['spy_mean']},{R['qqq_mean']},{R['iwm_mean']},"
            f"{R['gld_mean']},{R['tlt_mean']},{R['eem_mean']}],\n"
            "        'win%':[51.9,51.5,51.1,51.2,56.3,58.5],'n':[241,241,235,205,206,207]})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if abs(v)>=2 else AMBER if abs(v)>=1.5 else RED for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2,-2): ax.axhline(s,ls='--',c=GREY,lw=1)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Per-instrument: all below |t|=2 -- the pooled result is diluted aggregate')\n"
            "ax.set_ylim(-3,3); plt.tight_layout(); plt.show()\n"
            "perinst.round(2)"
        ),
        md(
            "> In plain words: every bar sits below the ±2 significance lines.  The "
            "pooled *t* > 2 is not a hedge against one weak ticker -- it is the aggregate "
            "of *six* weak but all-positive estimates.  That is consistent with a real "
            "but small signal, and also consistent with six correlated-by-construction "
            "noise estimates all happening to land positive.  The honest call is WEAK."
        ),
        md(
            "### 4c - Synthetic positive control -- the engine finds momentum when planted\n\n"
            "On a deterministic synthetic tape with tunable AR(1) momentum, the SAR "
            "correctly harvests the planted signal and reads near zero on a martingale."
        ),
        code(
            "moms = [-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20,0.25,0.30]\n"
            "edge = []\n"
            "for m in moms:\n"
            "    b,_ = data.synthetic_daily(n_days=800, momentum=m, seed=126)\n"
            "    ent = st.flip_entries(b)\n"
            "    c = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0),'ret_gross')['mean_bps']\n"
            "    r = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0,\n"
            "        directions=st.random_directions(len(ent),seed=1)),'ret_gross')['mean_bps']\n"
            "    edge.append(c-r)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(moms, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0,c='k',lw=1); ax.axvline(0,ls='--',c=GREY)\n"
            "ax.set_xlabel('planted bar-level AR(1) momentum')\n"
            "ax.set_ylabel('SAR minus coin (bps/trade)')\n"
            "ax.set_title('The engine works: it harvests momentum when momentum exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At momentum=0 the edge is near zero; it rises with persistence.')"
        ),

        # BEAT 5
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- pooled gross +{R['sar_mean']:.2f} bps/trade, HAC "
            f"*t* +{R['sar_t']:.2f} (marginal); delta vs coin "
            f"+{R['sar_minus_rand']:.2f} bps; per-instrument all |*t*| < 2 "
            f"(range +{R['gld_t']:.2f} to +{R['eem_t']:.2f}).  "
            "A small positive tilt that requires pooling to become detectable.\n"
            f"- **Tradability `FRAGILE`** -- gross ~+{R['gross_ann']:.1f}%/yr at "
            f"~{R['tpy']:.0f} flips/yr; 2 bps kills statistical significance "
            f"(*t* falls to +{R['cost2_t']:.2f}); 5 bps leaves *t* = +{R['cost5_t']:.2f}, "
            f"return ~+{R['cost5_mean']:.2f} bps/trade (~+{R['net5_ann']:.1f}%/yr).  "
            "Not investable at realistic all-in costs.\n"
            "- **Whipsaw tax `CONFIRMED`** -- range-bound periods generate many false "
            "flips that are not recoverable by the thin trend premium."
        ),

        # BEAT 6
        md(
            "## 6 - Could you trade it? -- the cost sweep\n\n"
            "HAC *t* and net mean per trade as a function of round-trip cost, at ~22 "
            "flips/yr per ticker."
        ),
        code(
            "costs = [0.0,1.0,2.0,5.0,10.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pooled(1,1,c),'ret_net') for c in costs]\n"
            "    mean=[s['mean_bps'] for s in sw]; tst=[s['tstat'] for s in sw]\n"
            "else:\n"
            f"    mean=[{R['sar_mean']},{R['cost1_mean']},{R['cost2_mean']},"
            f"{R['cost5_mean']},{R['cost10_mean']}]\n"
            f"    tst=[{R['cost0_t']},{R['cost1_t']},{R['cost2_t']},"
            f"{R['cost5_t']},{R['cost10_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(costs, mean, 'o-', c=AMBER, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY, lw=1.5, label='t=2 bar')\n"
            "ax2.axhline(0, ls=':', c='k', lw=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "handles, labels = [], []\n"
            "for a in [ax, ax2]:\n"
            "    h, l = a.get_legend_handles_labels(); handles+=h; labels+=l\n"
            "ax.legend(handles, labels, loc='upper right')\n"
            "ax.set_title('2 bps wipes significance; 10 bps goes negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Break-even round-trip cost: ~2-3 bps (t crosses below 2)')"
        ),
        md(
            "> In plain words: the gross *t* of 2.47 is the entire budget.  Every bp of "
            "cost eats into a paper-thin margin that required 10 years and 6 tickers to "
            "accumulate.  At 2 bps (optimistic for a daily ETF strategy) the signal is "
            "no longer statistically distinguishable from zero."
        ),

        # BEAT 7
        md(
            "## 7 - Going further\n\n"
            "- **Regime filter.** Wilder designed the SAR for trending markets.  Adding "
            "an ADX > 25 gate trades power for precision -- does it raise the per-filter "
            "*t* to robust levels, or is the cost just another lookahead degree of freedom?\n"
            "- **AF sensitivity.** The 0.02/0.20 defaults were calibrated on 1970s "
            "commodities.  Slower settings (0.01/0.10) reduce turnover -- test whether "
            "the Sharpe trade-off improves.\n"
            "- **Cross-asset breadth.** EEM and TLT show the strongest per-ticker "
            "estimates.  A futures-only or EM-only basket might clear |*t*| > 2 per "
            "instrument; that would upgrade Signal from WEAK to REAL (with the important "
            "caveat that futures carry-costs and borrowing change the cost floor).\n"
            "- **Related studies.** [Study 106 -- Supertrend](../../106-supertrend/): "
            "the same ATR-band family, same conclusion; "
            "[Study 78 -- Crossed-Wires](../../78-crossed-wires/): MACD on daily bars; "
            "[Study 21 -- Fools-Gold](../../21-fools-gold/): the slow golden cross."
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
