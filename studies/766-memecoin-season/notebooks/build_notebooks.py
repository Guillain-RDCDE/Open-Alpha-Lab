"""Generate the two narrative notebooks for Study 766 (Memecoin-Season).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached BTC/DOGE/SHIB
tapes under ../_cache/ (no network once cached), else quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere, no network.
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
# (BTC/DOGE/SHIB weekly close, yfinance, common window 2021-04-16 -> 2026-06-26, 272 weekly
#  bars / 267 rotation weeks; weekly top-trailing-4-week momentum rotation, one-week lag,
#  30 bps per leg; fingerprint b75635d0b392).
R = dict(
    win_lo="2021-04-16", win_hi="2026-06-26", n_weeks=272, n_rot=267,
    daily_lo="2021-04-16", daily_hi="2026-06-30", daily_n=1902,
    fp="b75635d0b392",
    # steelman: buy-and-hold total return over the common window
    bh_btc=-5, bh_doge=-80, bh_shib=300,
    # momentum rotation, gross and net (30 bps/leg)
    rot_gross_total=-60, rot_gross_sharpe=0.39, rot_gross_dd=-87,
    rot_total=-73.6, rot_cagr=-23.0, rot_sharpe=0.34, rot_dd=-89, rot_vol=162, rot_hit=42.3,
    n_switches=68, avg_turnover=0.51,
    # benchmarks
    btc_total=-2.5, btc_cagr=-0.5, btc_sharpe=0.26, btc_dd=-74, btc_vol=55,
    ew_total=217.2, ew_cagr=25.0, ew_sharpe=0.62, ew_dd=-80, ew_vol=118,
    # is the edge over BTC real?
    excess_mean_wk=0.717, excess_t=0.59,
    # random-rotation placebo
    placebo_seeds=4000, placebo_p_total=0.517, placebo_p_sharpe=0.410,
    rand_median_total=-72, rand_median_sharpe=0.28,
    # cost sweep: bps -> (net total %, net Sharpe)
    cost_sweep={0: (-60, 0.39), 10: (-65, 0.37), 30: (-74, 0.34), 50: (-80, 0.31), 100: (-90, 0.23)},
    # sub-period split at 2022-01-01
    mania_weeks=33, mania_rot_total=5.1, mania_rot_sharpe=1.01, mania_btc_total=-7.2,
    mania_btc_sharpe=0.18, mania_t=0.83,
    after_weeks=234, after_rot_total=-74.8, after_rot_sharpe=0.08, after_btc_total=29.6,
    after_btc_sharpe=0.37, after_t=-0.32,
    # synthetic control
    syn_null_mean=0.06, syn_null_sd=1.15, syn_null_fire=2, syn_planted_t=5.12,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin_flip%3F: Busted](https://img.shields.io/badge/Beats_a_coin_flip%3F-Busted-8b949e?style=flat-square)\n\n"
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
BLUE = "#2b6cb0"

from memecoin_season import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    WK = data.weekly_prices()
    DAILY = data.load_prices()
else:
    WK = DAILY = None
print("real cache present:", HAVE_REAL,
      "| weekly bars:", (0 if WK is None else len(WK)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Memecoin season: hop on the hot dog-coin and outrun Bitcoin? 🐕\n"
            "### A weekly momentum rotation across BTC / DOGE / SHIB — the strategy every "
            "bull market reinvents\n\n"
            + BADGES +
            "Every crypto cycle, the same story goes around: *forget boring Bitcoin — in a "
            "\"memecoin season\" the dog-coins go up 10×, 100×, 1000×, and all you have to do "
            "is rotate into whichever one is running.* It sounds almost too easy. So we built "
            "the literal strategy — every week, jump onto whichever of Bitcoin, Dogecoin and "
            "Shiba Inu has run hardest lately — and raced it against just buying and holding "
            "Bitcoin.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the vol math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **The one thing to know up front.** DOGE and SHIB are the *two memecoins that "
            "survived* out of literally thousands. In 2021 you didn't know which two those "
            "would be — so every number here is already the *best case*, and it still doesn't "
            "work. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did memecoins really blow past Bitcoin? | **Only one of them.** From April 2021 "
            f"(when all three first trade) to mid-2026: SHIB **+{R['bh_shib']}%**, but DOGE "
            f"**{R['bh_doge']}%** and BTC **{R['bh_btc']}%**. \"Memecoins win\" is really "
            "\"*the one memecoin that happened to survive* won.\" |\n"
            "| Does the momentum rotation harvest it? | **No — it's a disaster.** Net of a "
            f"gentle 30 bps a trade it returns **{R['rot_total']:.0f}%** versus BTC's "
            f"**{R['btc_total']:.0f}%** and an equal-weight basket's **+{R['ew_total']:.0f}%** "
            f"— an **{abs(R['rot_dd']):.0f}% drawdown** along the way. |\n"
            "| Is it the trading costs killing it? | **No — it's broken even for free.** At "
            f"**zero** cost the rotation still returns **{R['rot_gross_total']}%**. Chasing "
            "last week's winner in assets this violent just buys high and sells low. |\n"
            "| Better than picking a coin at random? | **No.** A monkey throwing darts each "
            f"week beats the momentum rotation **about half the time** (p = "
            f"{R['placebo_p_total']:.2f}). The \"signal\" adds nothing. |\n\n"
            "> The memecoin-season *phenomenon* is half-real (survivors did outrun BTC). The "
            "memecoin-season *strategy* is a mirage — it's the worst of every world."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"In a bull market, Bitcoin is the slow money. The real gains are in memecoins "
            "— DOGE, SHIB and their cousins run 10-100× while BTC merely doubles. You don't "
            "even need to pick the winner ahead of time: just rotate into whatever's pumping "
            "and ride the momentum. Trend-following on steroids.\"*\n\n"
            "It's a real observation dressed as a strategy. Memecoins *are* higher-beta than "
            "Bitcoin, and in euphoric windows they genuinely scream past it. The leap of faith "
            "is that a *mechanical momentum rotation* can convert that raw volatility into "
            "money you actually keep."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be the easiest trade in finance: no fundamentals, no "
            "valuation, just \"buy what's going up\" in the highest-octane corner of crypto. "
            "Every cycle, thousands of retail traders bet real money on exactly this thesis. "
            "The stakes are whether \"memecoin season\" is a *tradable edge* or just a "
            "survivorship-flavoured story people tell after the fact about the handful of coins "
            "that didn't go to zero."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The universe.** Bitcoin plus the two memecoins big enough to still be around: "
            "DOGE and SHIB. (That choice is already generous — see the survivorship warning.)\n"
            "- **The rotation.** Every Friday, hold whichever of the three had the best "
            "trailing 4-week return; hold it the *next* week (no peeking); pay a realistic "
            "30 bps each time you switch.\n"
            "- **The race.** Compare total return, risk-adjusted return (Sharpe) and worst "
            "drawdown against just holding Bitcoin, and against a naive equal-weight basket.\n"
            "- **The luck check.** Would picking a coin *at random* every week have done just "
            "as well? If yes, the momentum \"signal\" is worthless."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the steelman: did the memecoins really outrun BTC?** Here are all three, "
            "rebased to 100 at the start of the common window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reb = DAILY / DAILY.iloc[0] * 100\n"
            "else:\n"
            "    reb = None\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.8))\n"
            "if HAVE_REAL:\n"
            "    ax.plot(reb.index, reb['BTC'], color=AMBER, lw=1.8, label='BTC')\n"
            "    ax.plot(reb.index, reb['DOGE'], color=BLUE, lw=1.3, label='DOGE')\n"
            "    ax.plot(reb.index, reb['SHIB'], color=RED, lw=1.3, label='SHIB')\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_ylabel('growth of 100 (log scale)')\n"
            "else:\n"
            "    ax.bar(['BTC','DOGE','SHIB'], [R['bh_btc'], R['bh_doge'], R['bh_shib']],\n"
            "           color=[AMBER, BLUE, RED])\n"
            "    ax.set_ylabel('total return, %')\n"
            "ax.set_title('The common window starts April 2021 — right at the memecoin blow-off top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"buy & hold over the window: BTC {R['bh_btc']:+d}%, DOGE {R['bh_doge']:+d}%, SHIB {R['bh_shib']:+d}%\")"
        ),
        md(
            f"Look closely: only **SHIB** (+{R['bh_shib']}%) actually beat holding cash. **DOGE** "
            f"lost **{abs(R['bh_doge'])}%** — because the window starts in April 2021, right at "
            "DOGE's mania peak, and it never came back. \"Memecoins beat Bitcoin\" turns out to "
            "mean \"*the one that kept going up* beat Bitcoin\" — which you only know in "
            "hindsight. Now: can a momentum rotation navigate this?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    net = st.run_rotation(WK, lookback=4, cost_bps=30.0)\n"
            "    btc = st.btc_hodl(WK); ew = st.equal_weight(WK)\n"
            "    w_rot = (1+net['net_ret']).cumprod(); w_btc = (1+btc.reindex(net['net_ret'].index)).cumprod()\n"
            "    w_ew  = (1+ew.reindex(net['net_ret'].index)).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(10.5, 4.8))\n"
            "    ax.plot(w_rot.index, w_rot, color=RED, lw=1.8, label=f\"momentum rotation (net)  {(w_rot.iloc[-1]-1)*100:+.0f}%\")\n"
            "    ax.plot(w_btc.index, w_btc, color=AMBER, lw=1.6, label=f\"BTC hold  {(w_btc.iloc[-1]-1)*100:+.0f}%\")\n"
            "    ax.plot(w_ew.index, w_ew, color=GREEN, lw=1.6, label=f\"equal-weight basket  {(w_ew.iloc[-1]-1)*100:+.0f}%\")\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of 1 (log)')\n"
            "else:\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "    ax.bar(['rotation\\n(net)','BTC hold','equal-weight'], [R['rot_total'], R['btc_total'], R['ew_total']],\n"
            "           color=[RED, AMBER, GREEN])\n"
            "    ax.set_ylabel('total return, %')\n"
            "ax.set_title('The rotation finishes dead last — below even Bitcoin')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"rotation net {R['rot_total']:+.0f}%  |  BTC {R['btc_total']:+.0f}%  |  equal-weight +{R['ew_total']:.0f}%\")"
        ),
        md(
            f"The momentum rotation ends at **{R['rot_total']:.0f}%** — worse than simply "
            f"holding Bitcoin (**{R['btc_total']:.0f}%**), and a universe away from just "
            f"equal-weighting the three coins (**+{R['ew_total']:.0f}%**). The strategy that was "
            "supposed to *harvest* the mania managed to lose money in one of the great "
            "speculative runs of all time. Why? **Because it chases.**"
        ),
        code(
            "sweep = R['cost_sweep']\n"
            "cs = sorted(sweep); tot = [sweep[c][0] for c in cs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(cs, tot, 'o-', color=RED, lw=2)\n"
            "ax.axhline(R['btc_total'], ls='--', color=AMBER, label=f\"BTC hold ({R['btc_total']:+.0f}%)\")\n"
            "for c, v in zip(cs, tot): ax.annotate(f'{v:+d}%', (c, v), textcoords='offset points', xytext=(0,8), ha='center')\n"
            "ax.set_xlabel('trading cost (bps per leg)'); ax.set_ylabel('rotation net total return, %')\n"
            "ax.set_title('Costs make it worse — but it loses 60% even for FREE')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "That's the tell: at **zero** trading cost the rotation still ends around "
            f"**{R['rot_gross_total']}%**. This isn't a good strategy ruined by fees — it's a "
            "strategy that *structurally* buys whatever just spiked (near its local top) and "
            "sells whatever just dropped (near its local bottom), in assets that swing "
            f"**±{R['rot_vol']:.0f}% a year**. The volatility itself is the tax. "
            "**Last check: is the momentum signal even better than random?**"
        ),
        code(
            "labels = ['momentum\\nrotation', 'random\\n(median)']\n"
            "vals = [R['rot_total'], R['rand_median_total']]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(labels, vals, color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.0f}%', (i, v), ha='center', va='top')\n"
            "ax.set_ylabel('net total return, %')\n"
            "ax.set_title(f\"A coin flip beats momentum {R['placebo_p_total']*100:.0f}% of the time\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"momentum {R['rot_total']:+.0f}%  vs random-rotation median {R['rand_median_total']:+d}% \"\n"
            "      f\"| p(random >= momentum) = {R['placebo_p_total']:.3f}\")"
        ),
        md(
            f"Picking a coin **at random** every week does about as well as the momentum rule — "
            f"a random calendar matches or beats it **{R['placebo_p_total']*100:.0f}% of the "
            "time**. The \"momentum\" adds nothing you couldn't get from a dartboard. The signal "
            "is noise."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The rotation's edge over Bitcoin is statistically zero "
            f"(*t* = {R['excess_t']:.2f}), a random coin-flip does as well "
            f"(*p* = {R['placebo_p_total']:.2f}), and even the raw \"memecoins won\" claim is "
            "just one survivor (SHIB) carrying two losers.\n"
            "- **Tradability — Mirage.** Net of a mild 30 bps it returns "
            f"**{R['rot_total']:.0f}%** vs BTC **{R['btc_total']:.0f}%** and an equal-weight "
            f"basket **+{R['ew_total']:.0f}%**, with an **{abs(R['rot_dd']):.0f}%** drawdown — "
            "and it loses money even at zero cost.\n"
            "- **\"Beats a coin flip?\" — Busted.** No."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The survivorship point is the whole point.** The only thing that \"worked\" "
            "here was owning SHIB from the start — and in April 2021 SHIB was one of ten "
            "thousand indistinguishable dog-coins, most now worthless. A backtest on the "
            "survivors flatters *any* buy-and-hold and still can't rescue the rotation.\n"
            "- **What might change the picture:** a proper *cash* exit (sit out when all "
            "momentum is negative) softens the drawdown, but on this tape it still trails "
            "buy-and-hold — try it in the quants notebook. A slower lookback? A bigger, "
            "point-in-time memecoin universe including the dead ones? (That last one would "
            "make the numbers *worse*, not better.)\n\n"
            "*Think a different lookback, a cash filter, or a wider universe cracks it? Fork "
            "the repo and try — the engine is ~250 lines and the whole thing reruns offline.*"
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
            "# Memecoin-Season — a quantitative teardown 🔬\n"
            "### A weekly top-momentum rotation over BTC/DOGE/SHIB · excess-return *t* · a "
            "random-rotation placebo · the volatility tax · a persistence-planting control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — a mechanical momentum rotation across Bitcoin and the surviving "
            "memecoins beats holding Bitcoin, net of costs — is tested as a weekly long-one "
            "rotation with a one-week execution lag, benchmarked against BTC HODL and an "
            "equal-weight basket.\n\n"
            "> ⚠️ **Data note.** BTC/DOGE/SHIB daily close, yfinance, resampled to Friday-close "
            f"weekly bars. Common window **{R['win_lo']} → {R['win_hi']}** ({R['n_weeks']} "
            f"weekly bars, {R['n_rot']} rotation weeks; fingerprint `{R['fp']}`). SHIB's "
            "sub-1e-10 launch price stores as 0.0 in yfinance, so the tradable universe only "
            "begins in April 2021 — a short, mania-dominated sample, said loudly. **"
            "Survivorship is named on the Signal axis:** DOGE & SHIB are the two survivors of "
            "thousands; every return is an ex-post upper bound. Methods in "
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
            f"| **Signal** | `NONE` | rotation-minus-BTC weekly excess *t* = **{R['excess_t']:.2f}** "
            f"(n={R['n_rot']}); random-rotation placebo *p* = **{R['placebo_p_total']:.2f}** |\n"
            f"| **Tradability** | `MIRAGE` | net **{R['rot_total']:.0f}%** vs BTC "
            f"**{R['btc_total']:.0f}%** vs equal-weight **+{R['ew_total']:.0f}%**; "
            f"**{R['rot_gross_total']}%** even at zero cost; maxDD **{R['rot_dd']}%** |\n"
            f"| **Beats a coin flip?** | `BUSTED` | a random weekly coin pick matches/beats it "
            f"{R['placebo_p_total']*100:.0f}% of the time |\n\n"
            "> 💡 In plain words: the rotation is dominated by the two things it was supposed to "
            "beat (BTC and a naive basket), its signal is indistinguishable from noise, and its "
            "one flattering ingredient — owning SHIB — is pure hindsight."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the universe be $U = \\{\\text{BTC}, \\text{DOGE}, \\text{SHIB}\\}$, weekly "
            "returns $r_{i,t}$, and the trailing-$L$ momentum $m_{i,t} = P_{i,t}/P_{i,t-L}-1$ "
            "known at the close of week $t$. The rotation holds "
            "$a_t = \\arg\\max_i m_{i,t}$ over week $t{+}1$. The claim:\n\n"
            "- **H₁ (edge over BTC).** $E[r_{a_t, t+1} - r_{\\text{BTC}, t+1}] > 0$, net of costs.\n"
            "- **H₂ (the signal matters).** the rotation beats a random weekly pick from $U$.\n\n"
            "We would call this **Real** only if the weekly excess-over-BTC series clears "
            "$|t| \\ge 2$ *and* the placebo rejects random allocation. Neither happens."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the survivorship and volatility framing\n\n"
            "Two structural facts dominate everything below, so we state them before any "
            "number:\n\n"
            "1. **Survivorship (Signal axis).** Restricting the memecoin sleeve to DOGE and "
            "SHIB — chosen *because* they survived — inflates every buy-and-hold leg. The "
            "equal-weight basket's **+%d%%** is itself an artefact: it is what you'd have made "
            "*if you had known in 2021 which two of ten thousand coins to hold*. It is a "
            "benchmark the rotation still loses to, not an achievable target.\n"
            "2. **Volatility tax.** Memecoin weekly vol is enormous. A rotation that ends up "
            "**±%d%%/yr** volatile suffers a geometric-vs-arithmetic gap of order "
            "$-\\tfrac12\\sigma^2$ per period — which is exactly how a strategy can post a "
            "*positive* average weekly excess return and still compound to a deep loss."
            % (R["ew_total"], R["rot_vol"])
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** BTC/DOGE/SHIB daily close, yfinance, {R['daily_n']:,} rows "
            f"{R['daily_lo']} → {R['daily_hi']}; Friday-close weekly bars, {R['n_weeks']} of "
            f"them {R['win_lo']} → {R['win_hi']}. As-of 2026-06-30 (last complete month; the "
            "partial final week is dropped).\n"
            "- **Signal.** Trailing 4-week return per asset, known at the week's close.\n"
            "- **Execution.** Hold $\\arg\\max$ momentum over the **next** week "
            "($\\text{choice}.\\text{shift}(1)$) — the single documented lag, no look-ahead.\n"
            "- **Costs.** 30 bps per leg × NAV; a coin-to-coin switch is two legs. Gross and "
            "net both reported; a 0→100 bps sweep for break-even.\n"
            "- **Benchmarks.** BTC HODL and a weekly-rebalanced equal-weight basket, same window.\n"
            "- **Inference.** Paired weekly excess-over-BTC *t*; a 4,000-seed random-rotation "
            "placebo; a pre-2022/post-2022 split.\n"
            "- **Control.** A 3-asset synthetic world with tunable momentum persistence; the "
            "rotation must beat equal-weight only when persistence is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — rotation vs BTC vs equal-weight\n\n"
            "Growth of 1, net of 30 bps/leg, over the identical window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    net = st.run_rotation(WK, lookback=4, cost_bps=30.0)\n"
            "    gross = st.run_rotation(WK, lookback=4, cost_bps=0.0)\n"
            "    btc = st.btc_hodl(WK); ew = st.equal_weight(WK)\n"
            "    sn = st.summarize(net['net_ret']); sb = st.summarize(btc); se = st.summarize(ew)\n"
            "    sg = st.summarize(gross['net_ret'])\n"
            "    idx = net['net_ret'].index\n"
            "    w_rot = (1+net['net_ret']).cumprod(); w_btc=(1+btc.reindex(idx)).cumprod(); w_ew=(1+ew.reindex(idx)).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(10.6, 4.7))\n"
            "    ax.plot(w_rot.index, w_rot, color=RED, lw=1.9, label=f\"rotation net ({sn['total_pct']:+.0f}%, Sh {sn['sharpe']:.2f})\")\n"
            "    ax.plot(w_btc.index, w_btc, color=AMBER, lw=1.6, label=f\"BTC ({sb['total_pct']:+.0f}%, Sh {sb['sharpe']:.2f})\")\n"
            "    ax.plot(w_ew.index, w_ew, color=GREEN, lw=1.6, label=f\"equal-weight ({se['total_pct']:+.0f}%, Sh {se['sharpe']:.2f})\")\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of 1 (log)')\n"
            "    ax.set_title('Net of costs, the rotation finishes below everything'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"rotation net {sn['total_pct']:+.1f}% (Sh {sn['sharpe']:.2f}, vol {sn['vol_pct']:.0f}%, maxDD {sn['maxdd_pct']:.0f}%)\")\n"
            "    print(f\"BTC {sb['total_pct']:+.1f}% (Sh {sb['sharpe']:.2f}) | equal-weight {se['total_pct']:+.1f}% (Sh {se['sharpe']:.2f})\")\n"
            "else:\n"
            "    fig, (a1,a2) = plt.subplots(1,2, figsize=(11,4.4))\n"
            "    a1.bar(['rot','BTC','EW'], [R['rot_total'],R['btc_total'],R['ew_total']], color=[RED,AMBER,GREEN])\n"
            "    a1.set_ylabel('total %'); a1.set_title('total return')\n"
            "    a2.bar(['rot','BTC','EW'], [R['rot_sharpe'],R['btc_sharpe'],R['ew_sharpe']], color=[RED,AMBER,GREEN])\n"
            "    a2.set_ylabel('Sharpe'); a2.set_title('Sharpe'); plt.tight_layout(); plt.show()\n"
            "    print('rotation net', R['rot_total'], 'BTC', R['btc_total'], 'EW', R['ew_total'])"
        ),
        md(
            f"> 💡 In plain words: the rotation's Sharpe ({R['rot_sharpe']:.2f}) is barely above "
            f"BTC's ({R['btc_sharpe']:.2f}) and well below the equal-weight basket's "
            f"({R['ew_sharpe']:.2f}), while its total return ({R['rot_total']:.0f}%) is the "
            "worst of the three. It gives up return *and* runs the highest volatility "
            f"(**{R['rot_vol']:.0f}%/yr** vs BTC's {R['btc_vol']}%) — a strictly dominated "
            "outcome."
        ),
        md(
            "### 4b · Is the edge over BTC real? — and the volatility tax\n\n"
            "The paired weekly excess-over-BTC series, and the arithmetic-vs-geometric gap that "
            "explains the paradox."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex = st.excess_tstat(net['net_ret'], btc.reindex(net['net_ret'].index))\n"
            "    mean_wk, tval = ex['mean_excess_pct'], ex['t']\n"
            "    vol = sn['vol_pct']\n"
            "else:\n"
            "    mean_wk, tval, vol = R['excess_mean_wk'], R['excess_t'], R['rot_vol']\n"
            "sig_w = vol/np.sqrt(52)/100\n"
            "geo_drag = -0.5*sig_w**2*100\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['mean weekly\\nexcess vs BTC','vol drag\\n(-½σ² per wk)'], [mean_wk, geo_drag],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "for i,v in enumerate([mean_wk, geo_drag]): ax.annotate(f'{v:+.2f}%', (i,v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('% per week')\n"
            "ax.set_title(f'Positive weekly edge (t={tval:.2f}), but the vol tax eats it whole')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean weekly excess vs BTC = {mean_wk:+.3f}%  t = {tval:+.2f}  (|t|>=2 needed)')\n"
            "print(f'weekly vol drag ~ {geo_drag:+.2f}%/wk — larger than the edge, so it compounds to a loss')"
        ),
        md(
            f"> 💡 In plain words: the rotation's *average* weekly return actually beats BTC by "
            f"**{R['excess_mean_wk']:.2f}%** — but the *t* is only **{R['excess_t']:.2f}**, "
            "nowhere near the |t| ≥ 2 bar, so even that tiny gap is statistically "
            "indistinguishable from zero. And because the strategy is so volatile, the "
            "$-\\tfrac12\\sigma^2$ compounding drag per week is *larger* than the edge — which is "
            "precisely how a positive-mean strategy compounds to a catastrophic loss. Volatility "
            "isn't just risk here; it's a direct, mechanical cost."
        ),
        md(
            "### 4c · The random-rotation placebo — does momentum add anything?\n\n"
            "4,000 seeds, each picking a coin uniformly at random every week, same costs and "
            "lag. If the momentum rotation can't beat this, the signal is noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_rotation_placebo(WK, cost_bps=30.0, n_seeds=1500)  # lighter in-notebook\n"
            "    p_tot, p_shp = pl['p_total'], pl['p_sharpe']\n"
            "    mom_t, rnd_t = pl['mom_total_pct'], pl['rand_total_median_pct']\n"
            "else:\n"
            "    p_tot, p_shp = R['placebo_p_total'], R['placebo_p_sharpe']\n"
            "    mom_t, rnd_t = R['rot_total'], R['rand_median_total']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['momentum','random (median)'], [mom_t, rnd_t], color=[RED, GREY], width=.5)\n"
            "for i,v in enumerate([mom_t, rnd_t]): ax.annotate(f'{v:+.0f}%', (i,v), ha='center', va='top')\n"
            "ax.set_ylabel('net total return, %')\n"
            "ax.set_title(f'p(random total >= momentum) = {p_tot:.3f}  ·  p(random Sharpe >= momentum) = {p_shp:.3f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'canonical (results.md, {R[\"placebo_seeds\"]:,} seeds): p_total = {R[\"placebo_p_total\"]:.3f}, p_sharpe = {R[\"placebo_p_sharpe\"]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: a random weekly coin-pick matches or beats the momentum "
            f"rotation on total return **{R['placebo_p_total']*100:.0f}%** of the time and on "
            f"Sharpe **{R['placebo_p_sharpe']*100:.0f}%** of the time. The momentum ranking is "
            "not extracting any exploitable structure — you could replace it with a dice roll "
            "and do no worse."
        ),
        md(
            "### 4d · Where does any apparent edge live? — the 2021 mania vs after\n\n"
            "Split at 2022-01-01. A single euphoric sliver can carry an entire crypto backtest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = st.subperiod_table(WK, cut='2022-01-01', cost_bps=30.0)\n"
            "    seg = list(sub.index); rot = list(sub['rot_total_pct']); bt = list(sub['btc_total_pct']); ts = list(sub['excess_t'])\n"
            "else:\n"
            "    seg = ['mania (< 2022-01)','after (>= 2022-01)']\n"
            "    rot = [R['mania_rot_total'], R['after_rot_total']]; bt = [R['mania_btc_total'], R['after_btc_total']]\n"
            "    ts = [R['mania_t'], R['after_t']]\n"
            "x = np.arange(len(seg)); w=.36\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x-w/2, rot, width=w, color=RED, label='rotation (net)')\n"
            "ax.bar(x+w/2, bt, width=w, color=AMBER, label='BTC hold')\n"
            "for i,v in enumerate(rot): ax.annotate(f'{v:+.0f}%', (x[i]-w/2, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "for i,v in enumerate(bt): ax.annotate(f'{v:+.0f}%', (x[i]+w/2, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{s}\\nexcess t={t:+.2f}' for s,t in zip(seg,ts)])\n"
            "ax.set_ylabel('total return, %'); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_title('A whisker ahead in the 2021 mania, then it collapses'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: in the 2021 mania ({R['mania_weeks']} weeks) the rotation "
            f"edged BTC (+{R['mania_rot_total']:.0f}% vs {R['mania_btc_total']:.0f}%) but with "
            f"*t* = {R['mania_t']:.2f} — not significant. After 2022 ({R['after_weeks']} weeks) "
            f"it lost **{abs(R['after_rot_total']):.0f}%** while BTC *gained* "
            f"**+{R['after_btc_total']:.0f}%** (*t* = {R['after_t']:.2f}). There is no regime in "
            "which the edge is both positive and certified; the flattering half is a tiny, "
            "noisy, in-sample window."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A 3-asset synthetic weekly world with TUNABLE momentum persistence. Null "
            "(persistence = 0, pure random walks) checked over **20 seeds**; a planted-"
            "persistence world must be caught."
        ),
        code(
            "null_ts = []\n"
            "for s in range(20):\n"
            "    w = data.synthetic_world(persistence=0.0, seed=766+s)\n"
            "    null_ts.append(st.momentum_edge_from_returns(w))\n"
            "null_ts = np.asarray(null_ts)\n"
            "w = data.synthetic_world(persistence=0.35, seed=766)\n"
            "planted_t = st.momentum_edge_from_returns(w)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20)+np.linspace(-.12,.12,20), null_ts, color=GREY, s=40, label='null worlds (persistence=0), 20 seeds')\n"
            "ax.scatter([1],[planted_t], color=GREEN, s=90, zorder=5, label='planted persistence=0.35')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('rotation-minus-equal-weight excess t')\n"
            "ax.set_title('Control: the rotation harvests momentum WHEN it exists'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts)>=2).sum()}/20  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: when momentum persistence is genuinely present, the rotation "
            f"detects and harvests it cleanly (*t* = {R['syn_planted_t']:.2f}). On the null it "
            f"sits at zero (mean *t* = {R['syn_null_mean']:.2f}). So the engine works — which "
            "means the real-tape verdict isn't a broken backtest: memecoin weekly returns "
            "simply don't carry the exploitable week-to-week momentum the folklore assumes. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the weekly excess-over-BTC *t* is **{R['excess_t']:.2f}** "
            f"(n={R['n_rot']}), a random-rotation placebo matches/beats the strategy "
            f"({R['placebo_p_total']*100:.0f}% on return, {R['placebo_p_sharpe']*100:.0f}% on "
            "Sharpe), and no sub-period shows a certified edge. The momentum signal carries no "
            "exploitable information on this tape.\n"
            f"- **Tradability `MIRAGE`** — net of a mild 30 bps/leg the rotation returns "
            f"**{R['rot_total']:.0f}%** against BTC **{R['btc_total']:.0f}%** and equal-weight "
            f"**+{R['ew_total']:.0f}%**, with a **{R['rot_dd']}%** drawdown and "
            f"**{R['rot_vol']:.0f}%/yr** volatility; it loses **{R['rot_gross_total']}%** even "
            "at zero cost, so the failure is structural, not fee-driven. And its one flattering "
            "leg — owning SHIB — is survivorship (2 winners of thousands).\n"
            "- **\"Beats a coin flip?\" `BUSTED`** — a dartboard does as well."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The volatility tax is the real lesson.** In ultra-high-vol assets, a strategy "
            "with a *positive* average edge can still compound to ruin because "
            "$-\\tfrac12\\sigma^2$ swamps the mean. Any \"just ride the momentum\" pitch in "
            "memecoins runs headlong into this; the fix is sizing (vol-targeting), not "
            "signal-picking — and vol-targeting a −t signal buys you nothing.\n"
            "- **A cash-exit variant** (`cash_option=True` in `run_rotation`) reduces the "
            "drawdown but, on this survivorship-flattered tape, still trails BTC — a worthwhile "
            "fork to run and report.\n"
            "- **The honest universe is point-in-time.** Including the thousands of dead "
            "memecoins (a proper survivorship-free panel via the desk's opt-in guard) would "
            "push every number strictly *worse*; the study already uses the generous survivor "
            "universe and the claim still fails.\n\n"
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
