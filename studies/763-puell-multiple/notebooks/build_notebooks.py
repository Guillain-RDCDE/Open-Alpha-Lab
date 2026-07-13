"""Generate the two narrative notebooks for Study 763 (Puell-Multiple).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached BTC-USD tape under
../_cache/ (no network) and reconstruct the Puell Multiple from it plus the hardcoded halving
schedule; if the cache is absent they fall back to the frozen headline numbers in ``R`` (a mirror
of docs/results.md). The synthetic positive control always runs with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (BTC-USD yfinance 2014-09-17 ->
# 2026-06-30; Puell reconstructed from price + halving schedule; issuance-only; as-of 2026-06-30).
R = dict(
    btc_lo="2014-09-17", btc_hi="2026-06-30", btc_n=4305, fp_btc="9529d5277775",
    puell_lo="2015-09-16", puell_hi="2026-06-30", puell_n=3941,
    puell_min=0.40, puell_med=1.12, puell_max=5.95,
    high=4.0, low=0.5,
    # predictive regression: horizon -> (slope, HAC t, R2, n)
    reg={
        30: (0.0284, 0.67, 0.0037, 3911),
        90: (0.0027, 0.02, 0.0000, 3851),
        180: (-0.1616, -0.74, 0.0155, 3761),
    },
    horse_puell_t=-1.76, horse_mom_t=1.70,   # h=90, +trailing-180d price momentum
    # band forward returns: horizon -> {band: (mean%, median%, hit, welch_t, placebo_p, n)}
    band={
        30: dict(bottom=(6.79, 6.54, 0.75, 0.08, 0.479, 75),
                 neutral=(6.84, 3.03, 0.57, 0.25, None, 3814),
                 top=(-17.13, -20.30, 0.14, -4.60, 0.000, 22)),
        90: dict(bottom=(2.01, -1.81, 0.44, -11.10, 1.000, 75),
                 neutral=(24.03, 11.17, 0.61, 0.65, None, 3754),
                 top=(-44.18, -46.76, 0.00, -19.14, 0.000, 22)),
        180: dict(bottom=(60.93, 49.62, 0.61, 0.67, 0.304, 75),
                  neutral=(55.59, 31.69, 0.66, 0.22, None, 3664),
                  top=(-56.89, -59.87, 0.00, -43.47, 0.000, 22)),
    },
    top_days=22, top_episode="2017-12-04 -> 2018-01-06",   # ONE blow-off cluster (3 runs)
    bottom_days=75, bottom_episodes=3,                     # 2018 bear, 2020 halving-day, 2022 FTX
    placebo_draws=20000,
    # timer vs buy-and-hold (sell when Puell>=4, 10 bps)
    years=10.79, n_flips=7, exposure=99.42,
    strat_total=24841.1, strat_cagr=66.80, strat_sharpe=1.114,
    bh_total=25461.4, bh_cagr=67.18, bh_sharpe=1.107,
    excess_cagr=-0.97, excess_t=-0.21,
    sweep={3.0: (18921.1, 97.6, 1.09), 3.5: (20377.3, 98.9, 1.09),
           4.0: (24841.1, 99.4, 1.11), 4.5: (22944.2, 99.6, 1.10)},
    # synthetic control
    syn_planted_slope=-11.78, syn_planted_t=-26.09,
    syn_null_mean=0.06, syn_null_sd=1.22, syn_null_fire=1,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Times_tops_%26_bottoms%3F: Busted](https://img.shields.io/badge/Times_tops_%26_bottoms%3F-Busted-8b949e?style=flat-square)\n\n"
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

from puell_multiple import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    BTC = data.load_btc()
    PM = data.puell_multiple(BTC).dropna()
else:
    BTC = PM = None
print("real cache present:", HAVE_REAL,
      "| Puell points:", (0 if PM is None else len(PM)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Miner revenue is running hot. Is that when you sell? ⛏️\n"
            "### The Puell Multiple — a famous on-chain \"top/bottom\" gauge you can rebuild "
            "*exactly* from price alone\n\n"
            + BADGES +
            "Every day, the Bitcoin network mints new coins and hands them to miners, who mostly "
            "sell them to pay for electricity and machines. The **Puell Multiple** asks a simple "
            "question: is that daily flow of new-coin revenue running **hot or cold** versus its "
            "own past year?\n\n"
            "The folklore: when miner revenue is way above its yearly average (Puell **> 4**), "
            "everyone's euphoric and it's a **top — sell**. When it's way below (Puell **< 0.5**), "
            "miners are getting crushed and capitulating — a **bottom — buy**. It *sounds* deep "
            "because it's \"on-chain,\" not just price.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo and the horse "
            "race? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> 🔬 **The twist we'll keep coming back to.** The Puell Multiple's only non-price "
            "ingredient — how many coins are minted each day — is a *fixed, known* schedule (it "
            "halves every four years). So we can rebuild the whole thing from BTC's price and a "
            "hardcoded halving table, no mystery data feed required. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a high/low Puell reading predict what BTC does next? | **No.** In a proper "
            f"regression it explains ~nothing (best statistic **{R['reg'][30][1]:.2f}**, far below "
            "the bar), and it isn't even reliably pointing the \"contrarian\" way. |\n"
            "| Does \"buy the bottom\" (Puell < 0.5) work? | **No — it's worse than a random "
            f"day.** Buying deep lows earned **+{R['band'][90]['bottom'][0]:.0f}%** over the next "
            f"90 days versus **+{R['band'][90]['neutral'][0]:.0f}%** for an ordinary day. |\n"
            "| Does \"sell the top\" (Puell > 4) work? | **Once.** It fired on essentially a "
            "single event — the **2017 blow-off top** — and never cleanly again. One data point "
            "is a story, not a signal. |\n"
            "| Could you have timed the market with it? | **No.** The buy-low/sell-high rule ends "
            "up **99% invested** and actually **loses** to just holding BTC "
            f"(**+{R['strat_total']:,.0f}%** vs **+{R['bh_total']:,.0f}%**). |\n\n"
            "> A metric that turns out to be *price divided by its own one-year average* can't "
            "tell you much that the price chart didn't already."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Miners are the market's constant sellers. When their daily revenue is far above "
            "its 365-day average, the market is overheated — a top. When it's far below, miners "
            "are capitulating — a bottom. Watch the Puell Multiple: sell above 4, buy below "
            "0.5.\"*\n\n"
            "It's a genuinely clever framing — it ties price to the **supply side** (issuance), "
            "which is fixed by protocol and steps down at every halving. David Puell's original "
            "charts light up the 2013 and 2017 tops and the 2015 and 2018 bottoms."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, the Puell Multiple would be one of the cleanest cycle-timing tools in "
            "crypto — a supply-anchored \"are we hot or cold\" dial that beats staring at the "
            "price. It's on every on-chain dashboard and gets cited at every cycle. The stakes: "
            "if buying low-Puell and selling high-Puell genuinely beats holding, that's real, "
            "bankable timing alpha in the most liquid crypto asset. If it doesn't, it's one more "
            "chart that *looks* rigorous because it says \"on-chain.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Rebuild it honestly.** Daily miner revenue = (coins minted that day) × price. "
            "Coins minted per day is the **known halving schedule** (50 → 25 → 12.5 → 6.25 → "
            "3.125 BTC). Divide today's revenue by its trailing 365-day average — that's the "
            "Puell Multiple, reconstructed from price + a hardcoded table.\n"
            "- **The prediction test.** Does today's Puell reading forecast BTC's return over the "
            "next 1/3/6 months?\n"
            "- **The bands test.** After Puell > 4 (\"top\") and Puell < 0.5 (\"bottom\"), what "
            "did BTC *actually* do — and is that better than a random day?\n"
            "- **The trade test.** If you'd sold to cash whenever Puell > 4 and held otherwise, "
            "would you have beaten simply buying and holding?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the reconstructed Puell Multiple** with the classic sell/buy bands. "
            "Notice how rarely it actually reaches the extremes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pm = PM\n"
            "else:\n"
            "    pm = None\n"
            "fig, ax = plt.subplots(figsize=(10.6, 4.6))\n"
            "if pm is not None:\n"
            "    ax.plot(pm.index, pm.values, color=GREY, lw=1.1, label='Puell Multiple (reconstructed)')\n"
            "ax.axhline(R['high'], color=RED, ls='--', lw=1.3, label=f\"sell band (> {R['high']})\")\n"
            "ax.axhline(R['low'], color=GREEN, ls='--', lw=1.3, label=f\"buy band (< {R['low']})\")\n"
            "ax.axhline(1.0, color='k', lw=.7, alpha=.5)\n"
            "if pm is not None:\n"
            "    ax.fill_between(pm.index, R['high'], pm.values, where=(pm.values >= R['high']), color=RED, alpha=.35)\n"
            "    ax.fill_between(pm.index, R['low'], pm.values, where=(pm.values <= R['low']), color=GREEN, alpha=.35)\n"
            "ax.set_ylabel('Puell Multiple'); ax.set_title('The Puell Multiple: it clears 4 basically once (2017)')\n"
            "ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "print(f\"reconstructed from BTC price + halving schedule; range {R['puell_min']:.2f} .. {R['puell_max']:.2f}\")\n"
            "print(f\"days in SELL band (>4): {R['top_days']}  (all one 2017-18 blow-off)  |  \"\n"
            "      f\"days in BUY band (<0.5): {R['bottom_days']}  (~{R['bottom_episodes']} episodes)\")"
        ),
        md(
            "The red \"sell\" zone is essentially a **single spike in late 2017**. That's the "
            "first honest problem: a rule that only triggers once can't be tested like a rule "
            "that triggers often. Now — **does the reading predict what comes next?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tab = st.state_forward_stats(PM, BTC, horizon=90, high=R['high'], low=R['low'])\n"
            "    order = ['bottom (buy)', 'neutral', 'top (sell)']\n"
            "    means = [tab.loc[b, 'mean_pct'] for b in order]\n"
            "else:\n"
            "    means = [R['band'][90]['bottom'][0], R['band'][90]['neutral'][0], R['band'][90]['top'][0]]\n"
            "labels = ['bottom band\\n(Puell < 0.5)\\n\"BUY\"', 'a neutral day', 'top band\\n(Puell > 4)\\n\"SELL\"']\n"
            "cols = [GREEN, GREY, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(labels, means, color=cols, width=.6)\n"
            "for i, v in enumerate(means): ax.annotate(f'{v:+.0f}%', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_ylabel('average BTC return over the next 90 days')\n"
            "ax.set_title('The \"buy\" signal does WORSE than doing nothing')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Look at the green bar. After a \"buy the bottom\" signal, BTC returned "
            f"**+{R['band'][90]['bottom'][0]:.0f}%** over 90 days — *less* than the "
            f"**+{R['band'][90]['neutral'][0]:.0f}%** you'd get on an ordinary day. The "
            "capitulation-buy half of the folklore is empty. The red \"sell\" bar is dramatically "
            "negative — but remember, that's the one 2017 blow-off. **So would the trade have "
            "paid?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tb = st.backtest_timing(PM, BTC, high=R['high'], cost_bps=10.0)\n"
            "    st_tot, bh_tot = tb['strat_total_pct'], tb['bh_total_pct']\n"
            "    st_shp, bh_shp = tb['strat_sharpe'], tb['bh_sharpe']; expo = tb['exposure_pct']\n"
            "else:\n"
            "    st_tot, bh_tot = R['strat_total'], R['bh_total']\n"
            "    st_shp, bh_shp = R['strat_sharpe'], R['bh_sharpe']; expo = R['exposure']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "bars = ax.bar(['Puell timer\\n(sell when > 4)', 'just buy & hold'], [st_tot, bh_tot],\n"
            "              color=[AMBER, GREEN], width=.55)\n"
            "for i, v in enumerate([st_tot, bh_tot]): ax.annotate(f'{v:,.0f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('total return since the metric began')\n"
            "ax.set_title(f'The timer is {expo:.0f}% invested anyway — and quietly loses')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Puell timer: +{st_tot:,.0f}% (Sharpe {st_shp:.2f}, invested {expo:.1f}% of the time)')\n"
            "print(f'buy & hold : +{bh_tot:,.0f}% (Sharpe {bh_shp:.2f}, invested 100%)')"
        ),
        md(
            f"There's the whole story. Because Puell only crosses 4 at the very peak, the rule "
            f"spends **{R['exposure']:.0f}%** of its life fully invested — it's buy-and-hold "
            "wearing a costume. And in the few weeks it *does* step aside, it re-enters before "
            "the bear market is over, so it ends up with **less** money than just holding "
            f"(**+{R['strat_total']:,.0f}%** vs **+{R['bh_total']:,.0f}%**). No timing edge — a "
            "small timing *cost*."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The Puell reading doesn't forecast BTC's next move; the "
            "\"buy the bottom\" band does worse than a random day; the \"sell the top\" band is "
            "one 2017 event.\n"
            "- **Tradability — Mirage.** The buy-low/sell-high rule is 99% buy-and-hold and "
            "trails it net of costs. Any big-looking return is just being long a coin that went "
            "up 100×.\n"
            "- **\"Times tops & bottoms?\" — Busted.** One top, zero tradable bottoms."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The tell is the reconstruction.** Because the coins-per-day schedule is fixed "
            "and the blocks-per-day constant cancels, *within* a four-year halving epoch the "
            "Puell Multiple is just **price ÷ its own 365-day average**. It's a dressed-up price "
            "ratio; the only extra ingredient is a mechanical ~1-year dip after each halving.\n"
            "- **What would change our mind:** more genuine top signals (we effectively have "
            "one), or a fee-inclusive version that adds real information beyond price — but that "
            "version can't be rebuilt from price alone, so it can't be audited this cleanly.\n"
            "- **Sibling studies:** [293-mvrv-ratio](../../293-mvrv-ratio/) (another on-chain "
            "valuation band — same None/Mirage), [221-mayer-multiple](../../221-mayer-multiple/) "
            "(price ÷ 200-day average — the Puell Multiple's close cousin), "
            "[663-hash-ribbons](../../663-hash-ribbons/) (a hashrate buy signal — Weak/Fragile).\n\n"
            "*Think 4 and 0.5 are the wrong thresholds, or that fees would rescue it? Fork the "
            "repo, rebuild the metric, and try — the whole thing is a pure function of price and a "
            "hardcoded halving table.*"
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
            "# Puell-Multiple — a quantitative teardown 🔬\n"
            "### The exact reconstruction · a HAC forward-return regression + momentum horse race "
            "· a band event study with a random-date placebo · a timer-vs-HODL sweep · a 20-seed "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — David Puell's issuance-revenue oscillator (daily miner issuance value / its "
            "trailing 365-day average) is a contrarian top/bottom timer, sell > 4, buy < 0.5 — is "
            "tested on a **faithful reconstruction**, not a digitised proxy.\n\n"
            "> ⚠️ **Data note.** BTC-USD daily close, yfinance, "
            f"{R['btc_lo']} → {R['btc_hi']} ({R['btc_n']:,} rows, fingerprint `{R['fp_btc']}`). "
            "The Puell Multiple is `144·reward(t)·price(t) / trailing-365d mean` — and the 144 "
            "**cancels** in the ratio, so the only non-price input is the *known halving step "
            "function*. One named approximation: real daily block counts wobble a few % around "
            "144; that largely cancels and never moves a threshold crossing. Issuance-only (the "
            "original definition; fee-inclusive variants are out of scope). Single-survivor named "
            "on the Signal axis. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | forward-return-on-log(Puell) best HAC **t = "
            f"{R['reg'][30][1]:.2f}** (sign not stably contrarian), R² ≈ 0; horse-race t = "
            f"{R['horse_puell_t']:.2f}; buy band **+{R['band'][90]['bottom'][0]:.0f}%** < neutral "
            f"**+{R['band'][90]['neutral'][0]:.0f}%** at 90d |\n"
            f"| **Tradability** | `MIRAGE` | timer **+{R['strat_total']:,.0f}%** vs B&H "
            f"**+{R['bh_total']:,.0f}%**, {R['exposure']:.0f}% invested, excess "
            f"{R['excess_cagr']:.2f}%/yr (HAC t = {R['excess_t']:.2f}) |\n"
            f"| **Times tops & bottoms?** | `BUSTED` | sell band = one 2017 blow-off (n_eff ≈ 1); "
            f"buy band worse than a random day |\n\n"
            "> 💡 In plain words: the metric is a repackaged 365-day price ratio; on the real tape "
            "it neither predicts returns nor beats holding, and its one dramatic number is a "
            "single historical event with a fake-precise *t* bolted on."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, and why the reconstruction is exact\n\n"
            "Let $I_t = b \\cdot R(t) \\cdot P_t$ be daily miner **issuance value**, with $b$ = "
            "blocks/day (144), $R(t)$ the halving-scheduled block reward, $P_t$ the BTC price. "
            "The Puell Multiple is\n\n"
            "$$\\text{Puell}_t = \\frac{I_t}{\\frac{1}{365}\\sum_{k=0}^{364} I_{t-k}} "
            "= \\frac{R(t) P_t}{\\frac{1}{365}\\sum_{k} R(t-k) P_{t-k}}.$$\n\n"
            "The constant $b$ **cancels**. $R(t)$ is a deterministic step function. So the metric "
            "is a *pure function of the price tape and the halving schedule* — a genuine "
            "reconstruction of the canonical issuance-only Puell Multiple, not a proxy. Note the "
            "structural consequence: **within** a halving epoch $R$ is constant, so "
            "$\\text{Puell}_t = P_t / \\overline{P}_{365}$ — a 365-day price ratio. The only thing "
            "$R(t)$ adds is a ~1-year suppression after each halving (the denominator still holds "
            "pre-halving days at double reward). The contrarian claims:\n\n"
            "- **H₁ (predictive).** $E[r_{t\\to t+h}\\mid \\log\\text{Puell}_t]$ has a "
            "**negative** slope with $|t|\\ge 2$.\n"
            "- **H₂ (tradable).** A buy-low/sell-high timer beats buy-and-hold net of costs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two honest hazards to design around. **(a) Overlap.** Forward $h$-day returns on a "
            "daily grid overlap heavily, so OLS *t*-stats are badly inflated; we use "
            "Newey-West (HAC) errors with lag $\\ge h$. **(b) Clustered bands.** The band days "
            "are numerous but *not independent* — the top band is a single 2017-18 episode, so a "
            "Welch *t* over its ~22 days is fake precision. The guard is a **random-date placebo** "
            "(20 seeds × 1,000 draws of the same day-count), reported alongside the Welch *t* with "
            "a loud caveat, and — the ultimate arbiter for H₂ — the **excess-of-buy-and-hold** "
            "return with its own HAC *t*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Metric.** {R['puell_n']:,} daily Puell values {R['puell_lo']} → {R['puell_hi']} "
            f"(min {R['puell_min']:.2f}, median {R['puell_med']:.2f}, max {R['puell_max']:.2f}), "
            "reconstructed from price + the hardcoded halving table.\n"
            "- **Bands.** top = Puell ≥ 4 (sell), bottom = Puell ≤ 0.5 (buy), else neutral — "
            "David Puell's canonical thresholds.\n"
            f"- **Tape.** BTC-USD daily close {R['btc_lo']} → {R['btc_hi']}. As-of "
            f"{R['btc_hi']} (last complete month).\n"
            "- **Signal.** forward 30/90/180-day BTC log-return on log(Puell), HAC lag = horizon; "
            "a horse race adds trailing-180d price momentum.\n"
            "- **Bands.** mean forward return per band, Welch *t* + 20×1,000 random-date placebo.\n"
            "- **Execution.** Puell known at close t; timer exposure applies to t+1 (one-day lag, "
            "documented once); 10 bps one-way × NAV per flip; long/flat, no borrow; price-only.\n"
            "- **Control.** synthetic (Puell, price) world with a tunable contrarian link; the "
            "null must read ~zero across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The reconstruction, and the halving imprint\n\n"
            "The reconstructed metric with its sell/buy bands, plus the block-reward step "
            "underneath it. The ~1-year post-halving suppression is the only thing the metric "
            "adds over a plain 365-day price ratio."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pm = PM; reward = data.reward_at(BTC.index)\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.6, 6.2), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios': [3, 1]})\n"
            "    a1.plot(pm.index, pm.values, color=GREY, lw=1.0)\n"
            "    a1.axhline(R['high'], color=RED, ls='--', lw=1.2); a1.axhline(R['low'], color=GREEN, ls='--', lw=1.2)\n"
            "    a1.fill_between(pm.index, R['high'], pm.values, where=(pm.values >= R['high']), color=RED, alpha=.35)\n"
            "    a1.fill_between(pm.index, R['low'], pm.values, where=(pm.values <= R['low']), color=GREEN, alpha=.35)\n"
            "    a1.set_ylabel('Puell Multiple'); a1.set_title('Reconstructed Puell Multiple + halving-stepped block reward')\n"
            "    a2.step(reward.index, reward.values, where='post', color='k', lw=1.2)\n"
            "    a2.set_ylabel('reward (BTC)'); a2.set_yscale('log')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('(offline) frozen: Puell range', R['puell_min'], '..', R['puell_max'])\n"
            "print(f\"SELL band (>=4): {R['top_days']} days, all one blow-off cluster {R['top_episode']}\")\n"
            "print(f\"BUY band (<=0.5): {R['bottom_days']} days across ~{R['bottom_episodes']} episodes \"\n"
            "      '(2018 bear, 2020 halving-day dip, 2022 FTX bear)')"
        ),
        md(
            "> 💡 In plain words: the sell band is not a repeatable signal — it is essentially the "
            "**single** 2017 top (three runs all inside five weeks). Any statistic computed on it "
            "is really a statistic about one event."
        ),
        md(
            "### 4b · Predictive regression — log(Puell) → forward return (HAC)\n\n"
            "$r_{t\\to t+h} = a + b\\,\\log(\\text{Puell}_t) + e$. A contrarian gauge needs "
            "$b<0$, $|t|\\ge 2$."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hz = [30, 90, 180]\n"
            "    ts = [st.predictive_regression(PM, BTC, horizon=h)['t_puell'] for h in hz]\n"
            "    rc = st.predictive_regression(PM, BTC, horizon=90, add_price_control=True)\n"
            "    puell_t, mom_t = rc['t_puell'], rc['t_price']\n"
            "else:\n"
            "    hz = [30, 90, 180]; ts = [R['reg'][h][1] for h in hz]\n"
            "    puell_t, mom_t = R['horse_puell_t'], R['horse_mom_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "x = np.arange(len(hz))\n"
            "a1.bar(x, ts, color=[RED if abs(t) >= 2 else AMBER for t in ts], width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(ts): a1.annotate(f'{v:+.2f}', (x[i], v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'+{h}d' for h in hz]); a1.set_ylabel('HAC t (slope on log Puell)')\n"
            "a1.set_title('No horizon clears |t| = 2 — and the sign flips')\n"
            "a2.bar(['log(Puell)', 'price momentum'], [puell_t, mom_t], color=[AMBER, GREY], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate([puell_t, mom_t]): a2.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a2.set_ylabel('HAC t'); a2.set_title('Horse race at +90d: Puell adds nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, t in zip(hz, ts): print(f'+{h}d: HAC t = {t:+.2f}')\n"
            "print(f'horse race +90d: Puell t = {puell_t:+.2f}, price-mom t = {mom_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the best statistic is **{R['reg'][30][1]:.2f}** and the slope "
            "isn't even reliably the contrarian sign (positive at 30/90 days, negative at 180). "
            f"Controlling for price momentum, Puell's own *t* is **{R['horse_puell_t']:.2f}** — "
            "still short of 2, and only borderline-negative once the price trend is removed. "
            "There is no robust forward information here."
        ),
        md(
            "### 4c · Band event study — the placebo unmasks the clustering\n\n"
            "Mean forward return per band vs the unconditional distribution, Welch *t*, and the "
            "random-date placebo (right-tailed for buy, left-tailed for sell — the contrarian "
            "direction)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tab = st.state_forward_stats(PM, BTC, horizon=90, high=R['high'], low=R['low'])\n"
            "    rows = [('bottom (buy)', tab.loc['bottom (buy)']), ('top (sell)', tab.loc['top (sell)'])]\n"
            "    disp = {k: (v['mean_pct'], v['welch_t'], v['placebo_p'], int(v['n'])) for k, v in rows}\n"
            "    neutral_mean = tab.loc['neutral', 'mean_pct']\n"
            "else:\n"
            "    b = R['band'][90]\n"
            "    disp = {'bottom (buy)': (b['bottom'][0], b['bottom'][3], b['bottom'][4], b['bottom'][5]),\n"
            "            'top (sell)':   (b['top'][0], b['top'][3], b['top'][4], b['top'][5])}\n"
            "    neutral_mean = b['neutral'][0]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "labels = ['bottom (buy)\\nPuell<0.5', 'neutral day', 'top (sell)\\nPuell>4']\n"
            "vals = [disp['bottom (buy)'][0], neutral_mean, disp['top (sell)'][0]]\n"
            "ax.bar(labels, vals, color=[GREEN, GREY, RED], width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.0f}%', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean 90-day forward BTC return')\n"
            "ax.set_title('Buy band < neutral; sell band = one event')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k, (m, t, p, n) in disp.items():\n"
            "    print(f'{k:<14}: mean {m:+.1f}%  Welch t {t:+.2f}  placebo p {p:.3f}  n={n}')\n"
            "print(f'neutral day    : mean {neutral_mean:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the buy band earns **+{R['band'][90]['bottom'][0]:.0f}%** vs a "
            f"neutral day's **+{R['band'][90]['neutral'][0]:.0f}%** (placebo *p* = "
            f"{R['band'][90]['bottom'][4]:.2f} — a random day does at least as well essentially "
            "always): the capitulation-buy half is worse than nothing. The sell band's placebo "
            f"*p* rounds to {R['band'][90]['top'][4]:.3f} and its Welch *t* is "
            f"**{R['band'][90]['top'][3]:.1f}** — but that spans **{R['top_days']}** days that are "
            "**one** 2017-18 event. Drawing 22 *scattered* random days from a mostly-rising tape "
            "will of course rarely match a single concentrated crash; the placebo rejecting here "
            "is the clustering, not a repeatable edge. n_effective ≈ 1."
        ),
        md(
            "### 4d · Timer vs buy-and-hold — the decisive test for H₂\n\n"
            "Sell to cash when Puell ≥ 4 (one-day lag), else long; 10 bps one-way × NAV per flip; "
            "compared to continuous buy-and-hold, and swept across the sell threshold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sweep = {hi: st.backtest_timing(PM, BTC, high=hi, cost_bps=10.0) for hi in [3.0,3.5,4.0,4.5]}\n"
            "    tots = [sweep[hi]['strat_total_pct'] for hi in [3.0,3.5,4.0,4.5]]\n"
            "    bh = sweep[4.0]['bh_total_pct']\n"
            "else:\n"
            "    tots = [R['sweep'][hi][0] for hi in [3.0,3.5,4.0,4.5]]; bh = R['bh_total']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "xl = ['sell>3.0','sell>3.5','sell>4.0','sell>4.5']\n"
            "ax.bar(xl, tots, color=AMBER, width=.55, label='Puell timer')\n"
            "ax.axhline(bh, color=GREEN, lw=2, ls='--', label=f'buy & hold (+{bh:,.0f}%)')\n"
            "for i, v in enumerate(tots): ax.annotate(f'{v:,.0f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('total return'); ax.set_title('The timer trails buy-and-hold at every threshold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"timer (sell>4): +{R['strat_total']:,.0f}%  vs  buy&hold +{R['bh_total']:,.0f}%\")\n"
            "print(f\"excess {R['excess_cagr']:+.2f}%/yr, HAC t = {R['excess_t']:+.2f}; \"\n"
            "      f\"invested {R['exposure']:.1f}% of the time ({R['n_flips']} flips)\")"
        ),
        md(
            f"> 💡 In plain words: at **every** threshold the timer ends **below** the "
            f"buy-and-hold line. Puell only clears 4 at the peak itself, so the rule is "
            f"**{R['exposure']:.0f}%** invested and its brief cash windows re-enter into the "
            f"still-ongoing bear — the excess return is **{R['excess_cagr']:.2f}%/yr** with HAC "
            f"*t* = **{R['excess_t']:.2f}**. Not a timing edge; a small timing cost."
        ),
        md(
            "### 4e · Synthetic control — the engine is honest\n\n"
            "Point the *same* predictive regression at a synthetic (Puell, price) world with a "
            "tunable contrarian link. It must find a planted effect and read ~zero on the null "
            "(checked over 20 seeds)."
        ),
        code(
            "null_t = []\n"
            "for s in range(20):\n"
            "    pu, pr = data.synthetic_world(beta=0.0, seed=763 + s)\n"
            "    null_t.append(st.synthetic_detect(pu, pr, horizon=30)['t_puell'])\n"
            "null_t = np.asarray(null_t)\n"
            "pu, pr = data.synthetic_world(beta=0.5, seed=763)\n"
            "planted_t = st.synthetic_detect(pu, pr, horizon=30)['t_puell']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_t, color=GREY, s=40, label='null (beta=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted contrarian beta=0.5')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (slope on log Puell)')\n"
            "ax.set_title('Control: detector finds a real contrarian link, reads ~0 on the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'planted beta=0.5: HAC t = {planted_t:+.2f}')\n"
            "print(f'null 20 seeds: mean t = {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds')"
        ),
        md(
            f"> 💡 In plain words: the detector nails a planted contrarian link "
            f"(t = {R['syn_planted_t']:.1f}) and averages ~zero on the null "
            f"({R['syn_null_mean']:+.2f}, |t|≥2 in {R['syn_null_fire']}/20). The machinery works; "
            "the real BTC tape simply has no such link for the Puell Multiple to exploit. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — forward-return-on-log(Puell) never clears |t| = 2 (best "
            f"**{R['reg'][30][1]:.2f}**, sign not stably contrarian, R² ≈ 0), the horse race "
            f"leaves Puell at **t = {R['horse_puell_t']:.2f}**, and the buy band "
            f"(**+{R['band'][90]['bottom'][0]:.0f}%**) underperforms a neutral day "
            f"(**+{R['band'][90]['neutral'][0]:.0f}%**). The only dramatic number — the sell "
            f"band's **{R['band'][90]['top'][0]:.0f}%** — is a single 2017-18 blow-off "
            f"(n_eff ≈ 1) with an autocorrelation-inflated Welch *t*.\n"
            f"- **Tradability `MIRAGE`** — the buy-low/sell-high timer is **{R['exposure']:.0f}%** "
            f"buy-and-hold and **trails** it net of costs at every threshold "
            f"(**+{R['strat_total']:,.0f}%** vs **+{R['bh_total']:,.0f}%**, "
            f"{R['excess_cagr']:.2f}%/yr, HAC t = {R['excess_t']:.2f}). Any CAGR is long exposure "
            "to a single survivor.\n"
            "- **\"Times tops & bottoms?\" `BUSTED`** — one top, zero tradable bottoms."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The reconstruction is the lesson.** Because the blocks/day constant cancels and "
            "the reward is a known step, within a halving epoch the Puell Multiple *is* "
            "price / trailing-365d-mean(price). Most famous \"on-chain\" valuation oscillators "
            "collapse to a price transform once you rebuild them; the burden of proof is on the "
            "extra ingredient (here, the halving imprint) to add information beyond price — and it "
            "doesn't.\n"
            "- **A fairer test of the sell band** would resample *contiguous* blocks rather than "
            "scattered days in the placebo, and would need several independent top episodes to "
            "have any power — we effectively have one.\n"
            "- **Dedup map:** [293-mvrv-ratio](../../293-mvrv-ratio/) (MVRV valuation band, "
            "NONE/MIRAGE — same family, different metric and cadence), "
            "[221-mayer-multiple](../../221-mayer-multiple/) (price / 200-day SMA — the Puell "
            "Multiple's within-epoch cousin), [663-hash-ribbons](../../663-hash-ribbons/) "
            "(hashrate crossover buy event, WEAK/FRAGILE), "
            "[323-btc-halving](../../323-btc-halving/) (the halving calendar — the same schedule, "
            "tested as a date not a revenue oscillator), [210-crypto-trend](../../210-crypto-trend/) "
            "(200-day price trend-following).\n\n"
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
