"""Generate the two narrative notebooks for Study 632 (Crypto Cross-Sectional Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached weekly
panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Binance 1w klines +
# yfinance backfills, 44 coins, 2017-01-02 -> 2026-06-22 open, fingerprint df8a5231ba1d).
R = dict(
    start="2017-01-02", end_open="2026-06-22", asof="2026-06-28", years=9.5,
    weeks=495, coins=44, n_delisted=11, avg_n=34.9, live_from="2017-11-20",
    fingerprint="df8a5231ba1d",
    # headline: (k, weeks, wml bps/wk, ann %, HAC t, Sharpe, hit %, win-mkt bps, win-mkt t)
    headline=[(1, 449, 164.1, 133.8, 3.56, 1.12, 55.0, 113.2, 3.30),
              (2, 448, 135.4, 101.8, 1.88, 0.66, 53.8, 114.0, 2.22),
              (3, 447, 197.8, 177.8, 2.95, 0.99, 55.5, 136.0, 2.63),
              (4, 446, 121.6, 87.9, 2.61, 0.83, 54.3, 76.8, 2.51)],
    mkt_bps=148.0, mkt_t=1.90, win_bps=261.2, win_t=2.60,
    # sub-periods (k=1): (label, weeks, bps/wk, ann %, HAC t, hit %)
    subs=[("2017-2019", 111, 194.6, 173.4, 1.62, 55.9),
          ("2020-2021", 104, 371.6, 571.0, 3.80, 63.5),
          ("2022", 52, 10.5, 5.6, 0.13, 48.1),
          ("2023-2026", 182, 70.8, 44.5, 1.30, 51.6)],
    # regime split on lagged BTC>30w SMA: (weeks, bps/wk, HAC t, Sharpe)
    bull=(243, 255.8, 3.51, 1.47), bear=(206, 56.0, 1.13, 0.55), welch_diff=2.09,
    to_win=1.48, to_lose=1.60, borrow_ann=10.0,
    # costs: (one-way bps, WML net bps/wk, net ann %, net HAC t, long-only exc bps, exc t)
    costs=[(10, 114.1, 80.7, 2.47, 98.4, 2.86),
           (25, 67.8, 42.3, 1.46, 76.1, 2.21),
           (50, -9.3, -4.7, -0.20, 39.0, 1.13)],
    placebo=dict(seeds=20, mean_bps=7.2, std_bps=36.4, mean_t=0.20),
    # synthetic: (rho, WML bps/wk, HAC t)
    syn=[(0.0, -39.2, -1.09), (0.10, 396.7, 11.14)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Survives the 2022 bear?: Busted](https://img.shields.io/badge/Survives_the_2022_bear%3F-Busted-8b949e?style=flat-square)\n\n"
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

from crypto_xs_momentum import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
    RES = st.run_xs_momentum(PANEL, k=1)          # headline: 1-week formation
else:
    PANEL = RES = None
print("real panel cached:", HAVE_REAL,
      "| shape:", (None if PANEL is None else PANEL.shape),
      "| holding weeks:", (0 if RES is None else len(RES)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do last week's winning coins keep winning? 🪙\n"
            "### Crypto momentum — the one factor academia says crypto really has, tested in plain sight\n\n"
            + BADGES +
            "Every crypto Twitter thread has a version of it: *ride the hot coin*. Academia, for once, "
            "agrees — a famous *Journal of Finance* study (Liu, Tsyvinski & Wu, 2022) sifted the whole "
            "crypto factor zoo and found that almost nothing predicts coin returns... **except momentum**: "
            "the coins that beat the pack last week tend to beat it again next week.\n\n"
            "So we rebuilt it, the honest way: ~44 top coins from 2017 to 2026, **including the corpses** — "
            "LUNA with its −100% death week, FTX's token with its −93% week, Monero after Binance kicked it "
            "off — rank them every Sunday night on last week's return, buy the top fifth, short the bottom "
            "fifth, hold one week, repeat.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, turnover math and placebo tests? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Even with 11 delisted/dead pairs in the panel, the 33 *live* "
            "names are coins that are still big in 2026 — a **survivorship** tilt we name rather than hide. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do last week's winners keep winning? | **Yes — really.** Winners beat losers by about "
            f"**+1.6% per week** over 9 years (HAC *t* = {R['headline'][0][4]}) — a genuine signal, "
            "not luck. |\n"
            f"| Can you trade it? | **Barely.** The portfolio replaces ~150% of itself *every week*. At "
            "big-pair fees it survives; at alt-coin slippage it dies. |\n"
            f"| Did it survive the 2022 bear? | **No.** Through LUNA and FTX it made "
            f"**+{R['subs'][2][2]:.0f} bps/week ≈ nothing** (*t* = {R['subs'][2][4]}). It's a "
            "bull-market factor. |\n\n"
            "The rest of this notebook shows you exactly where those three answers come from."
        ),

        md(
            "## 1 · The race: winners, losers, and the market\n\n"
            "Each Sunday close we rank every eligible coin on its **past 1-week return**, then hold the "
            "top fifth (*winners*) and bottom fifth (*losers*) for the **next** week — decisions are made "
            "strictly before the money is at risk. Here is the cumulative ride (log scale — crypto "
            "numbers are absurd)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots()\n"
            "    for col, c, lab in [('win', GREEN, 'winners (top quintile)'),\n"
            "                        ('mkt', GREY, 'equal-weight market'),\n"
            "                        ('lose', RED, 'losers (bottom quintile)')]:\n"
            "        ax.plot(RES.index, (1 + RES[col]).cumprod(), color=c, lw=1.8, label=lab)\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_title('Buy last week\\'s winners vs the market vs the losers — gross, 1-week hold')\n"
            "    ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.legend()\n"
            "    plt.show()\n"
            "    print(f\"winners leg {R['win_bps']:+.0f} bps/wk | market {R['mkt_bps']:+.0f} | \"\n"
            "          f\"spread {R['headline'][0][2]:+.0f} bps/wk (HAC t={R['headline'][0][4]})\")\n"
            "else:\n"
            "    print('cache missing — the frozen numbers in R tell the story:', R['headline'][0])"
        ),
        md(
            "The green line is the whole claim: last week's winners **keep pulling away** from the pack, "
            "and the losers keep sinking. The gap — about **+1.6% per week** before costs — is the "
            "momentum premium.\n\n"
            "> 🔬 **For the quants.** Newey-West *t* = 3.56 on the weekly spread, 449 weeks, one-week "
            "execution lag, quintiles of ~7 coins from an average cross-section of ~35."
        ),

        md(
            "## 2 · When did it pay? Year by year\n\n"
            "A factor that only works in one regime is a weather vane, not a law. Let's see the same "
            "spread carved into calendar years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yearly = RES['wml'].groupby(RES.index.year).apply(lambda x: (1 + x).prod() - 1) * 100\n"
            "    colors = [GREEN if v > 0 else RED for v in yearly]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.bar(yearly.index.astype(str), yearly, color=colors)\n"
            "    ax.axhline(0, color='k', lw=.8)\n"
            "    ax.set_title('Winners-minus-losers, compounded per calendar year (gross)')\n"
            "    ax.set_ylabel('% per year')\n"
            "    plt.show()\n"
            "    print(yearly.round(1).to_string())\n"
            "else:\n"
            "    print('sub-period stats (frozen):', R['subs'])"
        ),
        md(
            f"The pattern is loud: momentum feasted in the **2020–2021 bull** "
            f"(+{R['subs'][1][2]:.0f} bps/week, *t* = {R['subs'][1][3+1]}), then went **silent in 2022** — "
            f"the year of LUNA and FTX — earning +{R['subs'][2][2]:.0f} bps/week, statistically zero. "
            f"Since 2023 it's positive but soft (+{R['subs'][3][2]:.0f} bps/week, *t* = {R['subs'][3][4]}: "
            "below the certification bar on its own).\n\n"
            "**And no — it did not survive the 2022 bear.** That's the grey badge at the top. The one "
            "factor crypto \"really has\" clocks out exactly when you'd want a factor to show up."
        ),

        md(
            "## 3 · The catch: this thing *churns*\n\n"
            "\"Buy last week's winners\" sounds lazy. It isn't — last week's winners are rarely *this* "
            "week's winners, so the portfolio replaces roughly **150% of itself every week**, per leg. "
            "Every one of those trades pays a fee and crosses a spread."
        ),
        code(
            "cost_labels = [f'{c[0]} bps' for c in R['costs']]\n"
            "net_ann = [c[2] for c in R['costs']]\n"
            "colors = [GREEN if v > 20 else (AMBER if v > 0 else RED) for v in net_ann]\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(cost_labels, net_ann, color=colors)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_title('Long-short spread, NET of costs + short borrow — annualised')\n"
            "ax.set_xlabel('one-way trading cost (fee + slippage)')\n"
            "ax.set_ylabel('net % per year')\n"
            "for x, v in zip(cost_labels, net_ann):\n"
            "    ax.text(x, v + 2, f'{v:+.0f}%', ha='center')\n"
            "plt.show()\n"
            "print('turnover ~%.1fx NAV/week per leg; shorts pay %.0f%%/yr borrow'\n"
            "      % (R['to_win'], R['borrow_ann']))"
        ),
        md(
            "At **10 bps** one-way (BTC/ETH-grade taker fees) the edge survives handsomely. At **25 bps** "
            "(realistic all-in cost on liquid alts) it's amber — positive but no longer provable. At "
            "**50 bps** (thin alts, 2018-era spreads) it's gone. A gentler vehicle — just **tilting a "
            "long-only basket toward recent winners**, no shorting — holds up better: it beats the market "
            f"by ~+{R['costs'][1][4]:.0f} bps/week even at 25 bps (*t* = {R['costs'][1][5]}).\n\n"
            "## The verdict\n\n"
            "**Real. Fragile. Bear-market myth busted.** Crypto cross-sectional momentum is one of the rare "
            "folk beliefs the tape actually certifies — +1.6%/week gross, *t* = 3.56, with the corpses in "
            "the panel. But it's a **bull-market, big-pair, high-churn** edge: costs above ~25 bps eat the "
            "long-short, the post-2022 sample can't certify itself alone, and in the one year you truly "
            "needed it, it paid nothing.\n\n"
            "*Numbers: [docs/results.md](../docs/results.md) · rigor: "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb) · not investment advice.*"
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
            "# Crypto Cross-Sectional Momentum — the quant teardown 🪙\n\n"
            + BADGES +
            "**Claim (Liu-Tsyvinski-Wu, JF 2022; Liu-Tsyvinski, RFS 2021).** Sort coins on past 1–4-week "
            "returns; winners minus losers earns a significant premium — momentum is the one robust "
            "return factor in crypto.\n\n"
            "**Design.** 44-coin weekly panel (Binance spot 1w klines, Monday-open UTC weeks, spliced "
            "with yfinance `-USD` pre-listing backfills), 2017-01 → 2026-06, **11 delisted/dead pairs "
            "included** (LUNA truncated at its halt so the −100% week stays in and Terra 2.0 stays out). "
            "Rank at week-*t* close on the past *k*-week return; winners = top quintile, losers = bottom "
            "(`max(2, n//5)` names, equal-weight); earn the *t+1* close-to-close return — **exactly one "
            "execution lag**. Eligibility requires a valid formation return, a close at *t*, and a "
            "tradable *t+1* return (a halted coin's final print is not buyable; its crash weeks are). "
            "Inference: Newey-West *t* (Bartlett, 4 lags). Costs: one-way bps × measured traded NAV, "
            "short leg pays 10%/yr borrow. Returns are price-only in USD (spot coins pay no "
            "distributions); all legs gross unless labeled net.\n\n"
            "> ⚠️ **Survivorship, named.** The delisted pairs soften the bias; the 33 live names are "
            "still-top-cap-in-2026 survivors and pre-2019 casualties without a Binance USDT pair are "
            "absent. This flatters *levels* more than the long-short *spread*, but it is a real tilt.\n\n"
            f"As-of **{R['asof']}** (last complete UTC week) · fingerprint `{R['fingerprint']}` · "
            "single source of truth: [docs/results.md](../docs/results.md)."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · Headline — the *k*-week sorts\n\n"
            "The claim as stated is the **1-week** sort; 2–4-week formations are the robustness fan. "
            "Everything below recomputes live from the cached panel and must match `R` "
            "(= docs/results.md) exactly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print(f\"{'k':>2} {'weeks':>6} {'WML bps/wk':>11} {'ann%':>8} {'HAC t':>6} \"\n"
            "          f\"{'SR':>5} {'hit%':>5}  {'WIN-MKT (t)':>14}\")\n"
            "    ts = {}\n"
            "    for k in (1, 2, 3, 4):\n"
            "        res = st.run_xs_momentum(PANEL, k=k)\n"
            "        s = st.summarize(res['wml']); ex = st.summarize(res['win'] - res['mkt'])\n"
            "        ts[k] = s['hac_t']\n"
            "        print(f\"{k:>2} {s['n']:>6} {s['mean_bps']:>+11.1f} {s['ann_pct']:>+8.1f} \"\n"
            "              f\"{s['hac_t']:>+6.2f} {s['sharpe']:>5.2f} {s['hit_pct']:>5.1f}  \"\n"
            "              f\"{ex['mean_bps']:>+8.1f} ({ex['hac_t']:+.2f})\")\n"
            "    fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "    ks = list(ts); vals = [ts[k] for k in ks]\n"
            "    ax.bar([str(k) for k in ks], vals,\n"
            "           color=[GREEN if v >= 2 else AMBER for v in vals])\n"
            "    ax.axhline(2, color=RED, ls='--', lw=1, label='t = 2 bar')\n"
            "    ax.set_title('HAC t of the weekly WML spread by formation horizon')\n"
            "    ax.set_xlabel('formation k (weeks)'); ax.set_ylabel('Newey-West t (4 lags)')\n"
            "    ax.legend(); plt.show()\n"
            "else:\n"
            "    print('cache missing — frozen:', R['headline'])"
        ),
        md(
            f"> 💡 **In plain words.** Buying last week's top coins and shorting the bottom ones paid "
            f"about **+1.6% a week before costs** for nine years — and the odds that a random pattern "
            f"produces a *t* of {R['headline'][0][4]} are far below the desk's tolerance. The 2-week sort "
            f"dips under the bar (*t* = {R['headline'][1][4]}); 3- and 4-week sorts clear it — the effect "
            "is a horizon *band*, not a single lucky knob.\n\n"
            f"Context: the equal-weight market itself earns +{R['mkt_bps']:.0f} bps/wk at only "
            f"*t* = {R['mkt_t']} — crypto **beta** alone can't certify itself; the momentum **spread** can."
        ),

        md(
            "## 2 · Sub-periods & the bear-regime split (the third axis)\n\n"
            "Two cuts, both pre-registered by the claim itself: calendar sub-periods, and a **tradable** "
            "regime flag — BTC above/below its 30-week SMA at the *formation* close, applied to the "
            "*next* week (same lag as the strategy, no look-ahead)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for a, b, lab in [('2017-01-01', '2019-12-31', '2017-2019'),\n"
            "                      ('2020-01-01', '2021-12-31', '2020-2021'),\n"
            "                      ('2022-01-01', '2022-12-31', '2022     '),\n"
            "                      ('2023-01-01', '2026-06-28', '2023-2026')]:\n"
            "        s = st.sub_period(RES, a, b)\n"
            "        print(f\"{lab}: n={s['n']:>3}  {s['mean_bps']:>+7.1f} bps/wk  \"\n"
            "              f\"HAC t={s['hac_t']:>+5.2f}  hit={s['hit_pct']:.1f}%\")\n"
            "    rs = st.regime_split(RES, st.btc_regime(PANEL))\n"
            "    print()\n"
            "    for lab in ('bull', 'bear'):\n"
            "        s = rs[lab]\n"
            "        print(f\"{lab}: n={s['n']:>3}  WML {s['mean_bps']:>+7.1f} bps/wk  \"\n"
            "              f\"HAC t={s['hac_t']:>+5.2f}  SR={s['sharpe']:.2f}\")\n"
            "    print(f\"Welch t (bull - bear) = {rs['welch_t_diff']:+.2f}\")\n"
            "    cum = (1 + RES['wml']).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(cum.index, cum, color=GREY, lw=1.6)\n"
            "    m22 = (cum.index >= '2022-01-01') & (cum.index <= '2022-12-31')\n"
            "    ax.plot(cum.index[m22], cum[m22], color=RED, lw=2.2, label='2022 (LUNA, FTX)')\n"
            "    ax.set_yscale('log'); ax.legend()\n"
            "    ax.set_title('Cumulative WML (gross, log) — the factor flatlines through 2022')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('frozen:', R['subs'], R['bull'], R['bear'], R['welch_diff'])"
        ),
        md(
            f"> 💡 **In plain words.** All the certification lives in the bull years: 2020–2021 alone runs "
            f"at *t* = {R['subs'][1][4]}, while **2022 — the year of LUNA and FTX — paid "
            f"+{R['subs'][2][2]:.1f} bps/week at *t* = {R['subs'][2][4]}**: nothing. In bear-regime weeks "
            f"overall the spread is +{R['bear'][1]:.0f} bps/wk at *t* = {R['bear'][2]} (can't certify), in "
            f"bull weeks +{R['bull'][1]:.0f} at *t* = {R['bull'][2]}, and the difference itself is "
            f"significant (Welch *t* = {R['welch_diff']}). **\"Does it survive the 2022 bear?\" — Busted.** "
            "Notably it doesn't *lose* in bears — momentum crashes of the equity kind don't show here — "
            "it just stops paying."
        ),

        md(
            "## 3 · Costs — turnover × fee grid, shorts pay borrow\n\n"
            "Weekly quintiles churn brutally; we *measure* traded NAV (buys + sells, each paying the "
            "one-way cost) instead of assuming it, and charge the short leg 10%/yr borrow (perp funding, "
            "historically positive, would usually *pay* a short — the charge is conservative)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print(f\"measured turnover: winners {RES['to_win'].mean():.2f}x NAV/wk, \"\n"
            "          f\"losers {RES['to_lose'].mean():.2f}x NAV/wk\")\n"
            "    for c in (10.0, 25.0, 50.0):\n"
            "        nw = st.summarize(st.net_wml(RES, c))\n"
            "        ex = st.summarize(st.net_long_only(RES, c) - RES['mkt'])\n"
            "        print(f\"c={c:>4.0f} bps: WML net {nw['mean_bps']:>+7.1f} bps/wk \"\n"
            "              f\"(ann {nw['ann_pct']:>+6.1f}%)  t={nw['hac_t']:>+5.2f} | \"\n"
            "              f\"long-only exc {ex['mean_bps']:>+6.1f} bps/wk  t={ex['hac_t']:>+5.2f}\")\n"
            "    cs = np.linspace(0, 60, 25)\n"
            "    ann_ls = [st.summarize(st.net_wml(RES, c))['ann_pct'] for c in cs]\n"
            "    ann_lo = [st.summarize(st.net_long_only(RES, c) - RES['mkt'])['mean_bps'] * 52.18 / 100\n"
            "              for c in cs]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(cs, ann_ls, color=RED, lw=2, label='long-short WML (net, ann %)')\n"
            "    ax.plot(cs, ann_lo, color=GREEN, lw=2, label='long-only winners minus market (net, ~ann %)')\n"
            "    ax.axhline(0, color='k', lw=.8)\n"
            "    ax.set_title('Net edge vs one-way cost — the long-short dies near ~45 bps')\n"
            "    ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net annualised edge (%)')\n"
            "    ax.legend(); plt.show()\n"
            "else:\n"
            "    print('frozen:', R['costs'])"
        ),
        md(
            f"> 💡 **In plain words.** At big-pair taker fees (10 bps) the long-short still certifies "
            f"(*t* = {R['costs'][0][3]}); at liquid-alt all-in costs (25 bps) it's positive but unprovable "
            f"(*t* = {R['costs'][1][3]}); at thin-alt costs it's dead. The **long-only winners tilt** — no "
            f"shorting, no borrow, half the churn — beats the market at *t* = {R['costs'][1][5]} even at "
            "25 bps: the deployable residue of the factor. Add that shorting the losers leg at size is "
            "partly fictional (borrow vanishes exactly when a coin is collapsing) and Tradability reads "
            "**Fragile**, not Investable."
        ),

        md(
            "## 4 · Placebo — 20-seed random ranks\n\n"
            "Same eligibility, same quintile sizes, same lag — but ranks drawn from noise. Averaged over "
            "20 seeds (single-seed baselines are banned at this desk)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_random_ranks(PANEL, k=1, n_seeds=20)\n"
            "    print(f\"shuffled WML mean = {pl['mean_bps']:+.1f} +/- {pl['std_bps']:.1f} bps/wk \"\n"
            "          f\"({pl['n_seeds']} seeds) | mean HAC t = {pl['mean_hac_t']:+.2f}\")\n"
            "    print(f\"observed WML mean = {R['headline'][0][2]:+.1f} bps/wk (HAC t = \"\n"
            "          f\"{R['headline'][0][4]:+.2f}) — far outside the shuffled cloud\")\n"
            "else:\n"
            "    print('frozen:', R['placebo'])"
        ),

        md(
            "## 5 · Synthetic control — machinery proof *(never market evidence)*\n\n"
            "A deterministic crypto-vol panel with a plantable idiosyncratic AR(1). At ρ = 0 the quintile "
            "detector must stay quiet; at ρ = 0.10 it must light up."
        ),
        code(
            "for rho in (0.0, 0.10):\n"
            "    ps = data.synthetic_panel(rho=rho, seed=632)\n"
            "    r = st.run_xs_momentum(ps, k=1)\n"
            "    s = st.summarize(r['wml'])\n"
            "    lab = 'null rho=0.00' if rho == 0 else 'planted rho=0.10'\n"
            "    print(f\"{lab}: WML {s['mean_bps']:>+7.1f} bps/wk  HAC t={s['hac_t']:>+6.2f}\")"
        ),
        md(
            "> 💡 **In plain words.** The engine can't be fooled into finding momentum where none was "
            "planted, and can't miss it where it was. So the real-tape *t* = 3.56 is the market talking, "
            "not the code.\n\n"
            "## Verdict\n\n"
            f"- **Signal — REAL.** +{R['headline'][0][2]:.0f} bps/wk WML, HAC *t* = {R['headline'][0][4]}, "
            f"Sharpe {R['headline'][0][5]}, robust at k = 3, 4; placebo ~0. Survivorship named (softened "
            "by 11 delisted pairs incl. LUNA's −100% week, not removed).\n"
            f"- **Tradability — FRAGILE.** ~1.5× NAV/leg/week churn; net *t* = {R['costs'][0][3]} at "
            f"10 bps but {R['costs'][1][3]} at 25 bps and dead at 50; bull-loaded and decayed post-2022. "
            f"Long-only tilt survives 25 bps (*t* = {R['costs'][1][5]}).\n"
            f"- **Survives the 2022 bear? — BUSTED.** +{R['subs'][2][2]:.1f} bps/wk, *t* = "
            f"{R['subs'][2][4]} through LUNA/FTX; bear-regime *t* = {R['bear'][2]}; bull−bear Welch *t* = "
            f"{R['welch_diff']}.\n\n"
            "*Single source of truth: [docs/results.md](../docs/results.md) · sources: "
            "[docs/references.md](../docs/references.md) · siblings: "
            "[251-crypto-reversal](../../251-crypto-reversal/README.md), "
            "[222-altseason-rotation](../../222-altseason-rotation/README.md), "
            "[210-crypto-trend](../../210-crypto-trend/README.md) · not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        path = os.path.join(HERE, name)
        nbf.write(builder(), path)
        print("wrote", path)
