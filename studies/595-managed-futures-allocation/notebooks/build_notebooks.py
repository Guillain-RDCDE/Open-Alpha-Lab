"""Generate the two narrative notebooks for Study 595 (Managed-Futures Sleeve).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the shared futures cache
at the repo root plus the study's cached ETF tape under ../_cache/ and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md). Heavy statistics (the 5,000-draw
bootstraps) are re-run LIGHT in-notebook (fewer draws) with the canonical numbers quoted from
``R``. The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (shared 18-futures panel +
# yfinance SPY/VBMFX/DBMF/KMLM/^IRX; joint window 2001-09-30 -> 2026-05-31, 297 months).
R = dict(
    asof="2026-07-03", joint_start="2001-09-30", joint_end="2026-05-31", n_months=297,
    fingerprint="2f0e9858dde3", fut_fp="b1cdf7bc6129",
    book=dict(sharpe=0.407, nw_t=2.05, vol=18.4, maxdd=-29.6, cagr=7.73, lev=3.71, turn=1.12),
    # correlations: (r, lo, hi, n)
    corr_6040=(-0.054, -0.167, 0.060, 297), corr_stocks=(-0.053, -0.166, 0.061, 297),
    corr_bonds=(-0.021, -0.135, 0.093, 297),
    corr_dbmf=(-0.212, -0.408, 0.002, 84), corr_kmlm=(-0.432, -0.611, -0.210, 65),
    corr_dbmf_book=(0.682, 0.547, 0.783, 84),
    p6040=dict(cagr=7.52, vol=9.37, sharpe=0.643, maxdd=-32.4, mar=0.232),
    blend=dict(cagr=7.72, vol=8.29, sharpe=0.738, maxdd=-23.4, mar=0.330),
    alpha=dict(ann=8.13, t=2.10, beta=-0.107, r2=0.3),
    dsharpe=dict(obs=0.095, lo=-0.064, hi=0.242, p=0.119, draws=5000, block=6),
    # sleeve sizes: (pct, sharpe, maxdd%, dS, lo, hi, p)
    sleeves=[(10, 0.713, -26.5, 0.070, -0.031, 0.162, 0.080),
             (15, 0.738, -23.4, 0.095, -0.064, 0.242, 0.119),
             (20, 0.753, -20.2, 0.110, -0.111, 0.317, 0.164)],
    # (one-way bps, alpha %/yr, alpha NW t)
    costs=[(2.0, 8.52, 2.20), (5.0, 8.13, 2.10), (10.0, 7.46, 1.92)],
    # (lookback, book Sharpe, alpha t, dSharpe)
    lookbacks=[(9, 0.209, 1.35, 0.045), (12, 0.407, 2.10, 0.095), (15, 0.352, 1.67, 0.060)],
    nw_lags=[(3, 2.06), (6, 2.10), (12, 2.25)],
    # (label, 60/40 S, blend S, dS, sleeve standalone S)
    subperiods=[("2001-09 - 2008-12", -0.08, 0.17, 0.257, 0.69),
                ("2009-01 - 2019-12", 1.20, 1.09, -0.107, 0.10),
                ("2020-01 - 2026-05", 0.63, 0.78, 0.147, 0.55)],
    y2022=dict(spy=-18.18, bonds=-13.24, p6040=-15.91, sleeve=33.13, blend=-9.02,
               dbmf=21.60, kmlm=24.24, contrib=5.0),
    bonds_axis=dict(corr=-0.021, beta=-0.093, r2=0.0,
                    p4555_sharpe=0.664, p4555_dd=-24.1, p4555_2022=-15.15),
    live=dict(start="2019-07-31", end="2026-06-30", n=84, dbmf_cagr=8.85, dbmf_vol=11.4,
              dbmf_sharpe=0.562, dbmf_dd=-17.3, s6040=0.671, dd6040=-20.1,
              sblend=0.790, ddblend=-13.5, ds=0.119, lo=-0.048, hi=0.279, p=0.090),
    placebo=dict(seeds=24, book_sharpe=-0.230, book_sd=0.197, ds=-0.0772, ds_sd=0.0448),
    # (planted Sharpe, dS, lo, hi, p, alpha t)
    syn=[(0.0, -0.026, -0.106, 0.062, 0.697, 0.00), (0.8, 0.156, 0.077, 0.246, 0.000, 4.45)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Bonds_in_disguise%3F: Busted](https://img.shields.io/badge/Bonds_in_disguise%3F-Busted-8b949e?style=flat-square)\n\n"
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

from managed_futures_allocation import data, strategy as st

SLEEVE, FEE, COST_BPS = 0.15, 0.0085, 5.0
HAVE_REAL = data.have_real()
if HAVE_REAL:
    FUT = data.load_futures()
    FM = data.drop_partial_last_month(data.monthly_from_daily_returns(FUT), FUT.index.max())
    ETFS = data.load_etfs()
    TR = data.monthly_total_returns(ETFS[["SPY", "VBMFX", "DBMF", "KMLM"]])
    RF = data.monthly_rf(ETFS["^IRX"])
    BOOK = st.tsmom_book(FM, lookback=12, cost_bps=COST_BPS)
    P6040_FULL = st.portfolio_6040(TR["SPY"], TR["VBMFX"])
    IDX = P6040_FULL.index.intersection(BOOK["net"].index).intersection(RF.dropna().index)
    P6040, RF_I = P6040_FULL.loc[IDX], RF.loc[IDX]
    E6040 = P6040 - RF_I
    MF_NET = BOOK["net"].loc[IDX]           # futures excess, net of turnover costs
    MF_FUNDED = RF_I + MF_NET               # cash-collateralised sleeve (gross of ETF fee)
    BLEND = st.blend_returns(P6040, MF_NET, RF_I, SLEEVE, FEE)
else:
    FUT = FM = ETFS = TR = RF = BOOK = None
    P6040 = RF_I = E6040 = MF_NET = MF_FUNDED = BLEND = None
print("real caches present:", HAVE_REAL,
      "| joint months:", (0 if P6040 is None else len(P6040)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The fire-extinguisher fund: does a trend sleeve fix a 60/40? 🧯\n"
            "### Managed futures in a normal portfolio — the ad, the tape, and 2022, in plain English\n\n"
            + BADGES +
            "After 2022 — the year stocks **and** bonds fell together — a new pitch took over the "
            "allocation world: *put 10-20% of your portfolio into a managed-futures fund.* These funds "
            "(trend-followers) buy whatever has been rising and short whatever has been falling, across "
            "stock indexes, bonds, oil, gold, currencies. The ad says two things: they're "
            "**uncorrelated** with everything else you own, and they **pay off in crises** — in 2022 "
            "they were up +20-30% while everything else burned.\n\n"
            "We've already tested whether trend-following is a great strategy *on its own* — twice "
            "([31-trade-winds](../../31-trade-winds/), [518-time-series-momentum](../../518-time-series-momentum/)) "
            "— and both times the answer was **Weak**. This study asks the different, sharper question the "
            "industry actually sells: even if it's mediocre alone, does *adding a slice of it* make a "
            "normal 60/40 portfolio **better**?\n\n"
            "> 📓 **Plain-language layer.** Want the t-stats, bootstraps and robustness grids? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Our long history is a *replication* (a simple 12-month "
            "trend rule on 18 futures, with costs and the real ETF's 0.85%/yr fee charged) — the live "
            "ETFs (DBMF, KMLM) only exist since 2019, and we cross-check on them. Every chart is drawn "
            "by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is it really uncorrelated with a 60/40? | **Yes — genuinely.** Correlation ≈ "
            f"**{R['corr_6040'][0]:.2f}** over 25 years; the live funds are even *negative* "
            "(they zig when your portfolio zags). |\n"
            "| Did it deliver in 2022? | **Yes.** Our sleeve made **+33%** (the live funds +22 to "
            f"+24%) while the 60/40 lost **{R['y2022']['p6040']:.0f}%**. The blend lost only "
            f"**{R['y2022']['blend']:.0f}%**. |\n"
            "| So does the sleeve make the portfolio better? | **Probably — but the tape can't prove "
            f"it.** Sharpe {R['p6040']['sharpe']:.2f} → {R['blend']['sharpe']:.2f} and the worst "
            "drawdown shrinks by 9 points, but the improvement is within statistical noise, and it "
            "spent **2009-2019 quietly costing you**. |\n"
            "| Is it secretly just a bond fund? | **No — busted.** Zero correlation to bonds, and in "
            "2022 it was *short* bonds and profited while bonds lost 13%. |"
        ),

        md(
            "## Two and a half decades, side by side\n\n"
            "Take a plain 60/40 (60% S&P 500, 40% total-bond, rebalanced monthly). Now carve out 15% "
            "and give it to the trend sleeve (paying the real fund's 0.85%/yr fee). Same money, two "
            "histories:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    w60 = (1 + P6040).cumprod(); wbl = (1 + BLEND).cumprod()\n"
            "    wmf = (1 + MF_FUNDED - FEE/12).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(w60.index, w60, color=GREY, lw=2, label='60/40')\n"
            "    ax.plot(wbl.index, wbl, color=GREEN, lw=2, label='60/40 + 15% trend sleeve')\n"
            "    ax.plot(wmf.index, wmf, color=AMBER, lw=1.2, alpha=.8, label='the sleeve alone (net of fee)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('60/40 vs 60/40 + 15% managed-futures sleeve — 2001-2026')\n"
            "    ax.legend()\n"
            "    plt.show()\n"
            "    print(f\"60/40: {R['p6040']['cagr']}%/yr, worst fall {R['p6040']['maxdd']}%\")\n"
            "    print(f\"blend: {R['blend']['cagr']}%/yr, worst fall {R['blend']['maxdd']}%\")\n"
            "else:\n"
            "    print('caches missing — see docs/results.md for the frozen numbers')\n"
        ),
        md(
            "Similar destination, **smoother road**: the blend ends slightly *ahead* "
            f"({R['blend']['cagr']}% vs {R['p6040']['cagr']}%/yr) with a whole point and a half less "
            "volatility, and its worst peak-to-trough fall is **−23%** instead of **−32%**. That "
            "combination — same return, less pain — is what \"diversification\" is supposed to buy.\n\n"
            "> 🔬 **For the quants:** the Sharpe goes 0.643 → 0.738; whether that +0.095 is "
            "*statistically certifiable* is the whole fight in notebook 02 (spoiler: it is not)."
        ),

        md(
            "## \"Uncorrelated\" — the half of the ad that is simply true\n\n"
            "Each dot below is one month: how the 60/40 did (x-axis) versus how the sleeve did "
            "(y-axis). If the sleeve were a closet stock fund the cloud would slope up; a closet "
            "bond fund, same thing. It doesn't slope at all:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(7.5, 6))\n"
            "    ax.scatter(E6040*100, MF_NET*100, s=14, alpha=.5, color=GREY)\n"
            "    ax.axhline(0, color='k', lw=.6); ax.axvline(0, color='k', lw=.6)\n"
            "    b = np.polyfit(E6040, MF_NET, 1)\n"
            "    xs = np.linspace(E6040.min(), E6040.max(), 50)\n"
            "    ax.plot(xs*100, (b[0]*xs+b[1])*100, color=RED, lw=2,\n"
            "            label=f'fit: slope {b[0]:+.2f} (flat = uncorrelated)')\n"
            "    ax.set_xlabel('60/40 monthly excess return (%)')\n"
            "    ax.set_ylabel('trend sleeve monthly return (%)')\n"
            "    ax.set_title('297 months: the sleeve does not care what your 60/40 is doing')\n"
            "    ax.legend()\n"
            "    plt.show()\n"
            "    c = st.corr_ci(MF_NET, E6040)\n"
            "    print(f\"correlation = {c['r']:+.3f}  (95% CI [{c['lo']:+.3f}, {c['hi']:+.3f}])\")\n"
        ),
        md(
            f"Correlation **{R['corr_6040'][0]:.2f}**, statistically indistinguishable from zero — "
            "and the *live* funds since 2019 are actually **negative** (DBMF "
            f"{R['corr_dbmf'][0]:.2f}, KMLM {R['corr_kmlm'][0]:.2f}). Almost nothing you can buy "
            "with one ticker has that property. This half of the pitch is **real**."
        ),

        md(
            "## 2022 — the year the fire actually happened\n\n"
            "The whole point of insurance is the one year you need it. In 2022 the Fed hiked, and "
            "for the first time in 40 years stocks and bonds fell **together** — the 60/40's "
            "diversification failed exactly when it was needed. The trend sleeve was short bonds, "
            "short stocks, long the dollar and long energy... by *following the trend down*:"
        ),
        code(
            "y = R['y2022']\n"
            "names = ['S&P 500', 'Bonds', '60/40', '60/40 +15% sleeve', 'Trend sleeve\\n(replication)',\n"
            "         'DBMF (live)', 'KMLM (live)']\n"
            "vals = [y['spy'], y['bonds'], y['p6040'], y['blend'], y['sleeve'], y['dbmf'], y['kmlm']]\n"
            "cols = [RED, RED, RED, AMBER, GREEN, GREEN, GREEN]\n"
            "fig, ax = plt.subplots()\n"
            "bars = ax.bar(names, vals, color=cols)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "for b_, v in zip(bars, vals):\n"
            "    ax.text(b_.get_x()+b_.get_width()/2, v + (1 if v > 0 else -2.6),\n"
            "            f'{v:+.1f}%', ha='center', fontsize=10)\n"
            "ax.set_ylabel('calendar-2022 total return (%)')\n"
            "ax.set_title('2022: everything fell — except the trend funds')\n"
            "plt.show()\n"
            "print(f\"sleeve contribution to the blend in 2022: about +{y['contrib']:.0f} points\")\n"
        ),
        md(
            "The blend lost **−9%** instead of **−16%**. One year proves nothing statistically — "
            "but it is exactly the payoff the product was sold on, and it happened in the live "
            "funds too, net of fees. *(Note our replication's +33% is flattering vs the live funds' "
            "+22-24% — real-world slippage is real.)*"
        ),

        md(
            "## The fine print: the decade the insurance just cost money\n\n"
            "Here's the part the ads skip. Split the 25 years into three regimes:"
        ),
        code(
            "labels = [s[0] for s in R['subperiods']]\n"
            "s60 = [s[1] for s in R['subperiods']]; sbl = [s[2] for s in R['subperiods']]\n"
            "x = np.arange(3); w = 0.36\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(x-w/2, s60, w, color=GREY, label='60/40')\n"
            "ax.bar(x+w/2, sbl, w, color=GREEN, label='+15% sleeve')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('Sharpe ratio (excess)')\n"
            "ax.set_title('The sleeve helps in stormy decades — and drags in calm ones')\n"
            "ax.axhline(0, color='k', lw=.8); ax.legend()\n"
            "plt.show()\n"
            "for lab, a, b_, d, _ in R['subperiods']:\n"
            "    print(f'{lab}: 60/40 Sharpe {a:+.2f} -> blend {b_:+.2f}  (change {d:+.3f})')\n"
        ),
        md(
            "In 2001-2008 (dot-com bust, GFC) and 2020-2026 (Covid, the 2022 inflation shock) the "
            "sleeve earned its keep. But through the **entire 2009-2019 bull decade** it was a drag "
            "— eleven years of paying for an extinguisher while nothing caught fire. That's not a "
            "flaw exactly — it's what insurance *is* — but it's why the improvement never rises out "
            "of the statistical noise, and why most investors who bought trend funds in 2010 had "
            "sold them by 2019, just in time to miss 2022."
        ),

        md(
            "## The verdict, in plain words\n\n"
            f"- **The \"uncorrelated\" claim is true.** ≈ {R['corr_6040'][0]:.2f} to a 60/40 over 25 "
            "years; the live funds are negative. Rare and genuine.\n"
            "- **It is not a bond fund in a costume.** Zero correlation to bonds — and in 2022 it "
            "made +33% *because* it was short the bonds that were falling. Putting the same 15% into "
            "more bonds instead would have made 2022 **worse** (−15% vs −9%).\n"
            f"- **The improvement is real-looking but unprovable.** Sharpe {R['p6040']['sharpe']:.2f} "
            f"→ {R['blend']['sharpe']:.2f}, worst fall −32% → −23% — but 25 years of data is not "
            "enough to certify a +0.10 Sharpe from a strategy this streaky, and the quants' "
            "confidence interval firmly includes zero.\n"
            "- **You can actually buy it** (DBMF/KMLM, one ticket, ~0.85%/yr) — cheap access to a "
            "benefit the tape can't guarantee. **Signal: Mixed. Tradability: Fragile. Bonds-in-"
            "disguise: Busted.**\n\n"
            "> 📓 Full statistics, robustness grids and the honest fine print: "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** · frozen numbers: "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R['fingerprint'] + "`).\n\n"
            "*Research & education, not investment advice.*"
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
            "# Managed-Futures Sleeve — the quant teardown 🧯\n\n"
            + BADGES +
            "**Claim under test:** a 10-20% trend-following sleeve improves a 60/40 — near-zero "
            "correlation plus 2022-style crisis alpha.\n\n"
            "**Dedup guard:** [31-trade-winds](../../31-trade-winds/) and "
            "[518-time-series-momentum](../../518-time-series-momentum/) already graded the "
            "*standalone* TSMOM premium **Weak** on this desk. We cite that and do not re-litigate "
            "it; the unit under test here is the **portfolio improvement**, whose decisive "
            "statistics are (1) the sleeve's **Newey-West alpha on the 60/40** (the mean-variance "
            "criterion for a beneficial addition), (2) a **paired moving-block bootstrap of "
            "ΔSharpe**, (3) the **correlation with a Fisher-z CI**, and (4) the **2022 "
            "attribution**.\n\n"
            "**Construction:** 12-mo TSMOM (sign of trailing 12-month return) on the shared "
            "18-futures panel, 40%/σ inverse-vol per contract / N_active, monthly rebalance, "
            "**exactly one month of execution lag** (signal & vol through month-end t−1, held "
            "month t), **5 bps one-way × traded notional** (2/10 shown); futures shorts post "
            "margin, **no borrow fee** (unfunded margin instruments). The blended sleeve is "
            "cash-collateralised (rf + excess) and pays the live wrapper's **0.85%/yr fee**. All "
            "Sharpe races are **excess-vs-excess** (^IRX). Numbers quoted from "
            "[`docs/results.md`](../docs/results.md); heavy bootstraps re-run light here."
        ),
        code(BOOT_CELL),

        md("## 0 · Data stamp"),
        code(
            "if HAVE_REAL:\n"
            "    try:\n"
            "        from quantlab import repro\n"
            "        joint = pd.concat([P6040, MF_NET, RF_I], axis=1)\n"
            "        joint.columns = ['p6040', 'mf_net', 'rf']\n"
            "        print('fingerprint(joint) =', repro.fingerprint(joint),\n"
            "              ' (frozen:', R['fingerprint'] + ')')\n"
            "        print('fingerprint(futures monthly) =', repro.fingerprint(FM),\n"
            "              ' (frozen:', R['fut_fp'] + ')')\n"
            "    except Exception as e:\n"
            "        print('quantlab.repro unavailable:', e)\n"
            "    print(f'futures: {FUT.shape[0]} days x {FUT.shape[1]} contracts, '\n"
            "          f'{FUT.index.min().date()} -> {FUT.index.max().date()} (partial last month dropped)')\n"
            "    print(f'joint window: {P6040.index.min().date()} -> {P6040.index.max().date()} '\n"
            "          f'({len(P6040)} months)')\n"
            "    print(f\"book: avg gross leverage {BOOK['leverage']:.2f}x, \"\n"
            "          f\"avg turnover {BOOK['turnover'].mean():.2f}x NAV/mo\")\n"
        ),
        md(
            "> ⚠️ **Panel caveats (named on the Signal axis).** The 18 contracts are liquid futures "
            "*still trading in 2026* — mild survivorship (TSMOM shorts a faller rather than deleting "
            "it, so far weaker than an equity sort, but named). Continuous series are yfinance "
            "front-month splices (study 31's roll construction). The replication is gross of "
            "slippage beyond the modeled bps — its 2022 (+33%) vs DBMF's live +21.6% shows the "
            "flattering gap. The **live ETFs are the honest net tape** and get their own section."
        ),

        md(
            "## 1 · The sleeve itself (context only — the premium is the siblings' fight)\n\n"
            "> 💡 **In plain words:** before asking whether the sleeve helps the portfolio, look at "
            "what it is: a diversified trend book that earns a positive but modest return with big "
            "swings — mediocre alone, interesting only for *when* it earns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.ann_stats(MF_FUNDED, RF_I)\n"
            "    print(f'net excess Sharpe {st.sharpe_excess(MF_NET):.3f}   '\n"
            "          f'NW t(mean) {st.nw_tstat(MF_NET):+.2f}   vol {s[\"vol\"]*100:.1f}%   '\n"
            "          f'maxDD {s[\"maxdd\"]*100:.1f}%   CAGR(funded) {s[\"cagr\"]*100:.2f}%')\n"
            "    print('(standalone premium graded Weak by studies 31 & 518 — cited, not re-litigated)')\n"
        ),

        md(
            "## 2 · Axis 1a — \"near-zero correlation\": decisively real\n\n"
            "Fisher-z 95% CIs on monthly excess returns. The rolling window shows the property is "
            "not an average of +0.5 and −0.5 regimes — it hugs zero throughout."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for lab, x_ in [('60/40', E6040), ('stocks', TR['SPY'].loc[IDX]-RF_I),\n"
            "                    ('bonds ', TR['VBMFX'].loc[IDX]-RF_I)]:\n"
            "        c = st.corr_ci(MF_NET, x_)\n"
            "        print(f'corr(MF, {lab}) = {c[\"r\"]:+.3f}  95% CI [{c[\"lo\"]:+.3f}, {c[\"hi\"]:+.3f}]  n={c[\"n\"]}')\n"
            "    roll = MF_NET.rolling(36).corr(E6040)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(roll.index, roll, color=GREY, lw=1.6)\n"
            "    ax.axhline(0, color='k', lw=.8)\n"
            "    ax.axhspan(-0.2, 0.2, color=GREEN, alpha=.12, label='|corr| < 0.2')\n"
            "    ax.set_ylim(-1, 1); ax.set_ylabel('rolling 36-mo corr(MF, 60/40)')\n"
            "    ax.set_title('The correlation hugs zero in every regime')\n"
            "    ax.legend()\n"
            "    plt.show()\n"
        ),
        md(
            f"Frozen: corr(MF, 60/40) = **{R['corr_6040'][0]:+.3f}** "
            f"[{R['corr_6040'][1]:+.3f}, {R['corr_6040'][2]:+.3f}] (n={R['corr_6040'][3]}); live "
            f"DBMF **{R['corr_dbmf'][0]:+.3f}**, KMLM **{R['corr_kmlm'][0]:+.3f}** (both negative). "
            "This half of the claim clears any reasonable bar.\n\n"
            "> 💡 **In plain words:** \"uncorrelated\" survives every test we throw at it. The fight "
            "is over whether uncorrelated-plus-weak-return actually *improves* the portfolio."
        ),

        md(
            "## 3 · Axis 1b — the improvement: point estimates vs inference\n\n"
            "Two complementary tests. **(a)** The mean-variance criterion: regress sleeve excess on "
            "60/40 excess — a positive NW alpha means the sleeve expands the attainable frontier. "
            "**(b)** The direct object: ΔSharpe of the 15% blend, with a paired moving-block "
            "bootstrap (joint rows resampled in blocks of 6, so the cross-correlation survives; "
            "Ledoit-Wolf spirit)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = st.ann_stats(P6040, RF_I); b = st.ann_stats(BLEND, RF_I)\n"
            "    print('               CAGR    vol   Sharpe  maxDD    MAR')\n"
            "    print(f\"60/40        : {a['cagr']*100:5.2f}%  {a['vol']*100:5.2f}%  {a['sharpe']:.3f}  \"\n"
            "          f\"{a['maxdd']*100:5.1f}%  {a['mar']:.3f}\")\n"
            "    print(f\"+15% sleeve  : {b['cagr']*100:5.2f}%  {b['vol']*100:5.2f}%  {b['sharpe']:.3f}  \"\n"
            "          f\"{b['maxdd']*100:5.1f}%  {b['mar']:.3f}\")\n"
            "    al = st.nw_alpha(MF_NET, E6040, lags=6)\n"
            "    print(f\"\\nMF alpha vs 60/40: {al['alpha_ann']*100:+.2f}%/yr  NW t = {al['t_alpha']:+.2f}  \"\n"
            "          f\"beta {al['beta']:+.3f}  R^2 {al['r2']*100:.1f}%\")\n"
            "    # light bootstrap re-run (canonical = 5,000 draws in docs/results.md)\n"
            "    bs = st.bootstrap_dsharpe(E6040, MF_NET, SLEEVE, FEE, n_draws=1500, seed=595)\n"
            "    print(f\"dSharpe {bs['obs']:+.3f}  95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]  \"\n"
            "          f\"p(one-sided) {bs['p_onesided']:.3f}   (light 1,500-draw rerun; frozen: \"\n"
            "          f\"{R['dsharpe']['obs']:+.3f} [{R['dsharpe']['lo']:+.3f}, {R['dsharpe']['hi']:+.3f}], \"\n"
            "          f\"p {R['dsharpe']['p']:.3f})\")\n"
        ),
        code(
            "if HAVE_REAL:\n"
            "    # bootstrap distribution figure\n"
            "    df = pd.concat([E6040, MF_NET], axis=1).dropna()\n"
            "    av = df.iloc[:,0].to_numpy(); bv = df.iloc[:,1].to_numpy() - FEE/12\n"
            "    rng = np.random.default_rng(595); n = len(av); block = 6\n"
            "    nb_ = int(np.ceil(n/block)); draws = []\n"
            "    for _ in range(1500):\n"
            "        stt = rng.integers(0, n-block+1, size=nb_)\n"
            "        ii = (stt[:,None] + np.arange(block)[None,:]).ravel()[:n]\n"
            "        aa, bb = av[ii], (1-SLEEVE)*av[ii] + SLEEVE*bv[ii]\n"
            "        draws.append((bb.mean()/bb.std() - aa.mean()/aa.std())*np.sqrt(12))\n"
            "    draws = np.array(draws)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.hist(draws, bins=50, color=GREY, alpha=.75)\n"
            "    ax.axvline(0, color='k', lw=1)\n"
            "    ax.axvline(R['dsharpe']['obs'], color=GREEN, lw=2,\n"
            "               label=f\"observed dSharpe {R['dsharpe']['obs']:+.3f}\")\n"
            "    ax.axvline(np.percentile(draws, 2.5), color=RED, ls='--', lw=1.4, label='2.5 / 97.5 pct')\n"
            "    ax.axvline(np.percentile(draws, 97.5), color=RED, ls='--', lw=1.4)\n"
            "    ax.set_xlabel('bootstrap dSharpe (blend - 60/40)')\n"
            "    ax.set_title('The improvement is positive — and its CI firmly includes zero')\n"
            "    ax.legend()\n"
            "    plt.show()\n"
        ),
        md(
            f"Frozen headline: Sharpe **{R['p6040']['sharpe']:.3f} → {R['blend']['sharpe']:.3f}**, "
            f"maxDD **{R['p6040']['maxdd']}% → {R['blend']['maxdd']}%**, MAR "
            f"{R['p6040']['mar']:.3f} → {R['blend']['mar']:.3f}; alpha "
            f"**{R['alpha']['ann']:+.2f}%/yr, NW t = {R['alpha']['t']:+.2f}**; ΔSharpe "
            f"**{R['dsharpe']['obs']:+.3f}**, CI **[{R['dsharpe']['lo']:+.3f}, "
            f"{R['dsharpe']['hi']:+.3f}]**, p = {R['dsharpe']['p']:.3f}.\n\n"
            "> 💡 **In plain words:** every point estimate says \"better\" — but the direct test of "
            "the claimed quantity never excludes zero, and the alpha's t = 2.10 clears the bar only "
            "at the exact headline configuration. That gap between *plausible* and *certified* is "
            "the whole verdict."
        ),

        md("## 4 · Robustness — where the t = 2.10 breaks"),
        code(
            "if HAVE_REAL:\n"
            "    print('sleeve-size (frozen 5,000-draw CIs):')\n"
            "    for pct, sh, dd, ds, lo, hi, p in R['sleeves']:\n"
            "        print(f'  {pct}%: Sharpe {sh:.3f}  maxDD {dd}%  dS {ds:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  p {p:.3f}')\n"
            "    print('\\ncost sweep (alpha NW t):')\n"
            "    for cb, alp, t_ in R['costs']:\n"
            "        print(f'  {cb:>4.1f} bps: alpha {alp:+.2f}%/yr  t {t_:+.2f}')\n"
            "    print('\\nlookback sweep — recomputed live:')\n"
            "    for lb in (9, 12, 15):\n"
            "        bk = st.tsmom_book(FM, lookback=lb, cost_bps=COST_BPS)\n"
            "        em = bk['net'].reindex(IDX).dropna(); e6 = E6040.reindex(em.index)\n"
            "        al2 = st.nw_alpha(em, e6, lags=6)\n"
            "        print(f'  {lb:>2d}m: book Sharpe {st.sharpe_excess(em):.3f}  alpha NW t {al2[\"t_alpha\"]:+.2f}')\n"
            "    print('\\nNW lags (12m, 5bp): ' + '  '.join(\n"
            "        f'lags={lg}: t={st.nw_alpha(MF_NET, E6040, lags=lg)[\"t_alpha\"]:+.2f}' for lg in (3, 6, 12)))\n"
        ),
        md(
            "Only the canonical **12-month** lookback at **≤5 bps** clears t = 2 (its 9m/15m "
            f"neighbours: t = {R['lookbacks'][0][2]:.2f} / {R['lookbacks'][2][2]:.2f}; 10 bps: "
            f"t = {R['costs'][2][2]:.2f}). A bar cleared only at the snoopable textbook config is "
            "not a certified effect — the same fragility 518 found on the standalone premium "
            "reappears at the portfolio level, as it must.\n\n"
            "> 💡 **In plain words:** if you have to pick the one exact recipe the papers use to get "
            "\"significant\", it isn't robustly significant."
        ),

        md("## 5 · Regime dependence — the 2009-2019 drought"),
        code(
            "if HAVE_REAL:\n"
            "    for lo, hi in [('2001-09', '2008-12'), ('2009-01', '2019-12'), ('2020-01', '2026-05')]:\n"
            "        sl_ = slice(pd.Timestamp(lo), pd.Timestamp(hi) + pd.offsets.MonthEnd(0))\n"
            "        s60 = st.sharpe_excess(E6040.loc[sl_])\n"
            "        sbl = st.sharpe_excess((BLEND - RF_I).loc[sl_])\n"
            "        print(f'{lo} -> {hi}: 60/40 {s60:+.2f}  blend {sbl:+.2f}  dSharpe {sbl-s60:+.3f}')\n"
            "    w60 = (1 + P6040).cumprod(); wbl = (1 + BLEND).cumprod()\n"
            "    dd60 = w60/w60.cummax() - 1; ddbl = wbl/wbl.cummax() - 1\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.fill_between(dd60.index, dd60*100, 0, color=GREY, alpha=.6, label='60/40')\n"
            "    ax.fill_between(ddbl.index, ddbl*100, 0, color=GREEN, alpha=.45, label='+15% sleeve')\n"
            "    ax.set_ylabel('drawdown (%)')\n"
            "    ax.set_title('Drawdowns: the sleeve earns its keep in the storms (GFC, 2022)')\n"
            "    ax.legend(loc='lower left')\n"
            "    plt.show()\n"
        ),
        md(
            "ΔSharpe by regime: **+0.257** (2001-2008) · **−0.107** (2009-2019) · **+0.147** "
            "(2020-2026). An eleven-year stretch where the sleeve made the portfolio *worse* is why "
            "the full-sample CI includes zero — and why the product is behaviourally hard to hold. "
            "This is the Tradability axis's decisive fact."
        ),

        md("## 6 · 2022 attribution — the crisis-alpha year"),
        code(
            "if HAVE_REAL:\n"
            "    for name, ser in [('SPY', TR['SPY']), ('VBMFX', TR['VBMFX']), ('60/40', P6040),\n"
            "                      ('sleeve (funded, net)', MF_FUNDED), ('blend', BLEND),\n"
            "                      ('DBMF live', TR['DBMF']), ('KMLM live', TR['KMLM'])]:\n"
            "        print(f'  {name:<22s}: {st.calendar_year_return(ser, 2022)*100:+7.2f}%')\n"
            "    print(f\"  sleeve contribution to blend ~ +{0.15*st.calendar_year_return(MF_FUNDED, 2022)*100:.1f} pp\")\n"
        ),
        md(
            "One calendar year — descriptive, no significance test possible (and we don't run one). "
            "But it is the exact scenario the product is sold on, delivered live and net of fees "
            "(DBMF +21.6%, KMLM +24.2%) in the only post-inception test the market has offered."
        ),

        md(
            "## 7 · Third axis — bonds in disguise? **BUSTED**\n\n"
            "The skeptic's null: \"trend funds are mostly long fixed-income carry; the sleeve is "
            "leveraged duration, so just buy more bonds.\" Three tests:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eb = TR['VBMFX'].loc[IDX] - RF_I\n"
            "    alb = st.nw_alpha(MF_NET, eb, lags=6)\n"
            "    cb_ = st.corr_ci(MF_NET, eb)\n"
            "    print(f'1) loading: corr(MF, bonds) = {cb_[\"r\"]:+.3f} [{cb_[\"lo\"]:+.3f}, {cb_[\"hi\"]:+.3f}]  '\n"
            "          f'beta {alb[\"beta\"]:+.3f}  R^2 {alb[\"r2\"]*100:.1f}%')\n"
            "    print(f'2) 2022: bonds {st.calendar_year_return(TR[\"VBMFX\"], 2022)*100:+.2f}% vs '\n"
            "          f'sleeve {st.calendar_year_return(MF_FUNDED, 2022)*100:+.2f}% — opposite signs')\n"
            "    p4555 = 0.45*TR['SPY'].loc[IDX] + 0.55*TR['VBMFX'].loc[IDX]\n"
            "    d = st.ann_stats(p4555, RF_I); b = st.ann_stats(BLEND, RF_I)\n"
            "    print(f'3) same 15 pts into bonds instead (45/55): Sharpe {d[\"sharpe\"]:.3f}  '\n"
            "          f'maxDD {d[\"maxdd\"]*100:.1f}%  2022 {st.calendar_year_return(p4555, 2022)*100:+.2f}%')\n"
            "    print(f'   vs the sleeve blend                  : Sharpe {b[\"sharpe\"]:.3f}  '\n"
            "          f'maxDD {b[\"maxdd\"]*100:.1f}%  2022 {st.calendar_year_return(BLEND, 2022)*100:+.2f}%')\n"
        ),
        md(
            "Zero bond loading (R² rounds to **0.0%**), opposite sign in the rising-rate crash "
            "(trend was *short* bonds in 2022), and the bonds-instead portfolio is worse on every "
            "line (Sharpe 0.664 vs 0.738, 2022 −15.15% vs −9.02%). The sleeve is its own animal — "
            "**busted**.\n\n"
            "> 💡 **In plain words:** in the one year the \"it's just bonds\" theory could have been "
            "right, the sleeve made money *by betting against bonds*."
        ),

        md(
            "## 8 · The live wrappers — DBMF/KMLM (2019-07 → 2026-06, net of fees; descriptive)\n\n"
            "The replication is gross of real-world slippage; the ETFs are the honest tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dbmf = TR['DBMF'].dropna().iloc[1:]\n"
            "    lidx = dbmf.index.intersection(P6040_FULL.index).intersection(RF.dropna().index)\n"
            "    p60l, rfl = P6040_FULL.loc[lidx], RF.loc[lidx]\n"
            "    cd = st.corr_ci(dbmf.loc[lidx]-rfl, p60l-rfl)\n"
            "    cr = st.corr_ci(dbmf.loc[lidx]-rfl, MF_NET.reindex(lidx))\n"
            "    print(f'window {lidx.min().date()} -> {lidx.max().date()} ({len(lidx)} months)')\n"
            "    print(f'corr(DBMF, 60/40) = {cd[\"r\"]:+.3f} [{cd[\"lo\"]:+.3f}, {cd[\"hi\"]:+.3f}]')\n"
            "    print(f'corr(DBMF, replication book) = {cr[\"r\"]:+.3f} [{cr[\"lo\"]:+.3f}, {cr[\"hi\"]:+.3f}]')\n"
            "    sd = st.ann_stats(dbmf.loc[lidx], rfl)\n"
            "    print(f'DBMF: CAGR {sd[\"cagr\"]*100:.2f}%  vol {sd[\"vol\"]*100:.1f}%  Sharpe {sd[\"sharpe\"]:.3f}  '\n"
            "          f'maxDD {sd[\"maxdd\"]*100:.1f}%')\n"
            "    bl_l = 0.85*p60l + 0.15*dbmf.loc[lidx]\n"
            "    a_l, b_l = st.ann_stats(p60l, rfl), st.ann_stats(bl_l, rfl)\n"
            "    bs_l = st.bootstrap_dsharpe(p60l-rfl, dbmf.loc[lidx]-rfl, SLEEVE, 0.0, n_draws=1500, seed=595)\n"
            "    print(f'60/40 Sharpe {a_l[\"sharpe\"]:.3f} maxDD {a_l[\"maxdd\"]*100:.1f}%  ->  +15% DBMF '\n"
            "          f'Sharpe {b_l[\"sharpe\"]:.3f} maxDD {b_l[\"maxdd\"]*100:.1f}%')\n"
            "    print(f'live dSharpe {bs_l[\"obs\"]:+.3f}  CI [{bs_l[\"lo\"]:+.3f}, {bs_l[\"hi\"]:+.3f}]  '\n"
            "          f'p {bs_l[\"p_onesided\"]:.3f}   (light rerun; frozen: '\n"
            "          f'{R[\"live\"][\"ds\"]:+.3f} [{R[\"live\"][\"lo\"]:+.3f}, {R[\"live\"][\"hi\"]:+.3f}], p {R[\"live\"][\"p\"]:.3f})')\n"
        ),
        md(
            f"DBMF tracks the replication at **{R['corr_dbmf_book'][0]:+.3f}** (same animal), runs "
            f"**{R['corr_dbmf'][0]:+.3f}** to the 60/40 live, and the live blend repeats the story: "
            f"Sharpe {R['live']['s6040']:.3f} → {R['live']['sblend']:.3f}, maxDD "
            f"{R['live']['dd6040']}% → {R['live']['ddblend']}%, ΔSharpe {R['live']['ds']:+.3f} with "
            "a CI that includes zero. 84 months cannot certify a Sharpe difference — but nothing in "
            "the live tape contradicts the replication."
        ),

        md(
            "## 9 · Random-sign placebo — is the improvement just \"any uncorrelated thing\"?\n\n"
            "Same sizing, lag, costs and fee; the trend sign replaced by an iid coin, **24 seeds** "
            "(house rule: random baselines average ≥ 20 seeds)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows_p = st.random_sign_book(FM, range(1, 25), cost_bps=COST_BPS)\n"
            "    shs, dss = [], []\n"
            "    for r_ in rows_p:\n"
            "        em = r_['net'].reindex(IDX).dropna(); e6 = E6040.reindex(em.index)\n"
            "        shs.append(st.sharpe_excess(em))\n"
            "        blx = (1-SLEEVE)*e6 + SLEEVE*(em - FEE/12)\n"
            "        dss.append(st.sharpe_excess(blx) - st.sharpe_excess(e6))\n"
            "    print(f'{len(rows_p)} seeds: mean placebo book Sharpe {np.mean(shs):+.3f} (sd {np.std(shs):.3f})')\n"
            "    print(f'          mean placebo dSharpe {np.mean(dss):+.4f} (sd {np.std(dss):.4f})')\n"
            "    print(f\"observed trend-sleeve dSharpe: {R['dsharpe']['obs']:+.3f}\")\n"
        ),
        md(
            "A zero-edge uncorrelated sleeve **hurts** the blend (mean ΔSharpe "
            f"**{R['placebo']['ds']:+.4f}**): with the fee and costs charged, \"low correlation\" "
            "alone is a drag, not a benefit. The observed +0.095 requires the trend signal itself — "
            "the improvement, whatever its significance, is not diversification arithmetic."
        ),

        md(
            "## 10 · Synthetic control — the machinery is faithful\n\n"
            "Seeded joint (stock, bond, MF) world, 312 months, corr 0, fee charged, planted MF "
            "Sharpe as the knob. The detector must stay quiet at 0 and fire at 0.8. *(Machinery "
            "proof only — never cited in support of a stamp.)*"
        ),
        code(
            "for ms in (0.0, 0.8):\n"
            "    w = data.synthetic_world(n_months=312, mf_sharpe=ms, rho_bond=0.0, seed=595)\n"
            "    e60s = 0.6*w['stock'] + 0.4*w['bond']\n"
            "    bs_s = st.bootstrap_dsharpe(e60s, w['mf'], 0.15, 0.0085, n_draws=1000, seed=595)\n"
            "    al_s = st.nw_alpha(w['mf'], e60s, lags=6)\n"
            "    print(f'planted MF Sharpe {ms:.1f}: dSharpe {bs_s[\"obs\"]:+.3f}  '\n"
            "          f'CI [{bs_s[\"lo\"]:+.3f}, {bs_s[\"hi\"]:+.3f}]  p {bs_s[\"p_onesided\"]:.3f}  '\n"
            "          f'alpha NW t {al_s[\"t_alpha\"]:+.2f}')\n"
        ),

        md(
            "## Verdict\n\n"
            "- **Signal — MIXED** *(Real on the correlation · Weak on the certified improvement)*: "
            f"corr **{R['corr_6040'][0]:+.3f}** [{R['corr_6040'][1]:+.3f}, {R['corr_6040'][2]:+.3f}] "
            "is decisively near-zero (live wrappers negative), but ΔSharpe "
            f"**{R['dsharpe']['obs']:+.3f}** carries a CI of [{R['dsharpe']['lo']:+.3f}, "
            f"{R['dsharpe']['hi']:+.3f}] and the alpha t = {R['alpha']['t']:+.2f} survives only the "
            "canonical 12m/≤5bp config (9m: +1.35, 15m: +1.67, 10bp: +1.92). Mild futures-panel "
            "survivorship named.\n"
            "- **Tradability — FRAGILE**: one-ticket access exists and every headline pays the "
            "0.85%/yr fee, but the benefit is regime-dependent (2009-2019 ΔSharpe −0.107) and "
            "statistically uncertified.\n"
            "- **Bonds in disguise? — BUSTED**: zero bond loading, opposite sign in 2022, and "
            "more-bonds-instead is worse on every line.\n\n"
            "Frozen numbers: [`docs/results.md`](../docs/results.md) (fingerprint `"
            + R['fingerprint'] + "`). *Research & education, not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)
