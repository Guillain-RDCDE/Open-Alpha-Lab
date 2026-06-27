"""Generate the two narrative notebooks for Study 508 (Momentum-Crashes).

    python notebooks/build_notebooks.py

Both notebooks follow the desk's seven beats. Synthetic figures run anywhere, offline and
deterministic; the real-panel cells use the cached yfinance parquets under ../_cache/ if present
and otherwise quote the frozen headline numbers in ``R`` (the single source of truth that mirrors
docs/results.md exactly).
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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    asof="2026-06-26",
    n_days=3424, n_tickers=38, n_hold=151,
    date_start="2012-05-18", date_end="2025-12-30",
    hold_start="2013-05", hold_end="2025-11",
    fp="e76dff230f36", fp_prices="e038fa42cc02", fp_spy="35681d7ebf6e",
    cost_bps=10.0, borrow_bps=50.0,
    # WML quintile (the level test)
    gross=-0.34, net=-1.43, vol=16.10, sharpe=-0.021, t=-0.072,
    hit=54.3, dd=-44.9, worst=-17.7, skew=-0.63, turn=24.8,
    placebo_p=0.537, placebo_mean=-0.07,
    # crash anatomy
    dd_peak="2020-09-30", dd_trough="2025-10-31",
    crash_months={
        "2020-10-31": (-17.7, 8.1, 25.8),
        "2025-07-31": (-14.0, -0.7, 13.2),
        "2022-12-31": (-13.5, 0.4, 13.9),
        "2021-01-31": (-12.6, -0.8, 11.8),
        "2023-02-28": (-9.0, 0.9, 9.9),
        "2016-01-31": (-7.2, -4.2, 3.0),
    },
    # Daniel-Moskowitz regimes (bear lookback = 12m)
    regimes={
        "all": (-0.34, 151),
        "calm (non-bear)": (0.67, 134),
        "bear (any)": (-8.25, 17),
        "bear & down": (-27.82, 10),
        "panic (bear & rebound)": (19.71, 7),
    },
    # vol-scaled repair
    vs_mean=2.21, vs_vol=15.99, vs_sharpe=0.138, vs_t=0.492,
    vs_dd=-35.6, vs_worst=-15.9, vs_skew=-0.27,
    # synthetic control
    ctrl={0.00: (5.46, 1.14), 0.15: (19.77, 4.64), 0.30: (45.54, 9.80), 0.50: (84.76, 13.96)},
    annual={2013: 1.1, 2014: -4.1, 2015: 26.0, 2016: -26.7, 2017: 28.2, 2018: 2.1,
            2019: 3.1, 2020: -2.7, 2021: -23.5, 2022: 4.8, 2023: -0.3, 2024: 22.7,
            2025: -29.6},
)


def _boot():
    import pprint
    return (
        "import sys, os\n"
        'sys.path.insert(0, os.path.abspath(".."))          # the study package\n'
        'sys.path.insert(0, os.path.abspath("../../.."))    # repo root\n'
        'import warnings; warnings.filterwarnings("ignore")\n'
        "import numpy as np, pandas as pd\n"
        "# Frozen headline numbers (mirror of docs/results.md) -- the offline fallback for CI.\n"
        "R = " + pprint.pformat(R, width=100) + "\n"
        + _BOOT_TAIL
    )


_BOOT_TAIL = """\
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from momentum_crashes import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return os.path.exists(os.path.join(study, "_cache", "momcrash_prices.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices = data.drop_partial_last_month(data.fetch_panel())
    market = data.fetch_market()
    mp = st.to_monthly(prices)
    mkt_m = st.to_monthly_series(market) if not market.empty else market
    ls = st.long_short(mp, frac=0.20, cost_bps=10.0, borrow_ann_bps=50.0)
    s_g = st.summary(ls["wml_gross"]); s_n = st.summary(ls["wml_net"])
    vs = st.vol_scaled(ls, col="wml_gross"); s_vs = st.summary(vs)
    print(f"WML gross: {s_g['mean']*100:+.2f}%/yr  HAC t={s_g['tstat']:+.3f}"
          f"  skew={s_g['skew']:+.2f}  maxDD={s_g['max_dd']*100:.1f}%")
    print(f"Vol-scaled: Sharpe {s_vs['sharpe']:+.3f}  skew={s_vs['skew']:+.2f}"
          f"  maxDD={s_vs['max_dd']*100:.1f}%")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Momentum-Crashes -- the factor that prints money until it detonates\n"
            "### Daniel-Moskowitz (2016): momentum's rare, savage left tail -- and the vol-scaling repair\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Crash%20%2B%20repair%3F: Confirmed](https://img.shields.io/badge/Crash%20%2B%20repair%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Cross-sectional momentum -- buy past winners, short past losers -- is the most "
            "replicated anomaly in finance. But in 2016 Kent Daniel and Tobias Moskowitz "
            "published the uncomfortable footnote: momentum carries **rare, brutal crashes**. "
            "They don't strike randomly -- they detonate in the *rebound out of a bear market*, "
            "when the beaten-down past-loser stocks (which the strategy is **short**) snap back "
            "violently and the dollar-neutral book gets run over.\n\n"
            "We build the canonical 12-1 momentum book on a large-cap survivor basket, then ask "
            "two separate questions: (1) does the *premium* still pay? and (2) is the *crash* "
            "real -- and can **vol-scaling** repair it?\n\n"
            "> **This is the plain-language layer.** Want the HAC inference, the placebo null, "
            "the regime sweep, and the vol-scaling comparison? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(_boot()),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the 12-1 WML spread pay? | **No.** {R['gross']:+.2f}%/yr gross, HAC "
            f"*t* = **{R['t']:+.3f}**, placebo *p* = **{R['placebo_p']:.2f}** -- a coin. |\n"
            f"| Is it tradable? | **No.** {R['net']:+.2f}%/yr net, **{R['dd']:.1f}%** "
            "drawdown, fat left tail. A mirage. |\n"
            f"| Is the crash real? | **Yes.** Worst month **{R['worst']:.1f}%** -- and the "
            f"*losers* (the shorts) ripped **+{R['crash_months']['2020-10-31'][2]:.1f}%** that "
            "month. Skew **{:+.2f}**. |\n".format(R['skew']) +
            f"| Does vol-scaling repair it? | **Yes.** Skew **{R['skew']:+.2f} -> "
            f"{R['vs_skew']:+.2f}**, drawdown **{R['dd']:.1f}% -> {R['vs_dd']:.1f}%**. |\n\n"
            "> The premium is gone on this survivor basket -- but the **crash dynamics are "
            "exactly as Daniel-Moskowitz describe**, and vol-scaling tames the tail without "
            "ever conjuring a tradable edge."
        ),

        md(
            "## 1 -- The claim\n\n"
            "> *\"Momentum strategies experience infrequent but large crashes. These crashes "
            "occur in panic states, following market declines and contemporaneous with market "
            "rebounds. The crashes are forecastable... and a dynamic, volatility-scaled momentum "
            "strategy nearly doubles the Sharpe ratio and eliminates the crashes.\"*\n\n"
            "-- Daniel & Moskowitz (2016), *Journal of Financial Economics*\n\n"
            "The mechanism is the short leg. After a market crash, the past losers are the "
            "battered, high-beta, distressed names. When the market turns, **those names rebound "
            "hardest** -- and a momentum book is short them. The long (winner) leg, full of "
            "defensive survivors, can't keep up. The dollar-neutral spread takes the hit."
        ),

        md(
            "## 2 -- So what?\n\n"
            "Momentum is everywhere: factor ETFs, AQR's flagship funds, risk-parity overlays, "
            "trend-followers. If the *level* premium has decayed but the *crash* risk has not, "
            "investors are being paid little to carry a fat, correlated left tail that fires "
            "exactly when everything else is recovering -- a written-call payoff dressed up as "
            "alpha. The Daniel-Moskowitz repair (vol-scaling) is the standard institutional fix; "
            "we test whether it earns its keep here."
        ),

        md(
            "## 3 -- How would we even know?\n\n"
            "Four disciplines keep us honest:\n\n"
            "1. **One execution lag.** Form the 12-1 signal on the month-end close; enter the "
            "*next* month. No same-bar fill, no look-ahead.\n"
            "2. **A placebo null.** Shuffle which stock the momentum signal points at and "
            "recompute the spread 2000 times. A real edge beats the shuffle; a data-mined one "
            "scores *p* ~ 0.5.\n"
            "3. **Name the survivorship bias on the *signal* axis.** The basket is survivors. "
            "The loser leg's worst names -- the ones that trended into delisting -- are absent, "
            "so the premium is an *upper bound* and the crash an *under*-statement.\n"
            "4. **Separate the level from the shape.** A flat mean can still hide a vicious "
            "left tail. We measure skewness, the worst months, and the deepest drawdown "
            "explicitly."
        ),

        md(
            "## 4 -- The teardown\n\n"
            "**First, does the engine work when momentum is planted?**"
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.15, 0.30, 0.50), frac=0.2, seed=508)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREY if abs(m) < 8 else GREEN for m in ctrl['wml_mean_ann']*100]\n"
            "ax.bar(ctrl['mom_strength'].astype(str), ctrl['wml_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['wml_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 1.5, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted momentum strength'); ax.set_ylabel('WML mean (%/yr)')\n"
            "ax.set_title('Synthetic control: the engine finds momentum when it is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: ~zero at the null, rising monotonically as the planted "
            "drift grows. So the flat real-tape verdict reflects **the market**, not a broken "
            "detector."
        ),

        md("**Now the honest test on the real yfinance panel -- and the crash.**"),
        code(
            "if HAVE_REAL:\n"
            "    gross, t_, dd, worst, skew = (s_g['mean']*100, s_g['tstat'],\n"
            "                                  s_g['max_dd']*100, s_g['worst']*100, s_g['skew'])\n"
            "    ann = ls['wml_gross'].groupby(ls.index.year).apply(\n"
            "        lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    gross, t_, dd, worst, skew = (R['gross'], R['t'], R['dd'], R['worst'], R['skew'])\n"
            "    ann = pd.Series(R['annual'])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "col_yr = [GREEN if v > 0 else RED for v in ann.values]\n"
            "axes[0].bar(ann.index, ann.values, color=col_yr)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('year'); axes[0].set_ylabel('WML return (%/yr)')\n"
            "axes[0].set_title(f'12-1 momentum year-by-year | mean {gross:+.1f}% t={t_:+.2f}')\n"
            "\n"
            "# The crash months: who moved -- winners or losers?\n"
            "cm = R['crash_months']\n"
            "dates = list(cm.keys()); wml=[cm[d][0] for d in dates]\n"
            "win=[cm[d][1] for d in dates]; los=[cm[d][2] for d in dates]\n"
            "x = np.arange(len(dates))\n"
            "axes[1].bar(x-0.25, win, 0.25, label='winner leg', color=AMBER)\n"
            "axes[1].bar(x, los, 0.25, label='loser leg (short)', color=RED)\n"
            "axes[1].bar(x+0.25, wml, 0.25, label='WML', color=GREY)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels([d[:7] for d in dates], rotation=45, fontsize=7)\n"
            "axes[1].set_ylabel('%'); axes[1].legend(fontsize=8)\n"
            "axes[1].set_title('The 6 worst WML months: the loser leg snaps back')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'WML {gross:+.2f}%/yr | HAC t={t_:+.3f} | maxDD {dd:.1f}% | worst {worst:+.1f}% | skew {skew:+.2f}')"
        ),
        md(
            f"The premium is **{R['gross']:+.2f}%/yr** at HAC *t* = **{R['t']:+.3f}** -- nothing. "
            "But look at the right panel: in **October 2020** (the vaccine-news rebound) the "
            f"loser leg ripped **+{R['crash_months']['2020-10-31'][2]:.1f}%** while the winners "
            f"managed only +{R['crash_months']['2020-10-31'][1]:.1f}%, costing the book "
            f"**{R['crash_months']['2020-10-31'][0]:.1f}%** in a single month. Every crash month "
            "is the same story: the shorts snap back."
        ),

        md(
            "### Does momentum behave differently in bear markets?\n\n"
            "Daniel-Moskowitz say momentum's return is *regime-dependent*: fine in calm "
            "markets, lethal in bear-market rebounds."
        ),
        code(
            "if HAVE_REAL and not mkt_m.empty:\n"
            "    rs = st.regime_split(ls, mkt_m, col='wml_gross')\n"
            "    reg = (rs['wml_mean_ann']*100)\n"
            "else:\n"
            "    reg = pd.Series({k: v[0] for k, v in R['regimes'].items()})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if v > 0 else RED for v in reg.values]\n"
            "ax.barh(range(len(reg)), reg.values, color=col)\n"
            "ax.set_yticks(range(len(reg))); ax.set_yticklabels(reg.index)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('WML mean (%/yr, annualised)')\n"
            "ax.set_title('Momentum is regime-dependent: it bleeds in bear-and-down months')\n"
            "ax.invert_yaxis(); plt.tight_layout(); plt.show()\n"
            "print(reg.round(2).to_string())"
        ),
        md(
            f"In **calm** markets momentum is flat-to-positive (+{R['regimes']['calm (non-bear)'][0]:.1f}%/yr); "
            f"in **bear-and-down** months it bleeds **{R['regimes']['bear & down'][0]:.1f}%/yr**. "
            "The regime-dependence is exactly the Daniel-Moskowitz claim."
        ),

        md(
            "### The repair: vol-scaled (dynamic) momentum\n\n"
            "Scale the position inversely to the spread's own recent volatility -- so the book "
            "shrinks before the storm. Does it tame the tail?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq_g = (1 + ls['wml_gross']).cumprod()\n"
            "    eq_v = (1 + vs).cumprod()\n"
            "    g_skew, v_skew = s_g['skew'], s_vs['skew']\n"
            "    g_dd, v_dd = s_g['max_dd']*100, s_vs['max_dd']*100\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "    ax.plot(eq_g.index, eq_g.values, c=RED, lw=1.6, label=f'gross (skew {g_skew:+.2f}, DD {g_dd:.0f}%)')\n"
            "    ax.plot(eq_v.index, eq_v.values, c=GREEN, lw=1.6, label=f'vol-scaled (skew {v_skew:+.2f}, DD {v_dd:.0f}%)')\n"
            "    ax.axhline(1, c='k', lw=.8, ls='--')\n"
            "    ax.set_ylabel('cumulative wealth'); ax.legend()\n"
            "    ax.set_title('Vol-scaling repairs the shape: a shallower drawdown, a thinner left tail')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'gross  : skew {g_skew:+.2f} | maxDD {g_dd:.1f}% | Sharpe {s_g[\"sharpe\"]:+.3f}')\n"
            "    print(f'scaled : skew {v_skew:+.2f} | maxDD {v_dd:.1f}% | Sharpe {s_vs[\"sharpe\"]:+.3f}')\n"
            "else:\n"
            "    print(f'Frozen: gross skew {R[\"skew\"]:+.2f} DD {R[\"dd\"]:.1f}% -> scaled skew {R[\"vs_skew\"]:+.2f} DD {R[\"vs_dd\"]:.1f}%')"
        ),
        md(
            f"Vol-scaling halves the negative skew (**{R['skew']:+.2f} -> {R['vs_skew']:+.2f}**) "
            f"and shaves ~9 points off the deepest drawdown (**{R['dd']:.1f}% -> {R['vs_dd']:.1f}%**). "
            "The Sharpe flips from slightly negative to slightly positive -- but at *t* = "
            f"+{R['vs_t']:.2f} it is still nowhere near tradable. It fixes the **shape**, not "
            "the **level**."
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** WML {R['gross']:+.2f}%/yr, HAC *t* = {R['t']:+.3f}, "
            f"placebo *p* = {R['placebo_p']:.2f}. The premium is gone on this survivor basket "
            "(which already biases it *up*).\n"
            f"- **Tradability -- MIRAGE.** {R['net']:+.2f}%/yr net, {R['dd']:.1f}% drawdown, "
            f"{R['skew']:+.2f} skew. No expectancy, fat crash risk.\n"
            "- **Crash + repair -- CONFIRMED.** The crash is real and loser-leg-driven; "
            "vol-scaling demonstrably tames the tail. Daniel-Moskowitz, faithfully reproduced."
        ),

        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No. Even setting aside the flat mean: the strategy turns over ~25%/month (two-sided "
            "trading costs), requires shorting the past losers (borrow fees, recall risk on "
            "exactly the distressed names that are hardest to borrow), and carries a -45% "
            "drawdown that lands when the rest of your book is recovering -- the worst possible "
            "correlation. Vol-scaling makes the ride less violent but does not turn a flat "
            "premium into a profit."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **[Study 507 -- Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)**: "
            "the level test on the same basket (decile vs quintile).\n"
            "- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: the "
            "crash-dodging cousin -- strip out the time-varying market beta the crash rides on.\n"
            "- **[Study 25 -- Clean-Slate](../../25-clean-slate/)**: residual momentum from a "
            "different angle.\n"
            "- **Broader, delisting-complete universe.** The true crash is *worse* than we see "
            "here -- the survivor basket hides the loser names that snapped back hardest.\n\n"
            "*Think the vol-scaled premium clears t > 2 on a delisting-complete universe net of "
            "borrow? Fork this, add the delisted shorts, and show it. That is the bar.*"
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
            "# Momentum-Crashes -- a quantitative teardown\n"
            "### 12-1 WML * HAC inference * label-shuffle placebo * regime conditioning * vol-scaling\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Crash%20%2B%20repair%3F: Confirmed](https://img.shields.io/badge/Crash%20%2B%20repair%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. Daniel-Moskowitz (2016) on the Jegadeesh-Titman "
            "(1993) 12-1 factor: rank by trailing 12-month return skipping the most recent month, "
            "long the top quintile / short the bottom, dollar-neutral, monthly hold.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, "
            f"{R['n_tickers']} S&P 500 large-caps, {R['date_start'][:4]}-{R['date_end'][:4]}, "
            f"as-of {R['asof']}, fingerprint `{R['fp']}`. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship is named on the SIGNAL axis:** basket = current large-caps "
            "projected backwards. The premium is an upper bound and the crash an under-statement."
        ),
        code(_boot()),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | WML **{R['gross']:+.2f}%/yr**, HAC *t* = "
            f"**{R['t']:+.3f}**, placebo *p* = **{R['placebo_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Net **{R['net']:+.2f}%/yr**, max DD "
            f"**{R['dd']:.1f}%**, skew **{R['skew']:+.2f}**. |\n"
            f"| **Crash + repair?** | `CONFIRMED` | Worst month **{R['worst']:.1f}%** "
            f"(loser leg +{R['crash_months']['2020-10-31'][2]:.1f}%); bear-and-down "
            f"**{R['regimes']['bear & down'][0]:.1f}%/yr**; vol-scaling skew "
            f"**{R['skew']:+.2f} -> {R['vs_skew']:+.2f}**, DD **{R['dd']:.1f}% -> {R['vs_dd']:.1f}%**. |\n\n"
            "> The level premium is dead; the crash is alive and exactly as the paper describes; "
            "vol-scaling repairs the shape but not the level."
        ),

        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r^{WML}_t = \\bar r^{win}_t - \\bar r^{los}_t$ be the dollar-neutral 12-1 "
            "spread. Daniel-Moskowitz (2016) assert:\n\n"
            "- **H1 (premium).** $\\mathbb{E}[r^{WML}] > 0$ unconditionally.\n"
            "- **H2 (crash).** The distribution of $r^{WML}$ is severely **left-skewed**; the "
            "worst realisations cluster in *panic states* (post-bear, contemporaneous rebound).\n"
            "- **H3 (regime).** $\\mathbb{E}[r^{WML}\\mid \\text{bear}] \\ll "
            "\\mathbb{E}[r^{WML}\\mid \\text{calm}]$.\n"
            "- **H4 (repair).** A constant-vol scaling $w_t = \\sigma^* / \\hat\\sigma_t$ lifts "
            "the Sharpe and removes the crash.\n\n"
            "On this survivor basket we **reject H1** (flat), **confirm H2** (skew "
            f"{R['skew']:+.2f}), **confirm H3** (bear-and-down {R['regimes']['bear & down'][0]:.1f}%/yr "
            f"vs calm +{R['regimes']['calm (non-bear)'][0]:.1f}%/yr), and **partially confirm H4** "
            "(skew and drawdown shrink; Sharpe flips positive but still |t| < 2)."
        ),

        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "If the premium is gone but the crash is not, momentum is a written-call: small, "
            "steady-ish carry punctuated by a violent loss that fires when the broad market is "
            "rebounding -- the worst time to take it. Factor-ETF buyers paying for 'momentum "
            "alpha' may be paying to short the recovery."
        ),

        md(
            "## 3 -- The protocol\n\n"
            f"- **Signal.** 12-1 trailing return (lookback {12}m, skip {1}m), no look-ahead.\n"
            "- **Book.** Long top quintile, short bottom quintile, equal-weight, dollar-neutral.\n"
            "- **Execution lag.** Form on month-end close, hold the next month (single forward "
            "shift -- no same-bar fill).\n"
            f"- **Costs.** {R['cost_bps']:.0f}bps/leg/rebalance + {R['borrow_bps']:.0f}bps/yr "
            "borrow on the short leg. Gross vs net.\n"
            "- **Inference.** Newey-West HAC *t* on monthly WML; 2000-permutation label-shuffle "
            "placebo.\n"
            "- **Regimes.** Bear = trailing-12m SPY < 0; panic = bear & market up this month.\n"
            "- **Repair.** Vol-scaled to 12%/yr target on trailing-6m realised vol, cap 3x.\n"
            f"- **Universe caveat.** {R['n_tickers']} large-cap survivors; survivorship-biased."
        ),

        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the WML engine is a faithful detector\n\n"
            "Sweep the planted momentum strength; the WML mean and HAC *t* should rise "
            "monotonically from ~0 at the null."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.15, 0.30, 0.50, 0.70), frac=0.2, seed=508)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ctrl['mom_strength'], ctrl['wml_mean_ann']*100, 'o-', c=AMBER, lw=2)\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(ctrl['mom_strength'], ctrl['tstat'], 's--', c=GREY, lw=1.5)\n"
            "ax2.axhline(2, c=GREEN, lw=1, ls=':'); ax2.axhline(-2, c=GREEN, lw=1, ls=':')\n"
            "ax.set_xlabel('planted momentum strength'); ax.set_ylabel('WML mean (%/yr)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Positive control: WML mean and t are monotone in planted strength')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),

        md(
            "### 4b -- The real WML: flat mean, fat left tail\n\n"
            "The level is nothing; the distribution is the story."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = ls['wml_gross']*100\n"
            "    g = s_g\n"
            "else:\n"
            "    r = pd.Series(np.random.default_rng(0).normal(R['gross']/12, R['vol']/np.sqrt(12)/100, 151))\n"
            "    g = dict(mean=R['gross']/100, tstat=R['t'], skew=R['skew'], worst=R['worst']/100, max_dd=R['dd']/100)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(r, bins=30, color=AMBER, edgecolor='white', alpha=0.85)\n"
            "ax.axvline(float(np.mean(r)), c=GREY, lw=2, ls='--', label=f'mean {float(np.mean(r)):+.2f}%/mo')\n"
            "ax.axvline(float(np.min(r)), c=RED, lw=2, ls=':', label=f'worst {float(np.min(r)):+.1f}%')\n"
            "ax.set_xlabel('monthly WML return (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'WML return distribution | skew {g[\"skew\"]:+.2f} (fat left tail)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean {g[\"mean\"]*100:+.2f}%/yr | HAC t {g[\"tstat\"]:+.3f} | skew {g[\"skew\"]:+.2f} | worst {g[\"worst\"]*100:+.1f}%')"
        ),

        md(
            "### 4c -- The placebo null\n\n"
            "Shuffle the signal-to-stock mapping 2000 times; the real mean should sit inside the "
            "placebo cloud if the edge is noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(mp, frac=0.2, n_perm=2000, seed=508)\n"
            "    print(f\"Real WML mean : {pb['real_mean_ann']*100:+.2f}%/yr\")\n"
            "    print(f\"Placebo mean  : {pb['placebo_mean_ann']*100:+.2f}%/yr\")\n"
            "    print(f\"One-sided p   : {pb['p_value']:.3f}\")\n"
            "else:\n"
            "    print(f\"Frozen: real {R['gross']:+.2f}%/yr vs placebo {R['placebo_mean']:+.2f}%/yr, p={R['placebo_p']:.3f}\")"
        ),
        md(
            f"> Placebo *p* = **{R['placebo_p']:.2f}** -- the real WML mean is squarely inside "
            "the shuffled-label cloud. No edge."
        ),

        md(
            "### 4d -- The crash anatomy: worst months and the drawdown episode"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ct = st.crash_table(ls, k=8)\n"
            "    ep = st.max_drawdown_episode(ls['wml_gross'])\n"
            "    print('Deepest drawdown: {:.1f}%  ({} -> {})'.format(\n"
            "          ep['max_dd']*100, ep['peak'].date(), ep['trough'].date()))\n"
            "    print(ct.mul(100).round(1).to_string())\n"
            "    eq = (1 + ls['wml_gross']).cumprod(); dd = (eq/eq.cummax()-1)*100\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    a1.plot(eq.index, eq.values, c=AMBER, lw=1.6); a1.axhline(1, c='k', lw=.8, ls='--')\n"
            "    a1.set_ylabel('cumulative wealth'); a1.set_title('WML equity curve and drawdown')\n"
            "    a2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.4)\n"
            "    a2.set_ylabel('drawdown (%)'); a2.set_xlabel('date'); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Frozen worst month:', R['worst'], '%  maxDD', R['dd'], '%')\n"
            "    print('Peak', R['dd_peak'], '-> trough', R['dd_trough'])"
        ),
        md(
            f"> The deepest drawdown is **{R['dd']:.1f}%** ({R['dd_peak']} -> {R['dd_trough']}). "
            "Note the leg attribution: in the crash table the **loser (short) leg** posts the "
            "big positive returns -- the snap-back is the engine of the crash."
        ),

        md(
            "### 4e -- Regime conditioning and the bear-lookback sweep\n\n"
            "Daniel-Moskowitz condition on bear/panic states. The bear-lookback window matters "
            "on a fast-recovering large-cap tape: a 24-month window almost never fires; 12 "
            "months resolves the COVID/2022 episodes."
        ),
        code(
            "if HAVE_REAL and not mkt_m.empty:\n"
            "    rows = []\n"
            "    for lb in [6, 12, 24]:\n"
            "        rs = st.regime_split(ls, mkt_m, bear_lookback=lb, col='wml_gross')\n"
            "        for reg, rr in rs.iterrows():\n"
            "            rows.append({'lookback': lb, 'regime': reg,\n"
            "                         'wml_%': rr['wml_mean_ann']*100, 'n': int(rr['n_months'])})\n"
            "    sweep = pd.DataFrame(rows)\n"
            "    print(sweep.round(2).to_string(index=False))\n"
            "    rs12 = st.regime_split(ls, mkt_m, bear_lookback=12, col='wml_gross')\n"
            "    reg = rs12['wml_mean_ann']*100\n"
            "else:\n"
            "    reg = pd.Series({k: v[0] for k, v in R['regimes'].items()})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if v > 0 else RED for v in reg.values]\n"
            "ax.barh(range(len(reg)), reg.values, color=col)\n"
            "ax.set_yticks(range(len(reg))); ax.set_yticklabels(reg.index)\n"
            "ax.axvline(0, c='k', lw=1); ax.invert_yaxis()\n"
            "ax.set_xlabel('WML mean (%/yr)'); ax.set_title('WML by market regime (bear lookback = 12m)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> Calm: **+{R['regimes']['calm (non-bear)'][0]:.1f}%/yr** "
            f"(n={R['regimes']['calm (non-bear)'][1]}); bear-and-down: "
            f"**{R['regimes']['bear & down'][0]:.1f}%/yr** (n={R['regimes']['bear & down'][1]}). "
            "The regime-dependence is the cleanest confirmation of the paper on this tape."
        ),

        md(
            "### 4f -- The vol-scaling repair, side by side"
        ),
        code(
            "if HAVE_REAL:\n"
            "    comp = pd.DataFrame({\n"
            "        'gross': [s_g['mean']*100, s_g['vol']*100, s_g['sharpe'], s_g['tstat'],\n"
            "                  s_g['max_dd']*100, s_g['worst']*100, s_g['skew']],\n"
            "        'vol-scaled': [s_vs['mean']*100, s_vs['vol']*100, s_vs['sharpe'], s_vs['tstat'],\n"
            "                       s_vs['max_dd']*100, s_vs['worst']*100, s_vs['skew']],\n"
            "    }, index=['mean %/yr','vol %/yr','Sharpe','HAC t','maxDD %','worst %','skew'])\n"
            "    print(comp.round(3).to_string())\n"
            "else:\n"
            "    print('Frozen: Sharpe', R['sharpe'], '->', R['vs_sharpe'],\n"
            "          '| skew', R['skew'], '->', R['vs_skew'], '| DD', R['dd'], '->', R['vs_dd'])"
        ),
        md(
            f"> Vol-scaling lifts the Sharpe ({R['sharpe']:+.3f} -> {R['vs_sharpe']:+.3f}), "
            f"halves the skew ({R['skew']:+.2f} -> {R['vs_skew']:+.2f}), and shaves the drawdown "
            f"({R['dd']:.1f}% -> {R['vs_dd']:.1f}%). It repairs the **shape**. But at HAC *t* = "
            f"**+{R['vs_t']:.2f}** it is still not a tradable **level** -- the premium was never "
            "there to scale."
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- WML {R['gross']:+.2f}%/yr, HAC *t* = {R['t']:+.3f}, "
            f"placebo *p* = {R['placebo_p']:.2f}. The most replicated anomaly in finance is "
            "flat on this survivor basket.\n"
            f"- **Tradability `MIRAGE`** -- {R['net']:+.2f}%/yr net, {R['dd']:.1f}% drawdown, "
            f"{R['skew']:+.2f} skew, {R['turn']:.0f}%/mo turnover. Nothing to monetise, fat tail "
            "to carry.\n"
            "- **Crash + repair `CONFIRMED`** -- the crash is real, loser-leg-driven, and "
            "regime-dependent; vol-scaling tames the tail without manufacturing a level. "
            "Daniel-Moskowitz (2016), reproduced honestly."
        ),

        md(
            "## 6 -- Could you trade it?\n\n"
            "No. Two-sided turnover ~25%/month, a short leg made of distressed/hard-to-borrow "
            "names, a -45% drawdown that arrives during market recoveries (worst-case "
            "correlation), and a mean the tape cannot tell from zero. Vol-scaling is the right "
            "*risk* fix, but it scales a premium that, here, isn't there."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **[Study 507 -- Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)**: "
            "the decile-vs-quintile level test on the same basket.\n"
            "- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: the "
            "crash-dodging variant -- residualise out the time-varying market beta the crash rides.\n"
            "- **[Study 25 -- Clean-Slate](../../25-clean-slate/)**: residual momentum, a "
            "different lens.\n"
            "- **Delisting-complete universe.** Add the loser names that delisted; the crash "
            "deepens and the premium likely falls further. That is the honest universe.\n"
            "- **Earlier sample.** 1927-2013 (Daniel-Moskowitz's window) contains 1932 and 2009 "
            "-- the canonical momentum crashes our 2012-start tape never sees.\n\n"
            "*Think vol-scaled momentum clears t > 2 net of borrow on a delisting-complete "
            "universe? Fork this and show it. That is the bar.*"
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
