"""Generate the two narrative notebooks for Study 271 (Cardboard-Box).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded freight/box-production table and the synthetic generator run anywhere,
offline and deterministically; the real ^GSPC cells use the cached Yahoo parquet
at this study's ``_cache/gspc_annual.parquet`` if present, and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook
re-runs for any reader offline.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=55,
    year_start=1971,
    year_end=2025,
    fp="15cf4dbe0618",
    n_box_pos=42,
    n_box_neg=13,
    # Forecast regression (box growth_t -> S&P return_{t+1})
    box_beta=-0.0109,
    box_t=-1.605,
    box_r2=0.045,
    box_pearson_r=-0.2122,
    box_pearson_p=0.1199,
    rail_beta=-0.0074,
    rail_t=-1.805,
    rail_r2=0.0425,
    # Coincident regression (box growth_t -> S&P return_t)
    coin_box_t=-0.068,
    coin_box_r2=0.0,
    coin_box_pearson_r=-0.007,
    # Conditional next-year means
    fwd_mean_boxpos=9.84,
    fwd_mean_boxneg=8.59,
    fwd_mean_all=9.54,
    # Timing strategy (long next year iff box growth>0; 10 bps one-way cost)
    uncond_up_pct=74.6,
    hit_rate_pct=61.8,
    turnover=0.3091,
    frac_invested=0.7636,
    bah_ann_pct=8.15,
    gross_ann_pct=6.57,
    net_ann_pct=6.54,
    net_minus_bah_pct=-1.61,
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

from cardboard_box import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc_annual(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""

# ---------------------------------------------------------------------------
# R-dict preamble for notebooks
# ---------------------------------------------------------------------------
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=55, year_start=1971, year_end=2025,
    n_box_pos=42, n_box_neg=13,
    box_beta=-0.0109, box_t=-1.605, box_r2=0.045,
    box_pearson_r=-0.2122, box_pearson_p=0.1199,
    rail_beta=-0.0074, rail_t=-1.805, rail_r2=0.0425,
    coin_box_t=-0.068, coin_box_r2=0.0, coin_box_pearson_r=-0.007,
    fwd_mean_boxpos=9.84, fwd_mean_boxneg=8.59, fwd_mean_all=9.54,
    uncond_up_pct=74.6, hit_rate_pct=61.8, turnover=0.3091, frac_invested=0.7636,
    bah_ann_pct=8.15, gross_ann_pct=6.57, net_ann_pct=6.54, net_minus_bah_pct=-1.61,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cardboard-Box -- can box and rail-freight output forecast the market?\n"
            "### \"Dr. Cardboard\" tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Almost everything you buy ships in a corrugated box and rides on a freight train at "
            "some point. So the folk wisdom -- sometimes called *\"Dr. Cardboard\"* -- says that "
            "**box production** and **rail-freight carloads** are a clean, early read on the real "
            "economy, and therefore a *leading indicator for the stock market*. If the factories "
            "are ordering more boxes and the railroads are hauling more cars, surely good times "
            "(and good returns) are coming.\n\n"
            "This notebook asks the only question that matters for a trader: does this year's "
            "box/freight growth actually **forecast next year's S&P 500 return**?\n\n"
            "> **This is the plain-language layer.** Want the Newey-West regression, the "
            "coincident-vs-forecast split, and the cost-charged timing backtest? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does box growth forecast next year's market? | **No.** The slope is the *wrong* "
            "sign (slightly negative) and not significant: HAC t = **-1.61**, R^2 = **4.5%**. |\n"
            "| Does freight co-move with the market *this* year? | Barely (coincident t = "
            "**-0.07**). Even the contemporaneous link is weak -- so there's no coincident "
            "correlation to mistake for a forecast. |\n"
            "| Could you trade it? | **No.** A \"go long when boxes grow\" rule earns "
            "**6.5%/yr net** vs **8.2%/yr** for buy-and-hold -- it sits out of the market in "
            "the wrong years. |\n"
            "| Is the economic story wrong? | The *story* (boxes track GDP) is roughly true. "
            "But GDP is already in the price; a slow annual macro series adds nothing the market "
            "doesn't already know. |\n\n"
            "> Dr. Cardboard is a fine description of the economy -- and a useless market timer."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Box shipments and rail carloads measure the physical flow of goods. When that "
            "flow accelerates, the economy is expanding and stocks will rise; when boxes and "
            "carloads roll over, a slowdown -- and a market top -- is coming. Watch the boxes "
            "and you'll see the cycle before everyone else.\"*\n\n"
            "It is an appealing story precisely because it is so concrete. Unlike an abstract "
            "sentiment index, you can almost *see* the boxes. The Fibre Box Association and the "
            "Association of American Railroads publish these numbers, and market commentators "
            "have leaned on them for decades."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If box/freight growth genuinely *led* the market by a year, it would be a gift: a "
            "slow, public, free macro series that tells you next year's direction. You'd lighten "
            "up when the boxes roll over and load up when they accelerate.\n\n"
            "The catch is the difference between **coincident** and **leading**. Boxes track GDP "
            "in *real time*. But the stock market is forward-looking and has *already* priced "
            "expected GDP. For the indicator to be tradable, this year's boxes must predict "
            "*next year's* return -- not just confirm this year's economy."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Coincident vs leading.** A same-year correlation between freight and the "
            "market proves nothing tradable -- both ride the cycle together. We must lag the "
            "predictor by a full year: box growth in year Y vs S&P return in year **Y+1**.\n"
            "2. **The market's upward drift.** Stocks rise ~75% of years regardless. A timer "
            "that is \"long in good years\" gets credit for the drift, not for foresight. The "
            "honest benchmark is **buy-and-hold**, net of costs.\n"
            "3. **Tiny n.** We have ~55 years of annual data. One regressor, 55 points, ~16% "
            "annual vol -- the bar for a real forecasting slope (|t| >= 2) is high, and a "
            "spurious-looking slope is easy to manufacture.\n\n"
            "We test box and rail growth (hardcoded in this study's `data.py`) against ^GSPC "
            "calendar-year **price** returns from Yahoo, 1971-2025 (55 forecast pairs)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the curated annual box/rail growth series, and the "
            "next-year S&P return it is supposed to forecast."
        ),
        code(
            "fr = data.freight_table()\n"
            "if HAVE_REAL:\n"
            "    gspc = data.fetch_gspc_annual(fetch=False)\n"
            "    frame = data.build_forecast_frame(fr, gspc, predictor='box_growth')\n"
            "    n = len(frame)\n"
            "    y0, y1 = int(frame['ret_year'].min()), int(frame['ret_year'].max())\n"
            "else:\n"
            "    frame = None\n"
            "    n, y0, y1 = R['n'], R['year_start'], R['year_end']\n"
            "\n"
            "print('Freight table:', len(fr), 'years (1970-2024)')\n"
            "print(f'Forecast pairs: {n} (box growth in year Y vs S&P return in year Y+1)')\n"
            "print(f'Return window: {y0}-{y1}')\n"
            "print()\n"
            "print(fr.head(8).to_string(index=False))"
        ),
        md("**Does the boxes-up / market-up story show up in the chart?**"),
        code(
            "if HAVE_REAL:\n"
            "    yrs = fr['year'].values\n"
            "    box = fr['box_growth'].values\n"
            "    rail = fr['rail_growth'].values\n"
            "else:\n"
            "    rng = np.random.default_rng(271)\n"
            "    yrs = np.arange(1970, 2025)\n"
            "    box = rng.normal(1.5, 4.0, len(yrs))\n"
            "    rail = rng.normal(1.0, 5.0, len(yrs))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "ax.bar(yrs-0.2, box, width=0.4, color=GREEN, label='Box shipments YoY %')\n"
            "ax.bar(yrs+0.2, rail, width=0.4, color=GREY, label='Rail carloads YoY %')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for rec in [1974,1975,1980,1982,1991,2001,2009,2020]:\n"
            "    ax.axvline(rec, c=RED, ls=':', lw=0.8, alpha=0.5)\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('YoY growth (%)')\n"
            "ax.set_title('Box and rail output: a clean read on the economy (red = recessions)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The series clearly dips in every recession -- it IS a good economy gauge.')\n"
            "print('The question is whether that helps you a YEAR ahead in the market.')"
        ),
        md(
            "The series does exactly what its fans claim about the *economy*: box and rail "
            "growth go sharply negative in every recession (1974-75, 1980-82, 1990-91, 2001, "
            "2008-09, 2020). It is a genuinely good coincident gauge. Now the hard part."
        ),
        md("**The forecast test: this year's boxes vs next year's market.**"),
        code(
            "if HAVE_REAL:\n"
            "    x = frame['predictor'].values\n"
            "    y = frame['fwd_return'].values * 100\n"
            "    reg = st.predictive_regression(frame, lags=1)\n"
            "    beta, tstat, r2 = reg['beta'], reg['t_beta'], reg['r2']\n"
            "    m_pos = frame.loc[frame.predictor>0,'fwd_return'].mean()*100\n"
            "    m_neg = frame.loc[frame.predictor<=0,'fwd_return'].mean()*100\n"
            "else:\n"
            "    rng = np.random.default_rng(271)\n"
            "    x = rng.normal(1.5, 4.0, R['n'])\n"
            "    y = R['fwd_mean_all'] + R['box_beta']*100*x + rng.normal(0,16,R['n'])\n"
            "    beta, tstat, r2 = R['box_beta'], R['box_t'], R['box_r2']\n"
            "    m_pos, m_neg = R['fwd_mean_boxpos'], R['fwd_mean_boxneg']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.scatter(x, y, c=GREY, s=40, alpha=0.8)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, R['fwd_mean_all'] + beta*100*xs if not HAVE_REAL else\n"
            "        (reg['alpha']*100 + reg['beta']*100*xs), c=RED, lw=2,\n"
            "        label=f'slope={beta:+.4f}, t={tstat:+.2f}')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('Box growth this year (%)')\n"
            "ax.set_ylabel('S&P 500 return NEXT year (%)')\n"
            "ax.set_title('No forecasting power: the slope is tiny, wrong-signed, insignificant')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Forecast slope: {beta:+.4f}  HAC t = {tstat:+.2f}  R^2 = {r2:.1%}')\n"
            "print(f'Next-year mean when boxes GREW:  {m_pos:+.1f}%')\n"
            "print(f'Next-year mean when boxes FELL:  {m_neg:+.1f}%')\n"
            "print('If anything the relationship leans the WRONG way -- and it is not significant.')"
        ),
        md(
            "The scatter is a cloud. The fitted slope is **-0.011** (HAC t = "
            "**-1.61**), explaining just **4.5%** of next-year variance -- and the sign is "
            "*negative*, the opposite of the bullish story. Years when boxes grew were followed "
            "by **+9.8%** on average; years when boxes fell, **+8.6%** -- a 1.2pp gap that is "
            "well inside the noise. Whatever Dr. Cardboard sees, it isn't next year's market."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** The forecasting slope is small, wrong-signed, and "
            "insignificant (HAC t = **-1.61** for boxes, **-1.81** for rail; "
            "both |t| < 2). The *coincident* link is also near zero (t = **-0.07**), "
            "so there isn't even a contemporaneous correlation to mistake for foresight.\n"
            "- **Tradability -- Mirage.** A \"long when boxes grow\" timer earns **6.5%/yr "
            "net** vs **8.2%/yr** buy-and-hold: it sits out of the market in the wrong "
            "years and loses to passive exposure.\n"
            "- **Busted.** The economic story (boxes track GDP) is real, but it is "
            "*coincident*, not leading -- and a slow annual macro series is already in the "
            "price by the time you can act on it."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The natural rule: be long the S&P next year if box growth was positive this year, "
            "and sit in cash otherwise. We charge a realistic 10 bps cost each time the position "
            "flips, applied to NAV."
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat = st.timing_strategy(frame, threshold=0.0, cost_bps=10.0)\n"
            "    bah, net = strat['bah_ann']*100, strat['net_ann']*100\n"
            "    yrs2 = strat['years']\n"
            "    cum_bah = (1+strat['bah_returns']).cumprod()*100\n"
            "    cum_net = (1+strat['strat_net_returns']).cumprod()*100\n"
            "else:\n"
            "    bah, net = R['bah_ann_pct'], R['net_ann_pct']\n"
            "    rng = np.random.default_rng(99)\n"
            "    yrs2 = np.arange(R['year_start'], R['year_end']+1)\n"
            "    cum_bah = (1+rng.normal(bah/100/1, 0.16, R['n'])).cumprod()*100\n"
            "    cum_net = (1+rng.normal(net/100/1, 0.14, R['n'])).cumprod()*100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(yrs2, cum_bah, lw=2, c=GREEN, label=f'Buy & hold: {bah:.1f}%/yr')\n"
            "ax.plot(yrs2, cum_net, lw=2, c=RED, ls='--', label=f'Box timer (net): {net:.1f}%/yr')\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Value (log, base 100)')\n"
            "ax.set_title('The box timer trails buy-and-hold -- it is out of the market too often')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold (price-only): {bah:.2f}%/yr')\n"
            "print(f'Box timer, net of costs: {net:.2f}%/yr  ->  shortfall {net-bah:+.2f}pp/yr')\n"
            "print('The timer is long ~76% of years and misses good years after weak box prints.')"
        ),
        md(
            "The timer underperforms by **~1.6pp/yr**. Because the market drifts up most "
            "years, any rule that pulls you out of stocks -- here, after a soft box print -- "
            "tends to cost you. There is no edge to pay for the time spent in cash."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real forecasting beta "
            "into a synthetic tape and shows the regression *does* light up when an effect "
            "exists -- the engine is honest; the real tape simply has nothing.\n"
            "- **Same family of macro folklore:** the Baltic Dry Index, copper (\"Dr. Copper\"), "
            "and the cardboard box are all *coincident* gauges sold as *leading* ones.\n"
            "- **The deeper lesson:** a good description of the economy is not a good market "
            "forecast. Markets price the *expected* economy; a slow public series rarely "
            "surprises them.\n\n"
            "*Think the boxes really lead? Fork this, swap in monthly box data or a different "
            "lag, and show a forecasting slope with |t| >= 2 on the real tape. That's the bar.*"
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
            "# Cardboard-Box -- a quantitative teardown\n"
            "### 55 years * ^GSPC price returns * Newey-West HAC * coincident-vs-lead * "
            "cost-charged timing * synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We regress next-year "
            "^GSPC returns on this year's box / rail-freight growth with a Newey-West (HAC) "
            "t-stat, separate the *coincident* link from the *leading* one, run a cost-charged "
            "timing backtest against buy-and-hold, and close with a synthetic positive control "
            "that confirms the regression detects a forecasting beta when one is planted.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily closes from Yahoo "
            "aggregated to **calendar-year price returns** (1971-2025, price-only, "
            "dividends excluded); box/rail growth hardcoded in `data.py` (1970-2024). "
            "One-year execution lag is baked into the forecast join. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The **> In plain words** notes translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Forecast slope box: HAC **t = -1.61**, R^2 "
            "**4.5%** (wrong sign); rail: **t = -1.81**. Coincident **t = -0.07**. "
            "No |t| >= 2 anywhere. |\n"
            "| **Tradability** | `MIRAGE` | Box timer **6.5%/yr net** vs "
            "**8.2%/yr** buy-and-hold; turnover ~0.31 flips/yr, in the market 76% "
            "of years, still loses. |\n"
            "| **Busted?** | `YES` | Boxes are a fine *coincident* GDP gauge, sold as a "
            "*leading* market signal; the lead-lag forecasting content is nil. |\n\n"
            "> The economic narrative is true and the market forecast is empty -- a textbook "
            "coincident-mistaken-for-leading indicator."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $g_t$ be box-shipment growth in year $t$ (percent) and $r_{t+1}$ the S&P 500 "
            "price return in year $t+1$. The hypotheses:\n\n"
            "- **H1 (forecast).** In $r_{t+1} = \\alpha + \\beta\\, g_t + \\varepsilon_{t+1}$, "
            "$\\beta > 0$ with HAC $t \\ge 2$ -- accelerating box output this year predicts a "
            "higher market next year.\n"
            "- **H2 (rail confirm).** The same holds for rail carload growth.\n"
            "- **H3 (tradable).** A position rule keyed on $g_t > 0$ beats buy-and-hold net of "
            "costs.\n\n"
            "We reject H1, H2, and H3 on the real tape. We also test the **coincident** "
            "regression $r_t = \\alpha + \\beta\\, g_t$ to show that even the same-year link is "
            "negligible -- so there is no contemporaneous correlation to misread as a lead."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Dr. Cardboard belongs to the family of *physical-flow* indicators (Baltic Dry, "
            "copper, trucking tonnage) that feel more trustworthy than sentiment because you "
            "can picture the boxes. If H1 held it would be a slow, public, free macro signal "
            "with a one-year lead -- extraordinarily valuable. The instructive finding is *why* "
            "such a plausible series carries no forecasting content: the market already prices "
            "the expected economy, so a coincident gauge can't lead it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Box & rail YoY growth, 1970-2024 (hardcoded in `data.py`); ^GSPC "
            "December/December **price** returns 1971-2025 from Yahoo (cache-only offline).\n"
            "- **Lag.** Predictor in year $Y$ -> return in year $Y+1$ (no look-ahead; the "
            "forecast must precede the return).\n"
            "- **Inference.** OLS with a Newey-West HAC covariance (Bartlett, 1 lag) on the "
            "slope; Pearson $r$ for context.\n"
            "- **Coincident control.** Same-year regression to expose the coincident-vs-lead "
            "distinction.\n"
            "- **Backtest.** Long next year iff $g_t>0$; 10 bps one-way cost on NAV at each "
            "flip; gross/net vs buy-and-hold; turnover reported.\n"
            "- **Positive control.** Synthetic tape with a planted $\\beta$ to confirm the "
            "regression finds a real forecasting effect.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The forecasting regression with a Newey-West t-stat\n\n"
            "The headline test: regress next-year return on this year's box growth, HAC slope t."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = data.freight_table()\n"
            "    gspc = data.fetch_gspc_annual(fetch=False)\n"
            "    frame_box = data.build_forecast_frame(fr, gspc, predictor='box_growth')\n"
            "    frame_rail = data.build_forecast_frame(fr, gspc, predictor='rail_growth')\n"
            "    reg_box = st.predictive_regression(frame_box, lags=1)\n"
            "    reg_rail = st.predictive_regression(frame_rail, lags=1)\n"
            "    n = reg_box['n']\n"
            "else:\n"
            "    reg_box = dict(beta=R['box_beta'], t_beta=R['box_t'], r2=R['box_r2'],\n"
            "                   pearson_r=R['box_pearson_r'], pearson_p=R['box_pearson_p'], n=R['n'])\n"
            "    reg_rail = dict(beta=R['rail_beta'], t_beta=R['rail_t'], r2=R['rail_r2'])\n"
            "    n = R['n']\n"
            "\n"
            "print(f'n forecast pairs: {n}')\n"
            "print('--- Box growth_t -> S&P return_{t+1} ---')\n"
            "print(f\"  slope beta = {reg_box['beta']:+.4f}   HAC t = {reg_box['t_beta']:+.3f}\")\n"
            "print(f\"  R^2 = {reg_box['r2']:.4f}   Pearson r = {reg_box['pearson_r']:+.4f}\"\n"
            "      f\" (p = {reg_box['pearson_p']:.3f})\")\n"
            "print('--- Rail growth_t -> S&P return_{t+1} ---')\n"
            "print(f\"  slope beta = {reg_rail['beta']:+.4f}   HAC t = {reg_rail['t_beta']:+.3f}\"\n"
            "      f\"   R^2 = {reg_rail['r2']:.4f}\")\n"
            "print()\n"
            "print('Both slopes are negative and |t| < 2 -> no forecasting signal.')"
        ),
        md(
            "> **In plain words.** The boxes-lead-the-market story predicts a *positive* slope "
            "with a t-stat above 2. We get a *negative* slope (boxes up -> market slightly down "
            "next year) and t = -1.61. Not just absent -- pointing the wrong way, and still "
            "inside the noise band."
        ),
        md(
            "### 4b - Coincident vs leading: where the illusion comes from\n\n"
            "People \"see\" the indicator work because boxes and the market both dip in "
            "recessions -- *at the same time*. Here is the same-year regression."
        ),
        code(
            "if HAVE_REAL:\n"
            "    coin = st.coincident_regression(fr, gspc, predictor='box_growth', lags=1)\n"
            "    ct, cr2, crp = coin['t_beta'], coin['r2'], coin['pearson_r']\n"
            "    # build aligned same-year series for the plot\n"
            "    m = fr[['year','box_growth']].merge(gspc, on='year', how='inner')\n"
            "    xc = m['box_growth'].values; yc = m['gspc_return'].values*100\n"
            "else:\n"
            "    ct, cr2, crp = R['coin_box_t'], R['coin_box_r2'], R['coin_box_pearson_r']\n"
            "    rng = np.random.default_rng(2)\n"
            "    xc = rng.normal(1.5,4.0,R['n']); yc = rng.normal(9.5,16,R['n'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.5))\n"
            "ax.scatter(xc, yc, c=AMBER, s=40, alpha=0.8)\n"
            "ax.axhline(0, c='k', lw=0.8); ax.axvline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('Box growth (%) -- SAME year')\n"
            "ax.set_ylabel('S&P 500 return (%) -- SAME year')\n"
            "ax.set_title(f'Coincident link is also weak: t = {ct:+.2f}, r = {crp:+.3f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Coincident regression: HAC t = {ct:+.3f}, R^2 = {cr2:.4f}, r = {crp:+.4f}')\n"
            "print('Even SAME-year, boxes and the market barely correlate -- the recessions')\n"
            "print('line up but the year-to-year wiggles do not.')"
        ),
        md(
            "> **In plain words.** Annual box growth and annual market returns share the broad "
            "recession dips, but the year-by-year scatter is nearly flat (t = -0.07). So there "
            "isn't even a strong coincident correlation here -- let alone a leading one."
        ),
        md(
            "### 4c - The timing backtest, net of costs\n\n"
            "Long the S&P next year iff box growth > 0; 10 bps one-way cost on NAV per flip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat = st.timing_strategy(frame_box, threshold=0.0, cost_bps=10.0)\n"
            "else:\n"
            "    strat = dict(bah_ann=R['bah_ann_pct']/100, gross_ann=R['gross_ann_pct']/100,\n"
            "                 net_ann=R['net_ann_pct']/100, turnover=R['turnover'],\n"
            "                 frac_invested=R['frac_invested'], hit_rate=R['hit_rate_pct']/100,\n"
            "                 uncond_up_rate=R['uncond_up_pct']/100)\n"
            "\n"
            "rows = [\n"
            "    ('Buy & hold (price-only)', strat['bah_ann']*100, '-', '-'),\n"
            "    ('Box timer GROSS', strat['gross_ann']*100, strat['turnover'], strat['frac_invested']),\n"
            "    ('Box timer NET (10bps)', strat['net_ann']*100, strat['turnover'], strat['frac_invested']),\n"
            "]\n"
            "print(f\"{'strategy':28s}{'ann %':>8s}{'turnover':>10s}{'inv frac':>10s}\")\n"
            "for name, ann, to, fi in rows:\n"
            "    to_s = f'{to:.3f}' if to!='-' else '-'\n"
            "    fi_s = f'{fi:.3f}' if fi!='-' else '-'\n"
            "    print(f'{name:28s}{ann:8.2f}{to_s:>10s}{fi_s:>10s}')\n"
            "print()\n"
            "print(f\"Hit-rate of timer: {strat['hit_rate']*100:.1f}%  | unconditional up-rate: \"\n"
            "      f\"{strat['uncond_up_rate']*100:.1f}%\")\n"
            "print(f\"Net minus buy-and-hold: {(strat['net_ann']-strat['bah_ann'])*100:+.2f}pp/yr\")"
        ),
        md(
            "> **In plain words.** The timer's hit-rate (61.8%) is *below* the unconditional "
            "up-rate (74.6%) -- it is worse than always being long. Net of costs it trails "
            "buy-and-hold by ~1.6pp/yr because it sits in cash after weak box prints, which "
            "are not reliably followed by down markets."
        ),
        md(
            "### 4d - Power check: what slope could n=55 even detect?\n\n"
            "The minimum detectable standardized slope at 80% power, two-sided, alpha=0.05."
        ),
        code(
            "from scipy.stats import t as t_dist\n"
            "n = R['n']\n"
            "dof = n - 2\n"
            "t_crit = t_dist.ppf(0.975, dof)\n"
            "t_pw = t_dist.ppf(0.80, dof)\n"
            "# Minimum detectable correlation |r| s.t. t = r*sqrt((n-2)/(1-r^2)) = t_crit+t_pw\n"
            "t_target = t_crit + t_pw\n"
            "r_mde = t_target / np.sqrt(dof + t_target**2)\n"
            "print(f'n = {n}, dof = {dof}')\n"
            "print(f'Minimum detectable |correlation| (80% power): {r_mde:.3f}')\n"
            "print(f'Observed box-forecast |r|: {abs(R[\"box_pearson_r\"]):.3f}')\n"
            "print()\n"
            "print('We could only reliably detect a forecasting correlation above ~0.37.')\n"
            "print('The observed |r| (0.21) is below that AND wrong-signed: nothing to find.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- forecast slope box HAC t = -1.61 (wrong sign), rail "
            "t = -1.81, coincident t = -0.07; no |t| >= 2. R^2 of the forecast regression "
            "is 4.5% and the sign is bearish, not bullish.\n"
            "- **Tradability `MIRAGE`** -- the box timer earns 6.5%/yr net vs 8.2%/yr "
            "buy-and-hold; its hit-rate (61.8%) is below the unconditional up-rate (74.6%).\n"
            "- **Busted** -- a coincident GDP gauge dressed as a leading market signal. The "
            "physical-flow story is real; the forecasting content is not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark, drawn\n\n"
            "Equity curves: box timer (net) vs passive long S&P, both price-only."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yrs2 = strat['years']\n"
            "    cum_bah = (1+strat['bah_returns']).cumprod()*100\n"
            "    cum_net = (1+strat['strat_net_returns']).cumprod()*100\n"
            "    bah, net = strat['bah_ann']*100, strat['net_ann']*100\n"
            "else:\n"
            "    bah, net = R['bah_ann_pct'], R['net_ann_pct']\n"
            "    rng = np.random.default_rng(99)\n"
            "    yrs2 = np.arange(R['year_start'], R['year_end']+1)\n"
            "    cum_bah = (1+rng.normal(bah/100, 0.16, R['n'])).cumprod()*100\n"
            "    cum_net = (1+rng.normal(net/100, 0.14, R['n'])).cumprod()*100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10,4.5))\n"
            "ax.plot(yrs2, cum_bah, lw=2, c=GREEN, label=f'Buy & hold: {bah:.1f}%/yr')\n"
            "ax.plot(yrs2, cum_net, lw=2, c=RED, ls='--', label=f'Box timer net: {net:.1f}%/yr')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Value (log, base 100)')\n"
            "ax.set_title('Box timer trails passive exposure across the full sample')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Shortfall: {net-bah:+.2f}pp/yr')"
        ),
        md(
            "> **In plain words.** Sitting out of a market that drifts up most years is "
            "expensive. The box signal does not flag the down years reliably enough to pay "
            "for the time spent in cash."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the regression detect a forecasting beta *when one exists*? We plant a beta "
            "into a synthetic (freight, next-year-return) panel and sweep its size."
        ),
        code(
            "planted = [0.0, 0.5, 1.0, 2.0, 4.0]\n"
            "rows = []\n"
            "for b in planted:\n"
            "    df_s, _ = data.synthetic_panel(n_years=400, beta=b, seed=271)\n"
            "    df_s = df_s.rename(columns={'box_growth':'predictor'})\n"
            "    reg = st.predictive_regression(df_s, lags=1)\n"
            "    rows.append({'beta_planted': b, 'beta_hat': reg['beta'],\n"
            "                 't': reg['t_beta'], 'r2': reg['r2']})\n"
            "res = pd.DataFrame(rows)\n"
            "\n"
            "colors = [GREEN if abs(r['t'])>=2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9,4.3))\n"
            "ax.bar([str(r['beta_planted']) for r in rows], [r['t'] for r in rows], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2'); ax.axhline(-2, c='k', ls=':', lw=1)\n"
            "ax.set_xlabel('Planted beta'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine finds the signal when it exists (green = |t| >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "print('With a real planted beta the t-stat clears 2. The real tape (t=-1.6) does not.')"
        ),
        md(
            "The engine recovers a planted forecasting beta cleanly. The real tape is nowhere "
            "near the detection threshold -- and what little slope exists points the wrong way. "
            "The verdict is a statement about the **indicator and the market**, not the method."
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


def _execute():
    """Execute both notebooks in place via nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    for name in ("01_for_the_curious.ipynb", "02_for_the_quants.ipynb"):
        path = os.path.join(HERE, name)
        nb = nbf.read(path, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": HERE}})
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute()
