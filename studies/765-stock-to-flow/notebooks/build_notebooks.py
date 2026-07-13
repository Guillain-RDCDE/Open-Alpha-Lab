"""Generate the two narrative notebooks for Study 765 (Stock-to-Flow).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached BTC-USD tape under
../_cache/ and reconstruct the S2F curve from the hardcoded issuance schedule (no network either
way), else quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic
positive control runs anywhere with no network.
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
# 2026-06-30; S2F reconstructed from the issuance schedule; fit frozen at publication 2019-03-22).
R = dict(
    btc_lo="2014-09-17", btc_hi="2026-06-30", btc_n=4305, fp="74d40e852645",
    sf_now=122, supply_now_m=20.05, pub="2019-03-22",
    # the fit that flatters
    steel_r2=0.8866, steel_b=2.458, steel_n=142,
    r2_sf=0.8803, r2_time=0.8758, corr_sf_time=0.9643,
    # frozen-at-publication coefficients + in/out-of-sample fit
    a=-0.8711, b=2.7661, n_train=1648, n_oos=2657,
    r2_in=0.7050, r2_oos=0.2132, rmse_in=0.7390, rmse_oos=0.7702,
    # date -> (actual price, frozen-model price, actual/model)
    pred={
        "2021-11-30": (57005, 30857, 1.85),
        "2022-11-30": (17169, 32384, 0.53),
        "2024-12-31": (93429, 239071, 0.39),
        "2026-06-30": (58559, 247369, 0.24),
    },
    # horizon -> (slope, HAC t OOS, HAC t full-sample)
    fwd={
        30: (-0.015, -0.62, -1.48),
        90: (-0.121, -1.51, -2.50),
        180: (-0.435, -1.68, -2.81),
        365: (-0.187, -0.45, -1.64),
    },
    # timer vs buy-and-hold — out-of-sample (2019-03 -> 2026-06)
    oos_expo=59, oos_switches=9, oos_years=7.3,
    oos_net=328, oos_gross=332, oos_net_sharpe=0.71, oos_bh=1355, oos_bh_sharpe=0.91,
    oos_plac_mean=87, oos_plac_p95=409,
    # timer vs buy-and-hold — post-2021 "broke" window (2021-11 -> 2026-06)
    p21_expo=78, p21_years=4.7, p21_net=7, p21_bh=-4, p21_net_sharpe=0.25, p21_bh_sharpe=0.24,
    p21_plac_mean=-39, p21_plac_p95=17,
    # synthetic control
    syn_null_mean=0.17, syn_null_sd=1.29, syn_null_fire=5, syn_planted=-5.96,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Holds_out--of--sample%3F: Busted](https://img.shields.io/badge/Holds_out--of--sample%3F-Busted-8b949e?style=flat-square)\n\n"
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

from stock_to_flow import data, strategy as st

SF = data.supply_flow_daily()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    BTC = data.load_btc()
    DF = data.join_price_sf(BTC)
    OOS = st.oos_fit_stats(DF, data.PUBLICATION_DATE)
    A, B = OOS["a"], OOS["b"]
    MODEL = st.model_price(DF, A, B)
    RESID = st.valuation_residual(DF, A, B)
else:
    BTC = DF = OOS = A = B = MODEL = RESID = None
print("real cache present:", HAVE_REAL, "| S2F curve points:", len(SF),
      "| SF today:", round(float(SF['sf'].iloc[-1] if not HAVE_REAL else DF['sf'].iloc[-1]), 1))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# A model that fit Bitcoin at 95% — and knew nothing 📐\n"
            "### Stock-to-Flow — the most famous Bitcoin valuation model, and the most famously "
            "busted\n\n"
            + BADGES +
            "In 2019 an anonymous analyst called **PlanB** published a chart that went viral. He "
            "measured Bitcoin's *scarcity* — the ratio of coins that already exist to the trickle "
            "of new ones mined each year (\"stock-to-flow\") — and showed it lined up almost "
            "perfectly with Bitcoin's price. The fit was a jaw-dropping **95%**. The model "
            "predicted a six-figure Bitcoin, and for a while it looked like prophecy.\n\n"
            "Then 2022 happened. Bitcoin crashed to $17,000 while the model insisted it should be "
            "worth *six figures*. The prophecy became a punchline. This notebook asks the "
            "question everyone skipped in 2019: **was there ever anything there?**\n\n"
            "> 📓 **Plain-language layer.** Want the spurious-regression proof and the HAC "
            "*t*-stats? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** The one honest test of any predictive model is to freeze it at "
            "the moment it was published and see what it did *next*. That's what we do here. House "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the model really fit at ~95%? | **Yes — and that's the trap.** On the full "
            f"history it fits at R² ≈ **{R['steel_r2']:.2f}**. But Bitcoin's stock-to-flow is "
            f"basically a *calendar* (it's set by a fixed schedule), and it's **"
            f"{R['corr_sf_time']*100:.0f}%** correlated with time itself. Fit price against a "
            f"plain clock and you get R² = **{R['r2_time']:.2f}** — the same thing. |\n"
            f"| What happened after it was published? | Freeze the model in March 2019 and let it "
            f"run: its fit **collapses** from {R['r2_in']*100:.0f}% to **{R['r2_oos']*100:.0f}%**. "
            f"By 2026 it predicts BTC at **${R['pred']['2026-06-30'][1]:,}** — the real price is "
            f"**${R['pred']['2026-06-30'][0]:,}**, four times lower. |\n"
            f"| Could you have traded it? | \"Buy when Bitcoin is cheap vs the model\" returned "
            f"**+{R['oos_net']}%** — while just **holding** returned **+{R['oos_bh']:,}%**. The "
            "strategy lost to doing nothing, and did no better than buying on **random** days. |\n"
            "| So was there ever a signal? | **No.** A beautiful curve fit to a clock. It "
            "described the past and predicted nothing. |\n\n"
            "> Fitting a rising line to another rising line is the oldest illusion in statistics. "
            "The 95% was real. The prophecy was not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Bitcoin is priced by its scarcity. Take the stock of existing coins, divide by "
            "the annual flow of new ones — that's the stock-to-flow ratio, and it doubles at every "
            "'halving.' Plot Bitcoin's market value against it and you get a straight line on a "
            "log-log chart with 95% R². Since the future supply schedule is known years in "
            "advance, the model tells you what Bitcoin will be worth: six figures after the 2020 "
            "halving, ~$288,000 after 2024.\"*\n\n"
            "It's a genuinely seductive idea: a valuation you can compute from Bitcoin's own "
            "**deterministic issuance rules**, no market data required. Gold has a high "
            "stock-to-flow; so does silver; Bitcoin's climbs toward gold's with every halving. If "
            "scarcity really set the price, this would be one of the few honest crystal balls in "
            "finance."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If S2F were real it would be extraordinary — a risk asset whose fair value you could "
            "read years ahead off a schedule carved into the protocol. It became the reference "
            "model of the 2020–2021 bull market: quoted in research notes, printed on dashboards, "
            "used to justify \"$100k is conservative.\" The stakes are simple. If the model works, "
            "scarcity is destiny and you buy every dip below the line. If it doesn't, it's a "
            "textbook lesson in how a spectacular-looking fit can be *manufactured* by two numbers "
            "that both happen to go up over time — and a cautionary tale about every model sold on "
            "its R²."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Rebuild the model honestly.** Reconstruct stock-to-flow straight from Bitcoin's "
            "halving schedule (this part is exact — issuance is consensus law), fit PlanB's "
            "log-log line, and reproduce the famous ~95%.\n"
            "- **Is the fit a clock?** Race the stock-to-flow fit against fitting price to *plain "
            "calendar time*. If a clock does just as well, the \"scarcity\" story adds nothing.\n"
            "- **Freeze it at publication.** Fit the model using *only* data available in March "
            "2019, then see how its predictions held up over the next seven years — the crash, "
            "the recoveries, today.\n"
            "- **Try to trade it.** Buy Bitcoin whenever it's below the model line (\"cheap\"), "
            "sit in cash otherwise. Beat buy-and-hold? Beat *random* buy dates?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the famous chart.** Here is Bitcoin's price against the model line built "
            "from stock-to-flow. It really does look like a law of nature."
        ),
        code(
            "if HAVE_REAL:\n"
            "    price = DF['price']; model = MODEL\n"
            "    dfp = DF\n"
            "else:\n"
            "    price = model = dfp = None\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
            "if HAVE_REAL:\n"
            "    ax.semilogy(price.index, price.values, color=GREY, lw=1.2, label='BTC price (actual)')\n"
            "    ax.semilogy(model.index, model.values, color=RED, lw=2.0, label='S2F model (frozen at 2019 publication)')\n"
            "    ax.axvline(pd.Timestamp(R['pub']), color='k', ls=':', lw=1.2)\n"
            "    ax.text(pd.Timestamp(R['pub']), price.min()*2, ' published\\n Mar 2019', fontsize=9)\n"
            "else:\n"
            "    ax.text(.5, .5, '(cached tape absent — see docs/results.md)', ha='center')\n"
            "ax.set_ylabel('USD (log scale)')\n"
            "ax.set_title('Stock-to-Flow: a model frozen in 2019, and what Bitcoin did next')\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()"
        ),
        md(
            f"Left of the dotted line (before publication) the red model hugs the price — that's "
            f"the **{R['r2_in']*100:.0f}%** in-sample fit. Right of it, watch the red line tear "
            f"away: as stock-to-flow doubled at the 2024 halving, the model's predicted price "
            f"exploded past **${R['pred']['2026-06-30'][1]:,}** while Bitcoin sat near "
            f"**${R['pred']['2026-06-30'][0]:,}**. **Now — why was the fit so good, and why did it "
            "mean nothing?**"
        ),
        code(
            "# Race the S2F fit against fitting price to a plain CLOCK\n"
            "if HAVE_REAL:\n"
            "    race = st.spurious_trend_race(DF)\n"
            "    r2_sf, r2_time, corr = race['r2_sf'], race['r2_time'], race['corr_sf_time']\n"
            "else:\n"
            "    r2_sf, r2_time, corr = R['r2_sf'], R['r2_time'], R['corr_sf_time']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['price vs\\nstock-to-flow', 'price vs\\nplain calendar'], [r2_sf, r2_time],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([r2_sf, r2_time]): a1.annotate(f'{v:.3f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylim(0, 1); a1.set_ylabel('R² (goodness of fit)')\n"
            "a1.set_title('A clock fits just as well')\n"
            "if HAVE_REAL:\n"
            "    a2.scatter(np.arange(len(DF)), np.log(DF['sf'].values), s=2, color=RED)\n"
            "    a2.set_xlabel('days'); a2.set_ylabel('ln(stock-to-flow)')\n"
            "    a2.set_title(f'Stock-to-flow IS a clock (corr with time = {corr:.2f})')\n"
            "else:\n"
            "    a2.text(.5, .5, f'corr(ln SF, time) = {corr:.2f}', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'R2 price~S2F = {r2_sf:.3f}   R2 price~time = {r2_time:.3f}   corr(lnSF,time) = {corr:.3f}')"
        ),
        md(
            f"There's the whole trick. Stock-to-flow is a staircase set by a fixed schedule — it's "
            f"**{R['corr_sf_time']*100:.0f}%** the same thing as a calendar. So \"fitting price to "
            f"stock-to-flow\" (R² {R['r2_sf']:.2f}) is almost identical to \"fitting price to the "
            f"passage of time\" (R² {R['r2_time']:.2f}). Any asset that trended up over these years "
            "would produce the same beautiful chart. The scarcity story adds essentially nothing "
            "the clock didn't already give you. **So did buying below the line at least pay?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tb = st.timer_backtest(DF, RESID, threshold=0.0, cost_bps=10.0, window_start=data.PUBLICATION_DATE)\n"
            "    st_net, bh = tb['net_total_pct'], tb['bh_total_pct']\n"
            "    st_shp, bh_shp = tb['net_sharpe'], tb['bh_sharpe']\n"
            "    pl = st.random_placebo(DF, tb['exposure_pct'], cost_bps=10.0, window_start=data.PUBLICATION_DATE, n_draws=2000)\n"
            "    plac = pl['p95_total_pct']\n"
            "else:\n"
            "    st_net, bh = R['oos_net'], R['oos_bh']\n"
            "    st_shp, bh_shp = R['oos_net_sharpe'], R['oos_bh_sharpe']\n"
            "    plac = R['oos_plac_p95']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "bars = ax.bar(['buy when\\ncheap vs model', 'just\\nhold', 'random buy dates\\n(95th pct)'],\n"
            "              [st_net, bh, plac], color=[RED, GREEN, GREY], width=.6)\n"
            "for i, v in enumerate([st_net, bh, plac]): ax.annotate(f'{v:+,.0f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('total return, out-of-sample (2019-03 → 2026-06)')\n"
            "ax.set_title('The model strategy lost to holding — and to luck')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer {st_net:+.0f}% (Sharpe {st_shp:.2f}) vs hold {bh:+.0f}% (Sharpe {bh_shp:.2f}); random p95 {plac:+.0f}%')"
        ),
        md(
            f"Not close. Buying Bitcoin whenever it was \"cheap vs the model\" made "
            f"**+{R['oos_net']}%** — versus **+{R['oos_bh']:,}%** for just holding, and it was even "
            f"*worse* on a risk-adjusted basis (Sharpe {R['oos_net_sharpe']} vs "
            f"{R['oos_bh_sharpe']}). Worse still, that +{R['oos_net']}% doesn't even beat picking "
            f"buy dates at **random** at the same frequency (whose luckiest 5% reach "
            f"**+{R['oos_plac_p95']}%**). The model's valuation gap carried no usable information."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The famous 95% fit is a *spurious regression*: stock-to-flow is "
            "96% a clock, and a plain time-trend fits equally well. The valuation gap predicts "
            "nothing out-of-sample.\n"
            "- **Tradability — Mirage.** Buying \"cheap vs model\" lost to buy-and-hold on both "
            "return and Sharpe, and didn't beat random buy dates.\n"
            "- **Holds out-of-sample? — Busted.** Frozen at publication, the model's fit collapsed "
            "and it over-predicted price four-fold by 2026, right through a crash it said couldn't "
            "happen."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The lesson outlives the model.** \"High R²\" on two trending series is the single "
            "most common way to fool yourself in markets. The fix is boring and non-negotiable: "
            "freeze the model at publication and score the future it never saw.\n"
            "- **What would change our mind:** a version of S2F whose *out-of-sample* residual "
            "predicted returns with |t| ≥ 2 — we looked, at four horizons, and found nothing "
            "close.\n"
            "- **Sibling studies:** [323-btc-halving](../../323-btc-halving/) (the halving as a "
            "*calendar event*, not a valuation), [293-mvrv-ratio](../../293-mvrv-ratio/) (a "
            "market-derived on-chain valuation ratio), "
            "[663-hash-ribbons](../../663-hash-ribbons/) (a miner-capitulation buy signal).\n\n"
            "*Think a different fitting window, a market-cap model, or the 365-day-smoothed S2F "
            "rescues it? Fork the repo — the reconstruction is exact and the out-of-sample test is "
            "the only thing that counts.*"
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
            "# Stock-to-Flow — a quantitative teardown 🔬\n"
            "### The spurious-regression race · the frozen out-of-sample R² collapse · a HAC "
            "residual→return regression · a timer vs buy-and-hold · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — `ln(price) = a + b·ln(SF)` with in-sample R² ≈ 0.95 — is tested three ways: "
            "**(1)** is the fit real or a spurious regression of two non-stationary trending "
            "series; **(2)** does it survive frozen at its publication date; **(3)** is the "
            "valuation residual a tradable signal net of costs. The S2F curve is **reconstructed "
            "from Bitcoin's exact issuance schedule** — consensus law, not a proxy.\n\n"
            "> ⚠️ **Data note.** S2F reconstructed from the halving schedule (stock exact at every "
            "halving, flow = reward × 144 blocks/day × 365). BTC-USD daily close, yfinance, "
            + R["btc_lo"] + " → " + R["btc_hi"] + " (fingerprint `" + R["fp"] + "`). "
            "Price-only == total-return for BTC. No survivorship on the Signal axis (single-asset "
            "index); BTC's single-survivor character is named on Tradability. Methods in "
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
            f"| **Signal** | `NONE` | out-of-sample residual→return HAC *t* peaks at "
            f"**{R['fwd'][180][1]:.2f}** (180d); full-sample R² ≈ {R['r2_sf']:.2f} matches a "
            f"time-trend's {R['r2_time']:.2f} (corr {R['corr_sf_time']:.2f}) — spurious |\n"
            f"| **Tradability** | `MIRAGE` | timer **+{R['oos_net']}%** vs B&H "
            f"**+{R['oos_bh']:,}%** OOS (Sharpe {R['oos_net_sharpe']} vs {R['oos_bh_sharpe']}), "
            f"below the random-timing p95 (+{R['oos_plac_p95']}%) |\n"
            f"| **Holds out-of-sample?** | `BUSTED` | frozen-at-pub R² "
            f"{R['r2_in']:.3f} → {R['r2_oos']:.3f}; model/actual = "
            f"{R['pred']['2026-06-30'][2]:.2f} by 2026 |\n\n"
            "> 💡 In plain words: the model *describes* the past because both its inputs trend "
            "with time; it *predicts* nothing because a trend is not a mechanism."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be BTC price and $SF_t = \\text{stock}_t / \\text{flow}_t$ the "
            "stock-to-flow ratio, both observed daily. PlanB's model:\n\n"
            "$$\\ln P_t = a + b\\,\\ln SF_t + \\varepsilon_t, \\qquad R^2 \\approx 0.95.$$\n\n"
            "- **H₁ (the model is real).** $b$ is a structural elasticity of price to scarcity, "
            "and $\\varepsilon_t$ is stationary noise around a true relationship — so the fitted "
            "line is a *valuation*, usable out-of-sample.\n"
            "- **H₂ (tradable).** The residual $\\varepsilon_t$ mean-reverts: when price is below "
            "the line ($\\varepsilon_t < 0$, \"cheap\"), forward returns are positive.\n\n"
            "The steelman fit uses PlanB's own dependent variable (market cap, monthly): on our "
            f"sample R² = **{R['steel_r2']:.4f}** (n = {R['steel_n']} months), b = {R['steel_b']}. "
            "The whole question is whether that number means anything — because both $\\ln P_t$ "
            "and $\\ln SF_t$ are **non-stationary and trending**, the exact setting where OLS R² "
            "and *t* are known to be spurious (Granger–Newbold 1974)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the spurious-regression problem\n\n"
            "$\\ln SF_t$ is a near-deterministic function of the block height: it is flat within "
            "a halving epoch and doubles at each halving, so it rises monotonically with calendar "
            "time. Regressing one trending (integrated) series on another manufactures high R² "
            "and large *t* even when the two are causally unrelated — the classic spurious "
            "regression. The clean diagnostic: **race the S2F fit against a fit on time itself.** "
            "If $\\ln P_t \\sim \\text{time}$ fits as well as $\\ln P_t \\sim \\ln SF_t$, then S2F "
            "is contributing nothing but a clock, and the reported R² certifies scarcity of "
            "*nothing*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **S2F curve.** Reconstructed from the issuance schedule; stock exact at each "
            "halving (10.5M/15.75M/18.375M/19.6875M), flow = reward × 144 × 365, SF doubling per "
            f"halving; SF today ≈ {R['sf_now']}.\n"
            f"- **Tape.** BTC-USD daily close {R['btc_lo']} → {R['btc_hi']} ({R['btc_n']:,} rows), "
            "as-of 2026-06-30 (last complete month). Note: yfinance BTC-USD begins 2014-09, so our "
            "in-sample window is shorter than PlanB's 2009-start fit — named.\n"
            f"- **Spurious race.** R²(price~lnSF) vs R²(price~time), and corr(lnSF, time).\n"
            f"- **Honest OOS.** Fit on data ≤ **{R['pub']}** (publication; {R['n_train']:,} days), "
            f"freeze $a,b$, score the untouched {R['n_oos']:,} days after.\n"
            "- **Signal.** Overlapping forward returns (30/90/180/365d) on the lagged residual, "
            "Newey-West *t* at lag 1.5× horizon; **REAL needs a negative slope with |t| ≥ 2 "
            "out-of-sample.**\n"
            "- **Execution.** Residual known at *t*'s close → position from *t+1* (one-day lag); "
            "10 bps one-way × NAV per switch; gross AND net; matched-exposure random-timing "
            "placebo.\n"
            "- **Control.** Synthetic world with an exogenous stationary valuation gap and a "
            "tunable planted mean-reversion coefficient; the null must read ~0 across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The fit is a clock — the spurious-regression race\n\n"
            "Full-sample OLS, daily. If scarcity mattered beyond the trend, the S2F fit would "
            "clearly beat a naive time-trend. It does not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    race = st.spurious_trend_race(DF)\n"
            "    r2_sf, r2_time, corr = race['r2_sf'], race['r2_time'], race['corr_sf_time']\n"
            "    y = np.log(DF['price'].values); x = np.log(DF['sf'].values); t = np.arange(len(DF))\n"
            "else:\n"
            "    r2_sf, r2_time, corr = R['r2_sf'], R['r2_time'], R['corr_sf_time']\n"
            "    y = x = t = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(['price ~ ln(SF)', 'price ~ time'], [r2_sf, r2_time], color=[RED, GREY], width=.5)\n"
            "for i, v in enumerate([r2_sf, r2_time]): a1.annotate(f'{v:.3f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylim(0, 1); a1.set_ylabel('R²'); a1.set_title('Spurious race: dead heat')\n"
            "if HAVE_REAL:\n"
            "    a2.scatter(t, x, s=2, color=RED, label='ln(SF)')\n"
            "    a2.set_xlabel('trading day #'); a2.set_ylabel('ln(SF)')\n"
            "    a2.set_title(f'ln(SF) vs time — corr = {corr:.3f}')\n"
            "else:\n"
            "    a2.text(.5,.5,f'corr(lnSF,time)={corr:.3f}',ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'R2(price~lnSF)={r2_sf:.4f}  R2(price~time)={r2_time:.4f}  corr(lnSF,time)={corr:.4f}')"
        ),
        md(
            f"> 💡 In plain words: R² {R['r2_sf']:.3f} vs {R['r2_time']:.3f} is a statistical dead "
            f"heat, and ln(SF) is {R['corr_sf_time']*100:.0f}% correlated with the day counter. "
            "The model's explanatory power is the trend, relabelled \"scarcity.\" Under "
            "non-stationarity the reported R² and *t* are exactly the quantities Granger–Newbold "
            "warned are meaningless."
        ),
        md(
            "### 4b · Freeze it at publication — the out-of-sample collapse\n\n"
            f"Fit on data ≤ {R['pub']}, freeze $a = {R['a']}, b = {R['b']}$, score forward with "
            "the frozen line (so OOS R² can go negative)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.oos_fit_stats(DF, data.PUBLICATION_DATE)\n"
            "    r2_in, r2_oos = o['r2_in'], o['r2_oos']\n"
            "    pred = {d: (float(DF.loc[DF.index<=pd.Timestamp(d),'price'].iloc[-1]),\n"
            "                float(MODEL.loc[MODEL.index<=pd.Timestamp(d)].iloc[-1]))\n"
            "            for d in R['pred']}\n"
            "else:\n"
            "    r2_in, r2_oos = R['r2_in'], R['r2_oos']\n"
            "    pred = {d: (v[0], v[1]) for d, v in R['pred'].items()}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar(['in-sample\\n(→2019-03)', 'out-of-sample\\n(2019-03→)'], [r2_in, r2_oos], color=[GREY, RED], width=.5)\n"
            "for i, v in enumerate([r2_in, r2_oos]): a1.annotate(f'{v:.3f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('R²'); a1.set_title('Fit collapses out-of-sample')\n"
            "ds = list(pred.keys()); act = [pred[d][0] for d in ds]; mod = [pred[d][1] for d in ds]\n"
            "xp = np.arange(len(ds)); w = 0.38\n"
            "a2.bar(xp - w/2, act, width=w, color=GREEN, label='actual')\n"
            "a2.bar(xp + w/2, mod, width=w, color=RED, label='frozen model')\n"
            "a2.set_yscale('log'); a2.set_xticks(xp); a2.set_xticklabels([d[:7] for d in ds], rotation=30, fontsize=8)\n"
            "a2.set_ylabel('USD (log)'); a2.set_title('Model vs reality: 4x too high by 2026'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for d in ds: print(f'{d}: actual ${pred[d][0]:,.0f}  model ${pred[d][1]:,.0f}  ratio {pred[d][0]/pred[d][1]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: in-sample R² {R['r2_in']:.2f} → out-of-sample "
            f"{R['r2_oos']:.2f}. The frozen model **under**-shot into the 2021 top, then as SF "
            f"doubled at the 2024 halving its prediction rocketed to "
            f"${R['pred']['2026-06-30'][1]:,} while price sat at ${R['pred']['2026-06-30'][0]:,} "
            "— a factor of four. A model that predicted a six-figure *floor* straight through the "
            "2022 crash to $17k is not a valuation; it's a curve fit that ran out of past."
        ),
        md(
            "### 4c · Is the residual tradable? — HAC regression + timer\n\n"
            "Left: forward BTC returns regressed on the lagged valuation residual, Newey-West *t*, "
            "**out-of-sample** (frozen coefficients — no look-ahead). Right: the "
            "long-when-cheap timer vs buy-and-hold, net of costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prO = st.predictive_regression(DF, RESID, window_start=data.PUBLICATION_DATE)\n"
            "    hz = list(prO.index); ts = list(prO['hac_t'])\n"
            "    tb = st.timer_backtest(DF, RESID, threshold=0.0, cost_bps=10.0, window_start=data.PUBLICATION_DATE)\n"
            "    st_net, bh, plac = tb['net_total_pct'], tb['bh_total_pct'], None\n"
            "    pl = st.random_placebo(DF, tb['exposure_pct'], cost_bps=10.0, window_start=data.PUBLICATION_DATE, n_draws=2000)\n"
            "    plac = pl['p95_total_pct']\n"
            "else:\n"
            "    hz = [30, 90, 180, 365]; ts = [R['fwd'][h][1] for h in hz]\n"
            "    st_net, bh, plac = R['oos_net'], R['oos_bh'], R['oos_plac_p95']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "x = np.arange(len(hz))\n"
            "a1.bar(x, ts, color=[RED if abs(v) >= 2 else AMBER for v in ts], width=.55)\n"
            "a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'+{h}d' for h in hz]); a1.set_ylabel('HAC t (out-of-sample)')\n"
            "a1.set_title('Residual → return: no horizon clears |t|=2')\n"
            "a2.bar(['timer', 'buy & hold', 'random p95'], [st_net, bh, plac], color=[RED, GREEN, GREY], width=.6)\n"
            "for i, v in enumerate([st_net, bh, plac]): a2.annotate(f'{v:+,.0f}%', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('net total return, OOS'); a2.set_title('Timer loses to holding and to luck')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, v in zip(hz, ts): print(f'+{h}d HAC t (OOS) = {v:+.2f}')\n"
            "print(f'timer {st_net:+.0f}% vs hold {bh:+.0f}% vs random-p95 {plac:+.0f}%')"
        ),
        md(
            f"> 💡 In plain words: out-of-sample, the best HAC *t* is **{R['fwd'][180][1]:.2f}** "
            f"(180d) — the slope has the \"buy cheap\" sign, but nowhere near significant. (The "
            f"full-sample version *does* cross 2 at 90/180d — but the residual is defined against "
            f"a line fit on that same data, so that's the fit grading its own homework, not "
            f"evidence.) And the timer earns **+{R['oos_net']}%** where holding earns "
            f"**+{R['oos_bh']:,}%** and even random buy dates' luckiest 5% reach "
            f"**+{R['oos_plac_p95']}%**. No signal, no edge."
        ),
        md(
            "### 4d · The post-2021 \"broke\" window, on its own\n\n"
            "The model became a meme *after* 2021, so test the rule only there (2021-11 → today) "
            "— the window where believers said it would still call the bottom."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tb2 = st.timer_backtest(DF, RESID, threshold=0.0, cost_bps=10.0, window_start='2021-11-01')\n"
            "    net, bh2, expo = tb2['net_total_pct'], tb2['bh_total_pct'], tb2['exposure_pct']\n"
            "    ns, bs = tb2['net_sharpe'], tb2['bh_sharpe']\n"
            "else:\n"
            "    net, bh2, expo = R['p21_net'], R['p21_bh'], R['p21_expo']\n"
            "    ns, bs = R['p21_net_sharpe'], R['p21_bh_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "b = ax.bar(['S2F timer', 'buy & hold'], [net, bh2], color=[RED, GREEN], width=.5)\n"
            "for i, v in enumerate([net, bh2]): ax.annotate(f'{v:+.0f}%', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net total return, 2021-11 → 2026-06')\n"
            "ax.set_title(f'Post-2021: a wash (exposure {expo:.0f}%, Sharpe {ns:.2f} vs {bs:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post-2021: timer {net:+.0f}% (Sharpe {ns:.2f}) vs hold {bh2:+.0f}% (Sharpe {bs:.2f}), exposure {expo:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: {R['p21_net']:+d}% vs {R['p21_bh']:+d}% — a coin-flip. Once the "
            "model line explodes above the price, the residual is negative essentially every day, "
            f"so the rule reads \"always cheap, always long\" (exposure {R['p21_expo']}%) and just "
            "*becomes* buy-and-hold. The signal doesn't fail dramatically; it dissolves into the "
            "thing it was supposed to beat."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic world: an exogenous *stationary* valuation gap and an explicitly-planted "
            "mean-reversion coefficient `beta`. The detector regresses forward returns on the gap. "
            "The null (`beta = 0`, gap carries no return information) is checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(data.synthetic_world(beta=0.0, seed=765 + s)) for s in range(20)])\n"
            "planted = st.synthetic_detect(data.synthetic_world(beta=0.03, seed=765))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40, label='null worlds (beta=0), 20 seeds')\n"
            "ax.scatter([1], [planted], color=RED, s=90, zorder=5, label='planted beta=0.03')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (gap → forward return)')\n"
            "ax.set_title('Control: detector unbiased under the null, recovers a real effect')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20  |  planted t = {planted:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the detector recovers a planted valuation signal cleanly "
            f"(t = {R['syn_planted']:.2f}) and is unbiased under the null (mean "
            f"{R['syn_null_mean']:+.2f}). But note the null's *spread*: {R['syn_null_fire']}/20 "
            "seeds cross |t| = 2 by chance, because overlapping windows and a persistent regressor "
            "inflate the tails — the very mechanism that makes the real full-sample *t*'s "
            "untrustworthy. It is exactly why we bar the Signal stamp on anything short of a clean "
            "**out-of-sample** |t| ≥ 2. *(A faithful-engine / power check only — never cited in "
            "support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the full-sample R² ≈ {R['r2_sf']:.2f} is a spurious "
            f"regression: ln(SF) is {R['corr_sf_time']*100:.0f}% correlated with time and a plain "
            f"time-trend fits equally well ({R['r2_time']:.2f}). Out-of-sample the residual "
            f"predicts forward returns at no horizon with |t| ≥ 2 (best {R['fwd'][180][1]:.2f}, "
            "180d).\n"
            f"- **Tradability `MIRAGE`** — the long-when-cheap timer returns "
            f"+{R['oos_net']}% net vs buy-and-hold's +{R['oos_bh']:,}% out-of-sample (Sharpe "
            f"{R['oos_net_sharpe']} vs {R['oos_bh_sharpe']}), below the matched-exposure "
            f"random-timing p95 (+{R['oos_plac_p95']}%); post-2021 it collapses to buy-and-hold "
            f"(+{R['p21_net']}% vs {R['p21_bh']}%).\n"
            f"- **\"Holds out-of-sample?\" `BUSTED`** — coefficients frozen at publication, R² "
            f"falls {R['r2_in']:.3f} → {R['r2_oos']:.3f}; the model implied a six-figure BTC floor "
            f"through the 2022 crash to $17k and stands {R['pred']['2026-06-30'][2]:.2f}× of the "
            f"tape by 2026 (${R['pred']['2026-06-30'][1]:,} model vs "
            f"${R['pred']['2026-06-30'][0]:,} actual)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The cointegration objection, and why it doesn't save it.** A defender might say "
            "price and SF are *cointegrated* rather than spuriously correlated. But cointegration "
            "is a testable claim about a *stationary* residual that mean-reverts to a tradable "
            "spread — and §4c shows the residual neither predicts returns OOS nor pays. Absent "
            "that, \"cointegrated\" is just \"spurious\" wearing a better coat.\n"
            "- **What a real rescue would need:** an out-of-sample residual→return |t| ≥ 2, or a "
            "timer that beats both buy-and-hold and the random-timing placebo net of costs. "
            "Neither exists on this tape at any of four horizons.\n"
            "- **Dedup map:** [323-btc-halving](../../323-btc-halving/) (the halving as a calendar "
            "event, not a valuation level), [293-mvrv-ratio](../../293-mvrv-ratio/) (a "
            "market-derived on-chain valuation ratio and mean-reversion timer), "
            "[663-hash-ribbons](../../663-hash-ribbons/) (a miner-capitulation discrete buy "
            "signal), [221-mayer-multiple](../../221-mayer-multiple/) (price/200-day-SMA bands), "
            "[210-crypto-trend](../../210-crypto-trend/) (200-day price SMA trend-following). None "
            "test PlanB's literal `ln(price) ~ ln(SF)` level model.\n\n"
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
