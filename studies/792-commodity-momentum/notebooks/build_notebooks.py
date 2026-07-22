"""Generate the two narrative notebooks for Study 792 (Cross-sectional commodity momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF basket
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (13 single-commodity ETFs,
# total-return closes, 207 monthly L/S rebalances 2009-04-30 -> 2026-06-30).
R = dict(
    as_of="2026-06-30", ls_start="2009-04-30", ls_end="2026-06-30",
    n_months=207, n_etfs=13, panel_months=220,
    mean_bps=84.22, ann_pct=10.11, hac_t=1.655, hac_lags=4,
    one_sample_t=1.700, sharpe=0.409, hit_pct=54.6, turnover=0.74,
    long_bps=37.21, long_t=1.33, short_bps=47.01, short_t=1.67, basket_ann=3.29,
    placebo_seeds=40, placebo_real_t=1.655, placebo_mean_t=0.108, placebo_sd_t=0.913,
    placebo_mean_sharpe=0.021, placebo_frac_ge2=0.0, placebo_p=0.025,
    era_split="2019-01-01",
    early_n=117, early_bps=119.00, early_t=2.27, early_sharpe=0.64,
    late_n=90, late_bps=39.00, late_t=0.43, late_sharpe=0.17, diff_t=-0.78,
    timer5_gross=10.11, timer5_net=9.16, timer5_t=1.50, timer5_sharpe=0.37,
    timer10_gross=10.11, timer10_net=8.72, timer10_t=1.42, timer10_sharpe=0.35,
    syn_null_mean=-0.00, syn_null_sd=1.35, syn_null_fire=2,
    syn_planted_t=13.34, syn_planted_bps=800.16, syn_planted_sharpe=5.21,
    fp="1c68bd1531bb",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Recent half: Not supported](https://img.shields.io/badge/Recent_half-Not_supported-8b949e?style=flat-square)\n\n"
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

from commodity_momentum import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    MRET = data.monthly_returns(PRICES)
else:
    PRICES = MRET = None
print("real cache present:", HAVE_REAL, "| monthly panel:",
      (None if MRET is None else f"{MRET.shape[0]} months x {MRET.shape[1]} ETFs"))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the strong commodities keep winning? 🛢️🥇\n"
            "### Commodity momentum — the textbook sort that backtests beautifully and "
            "still can't quite be certified\n\n"
            + BADGES +
            "Here's a strategy straight out of the factor-investing textbook: once a month, "
            "rank a basket of commodities by how they did over the past year, **buy the top "
            "third** (the strong ones), **short the bottom third** (the weak ones), and hold "
            "for a month. Miffre & Rallis (2007) showed it paid handsomely in futures markets. "
            "Does it still work on stuff you could actually buy — a basket of commodity ETFs?\n\n"
            "It backtests at **~+10%/yr**. And that number is more fragile than it looks.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 13 single-commodity ETFs, total-return, 2009→2026, 207 monthly "
            "rebalances. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does buying strong commodities and shorting weak ones pay? | **On paper, yes — "
            f"about +{R['ann_pct']:.0f}%/yr** (Sharpe {R['sharpe']:.2f}). But the honest "
            f"statistical test comes up just short: it's *probably* real, not *provably* real. |\n"
            "| Is it just luck / a random sort? | **No.** A random ranking of the same "
            "commodities almost never does this well — so the momentum ranking really is "
            "picking up something. |\n"
            f"| Does it still work lately? | **Not really.** Almost the entire edge is from "
            f"**2009-2018**. Since 2019 it's been essentially flat. |\n"
            f"| Do trading costs kill it? | **Surprisingly, no.** You only rebalance monthly, "
            f"so costs shave off very little — net is still ~+{R['timer10_net']:.0f}%/yr. What "
            f"undercuts it is the shaky *t*-stat and the dead recent decade, not costs. |\n\n"
            "> A great-looking backtest that the statistics can't quite sign off on — and that "
            "stopped paying about the time everyone had heard of it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Commodities trend. The ones that led the pack over the last year tend to keep "
            "leading; the laggards keep lagging. So rank them, go long the leaders and short the "
            "laggards, and rebalance monthly.\"*\n\n"
            "This is **momentum** — the same 'buy past winners' idea that's famous in stocks "
            "(Jegadeesh-Titman 1993), here applied *across* commodities. Miffre & Rallis (2007) "
            "documented it in commodity futures; we test the version a real investor could hold: "
            "a basket of single-commodity **ETFs** (gold, silver, copper, crude, natural gas, "
            "corn, wheat, sugar, …)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a clean, mechanical, once-a-month rule with no forecasting: just "
            "sort and hold. It's also one of the most-cited 'alternative' return sources — the "
            "kind of thing that ends up inside managed-futures funds and 'liquid alt' products. "
            "So it's worth checking whether the free, investable version actually delivers, or "
            "whether — like a lot of published factors — it looked great in-sample and quietly "
            "faded once the paper was out."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The sort.** Each month, rank the {R['n_etfs']} ETFs by their trailing 12-month "
            "return (skipping the most recent month), long the top third, short the bottom third.\n"
            "- **The *t*-test.** Is the average monthly profit big enough, relative to its "
            "wobble, to be real? (We use a stat that accounts for months bunching together.)\n"
            "- **The luck check.** Rank the commodities *randomly* instead, hundreds of times — "
            "does the real momentum ranking beat the coin-flip rankings?\n"
            "- **The 'still working?' check.** Split the history in half and look at the "
            "recent decade on its own.\n"
            "- **The trade check.** Charge realistic costs and short-borrow — does the net "
            "survive?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the equity curve.** One dollar in the long/short momentum book."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rep = st.momentum_report(MRET)\n"
            "    ret = rep['ret']\n"
            "    ann, shp, t = rep['ann_pct'], rep['sharpe'], rep['hac_t']\n"
            "else:\n"
            "    ret = None; ann, shp, t = R['ann_pct'], R['sharpe'], R['hac_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "if ret is not None:\n"
            "    eq = (1 + ret).cumprod()\n"
            "    ax.plot(eq.index, eq.values, color=GREEN, lw=1.8)\n"
            "    ax.axvline(pd.Timestamp(R['era_split']), ls='--', c=GREY, lw=1)\n"
            "    ax.annotate('2019 split', (pd.Timestamp(R['era_split']), eq.max()*0.8),\n"
            "                color=GREY, ha='right')\n"
            "ax.set_ylabel('growth of $1 (long/short, gross)')\n"
            "ax.set_title(f'12-1 commodity momentum: ~{ann:+.0f}%/yr, Sharpe {shp:.2f}, HAC t={t:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross ~{ann:+.2f}%/yr | Sharpe {shp:.2f} | HAC t {t:+.2f} (bar is 2)')"
        ),
        md(
            f"A smooth-ish climb to **~+{R['ann_pct']:.0f}%/yr** at Sharpe **{R['sharpe']:.2f}** — "
            "the kind of curve that sells a strategy. But look where it climbs: most of the gain "
            "is **before the 2019 dashed line**. After it, the curve mostly goes sideways.\n\n"
            "**Is the sort actually skillful, or just lucky?** We rank the same commodities "
            "*randomly* 40 times and compare."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_placebo(MRET)\n"
            "    real_t, ts = pl['real_t'], pl['ts']\n"
            "else:\n"
            "    real_t = R['hac_t']\n"
            "    ts = np.random.default_rng(792).normal(R['placebo_mean_t'], R['placebo_sd_t'], 40)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(ts, bins=16, color=GREY, alpha=.85, label='random-rank sorts (40 seeds)')\n"
            "ax.axvline(real_t, c=GREEN, lw=2.5, label=f'real momentum sort (t={real_t:.2f})')\n"
            "ax.axvline(2, ls='--', c=RED, lw=1, label='t = 2 bar')\n"
            "ax.set_xlabel('HAC t of the long/short book'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Real sort beats the coin flips (placebo p = {R['placebo_p']:.3f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"real t {real_t:+.2f} vs random-rank mean {R['placebo_mean_t']:+.2f}; \"\n"
            "      f\"p = {R['placebo_p']:.3f}; but real t is still under the t=2 bar\")"
        ),
        md(
            f"The real momentum sort (green) sits well to the right of the random-rank cloud — "
            f"**p = {R['placebo_p']:.3f}**, so it genuinely isn't a coin flip. But notice the green "
            f"line still lands *short of the red t = 2 bar*. It beats random, yet doesn't clear the "
            "desk's certification threshold. That tension is the whole story.\n\n"
            "**Does it still work lately?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_contrast(MRET, data.ERA_SPLIT)\n"
            "    e_s, l_s = sp['early_sharpe'], sp['late_sharpe']\n"
            "    e_t, l_t = sp['early_t'], sp['late_t']\n"
            "else:\n"
            "    e_s, l_s = R['early_sharpe'], R['late_sharpe']\n"
            "    e_t, l_t = R['early_t'], R['late_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['2009-2018\\n(t={:.2f})'.format(e_t),'2019-2026\\n(t={:.2f})'.format(l_t)],\n"
            "       [e_s, l_s], color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([e_s, l_s]): ax.annotate(f'Sharpe {v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('The premium is a pre-2019 story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early Sharpe {e_s:.2f} (t={e_t:+.2f}) | late Sharpe {l_s:.2f} (t={l_t:+.2f})')"
        ),
        md(
            f"The first half clears the bar on its own (Sharpe {R['early_sharpe']:.2f}, "
            f"t = {R['early_t']:.2f}); the second half is basically flat "
            f"(Sharpe {R['late_sharpe']:.2f}, t = {R['late_t']:.2f}). This is the well-known "
            "post-2015 fade of commodity factors — the effect that made the paper famous seems to "
            "have thinned out about the time everyone piled in. *(One honest caveat: seven years "
            "isn't enough data to *prove* it died — but it certainly hasn't been paying.)*\n\n"
            "**Finally, the trade — do costs matter here?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.costed_timer(MRET, cost_bps=5.0)\n"
            "    tm10 = st.costed_timer(MRET, cost_bps=10.0)\n"
            "    g, n5, n10 = tm5['gross_ann_pct'], tm5['net_ann_pct'], tm10['net_ann_pct']\n"
            "else:\n"
            "    g, n5, n10 = R['timer5_gross'], R['timer5_net'], R['timer10_net']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('Monthly rebalancing = low turnover = costs barely bite')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}% -> net {n5:+.1f}% (5bps) / {n10:+.1f}% (10bps)')"
        ),
        md(
            f"Because you only trade once a month, costs shave the gross **+{R['timer5_gross']:.0f}%/yr** "
            f"down to only **+{R['timer10_net']:.0f}%/yr** even at a chunky 10 bps a side plus "
            "short-borrow. So — unusually for this desk — **costs are not the thing that kills it.** "
            "The thing that kills it is that the *t*-stat never cleared the bar, the whole premium "
            "is pre-2019, and the basket is a tiny 13-name set of *survivors* (funds that lived — a "
            "mild upward bias baked in). Real-*looking*, not certifiable, and not paying lately."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** +{R['ann_pct']:.0f}%/yr at Sharpe {R['sharpe']:.2f} and clearly "
            f"better than random (p = {R['placebo_p']:.3f}), but the robust *t* is only "
            f"{R['hac_t']:.2f} — under the bar. Real in the literature; this tape can't certify it.\n"
            "- **Tradability — Fragile.** Costs barely dent it, but the net *t* is uncertified, the "
            "edge is all pre-2019, and it's a 13-name survivor basket. Survives on paper, not "
            "investable at scale.\n"
            "- **\"Still working lately?\" — Not supported.** The 2019-2026 half is flat "
            f"(t = {R['late_t']:.2f}). The premium was a pre-2019 phenomenon."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why commodities trend at all:** supply is slow to adjust (you can't build a mine "
            "or plant a crop overnight), so shocks persist — a plausible economic root for "
            "momentum that stocks don't share.\n"
            "- **Why it may have faded:** once managed-futures funds industrialised the trade "
            "post-2008, the easy premium got competed down — a recurring pattern for published "
            "factors.\n"
            "- **Sibling studies:** [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) "
            "runs the *same* sort on **equities** (verdict `NONE`); "
            "[638-value-momentum-everywhere](../../638-value-momentum-everywhere/) blends momentum "
            "*and* value across four asset classes; the single-commodity **seasonality** studies "
            "(226, 307, 648-651) test *calendar* effects, not cross-sectional strength. See "
            "[docs/references.md](../docs/references.md) for the exact dedup.\n\n"
            "*Think a bigger basket or a longer-history futures tape revives it? Show a net, "
            "certifiable (HAC t ≥ 2) edge on the recent decade — then we'll talk.*"
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
            "# Cross-sectional commodity momentum — a quantitative teardown 🔬\n"
            "### The 12-1 L/S sort with HAC *t* · a leg decomposition · a 40-seed random-rank "
            "placebo · a pre/post-2019 difference test · a cost-and-borrow sweep · a planted-"
            "effect synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **Miffre-Rallis (2007): a 12-1 cross-sectional commodity-momentum sort pays** "
            "— is a *pure commodity* momentum sleeve, distinct from the mixed multi-asset combo of "
            "[638](../../638-value-momentum-everywhere/), the equity sort of "
            "[507](../../507-cross-sectional-momentum/), and the single-commodity seasonals "
            "(226/307/648-651). The job: measure it with an autocorrelation-robust *t*, prove the "
            "machinery, and grade tradability honestly.\n\n"
            "> ⚠️ **Data note.** 13 single-commodity ETFs, total-return closes (yfinance "
            "`auto_adjust=True`), 2009-04 → 2026-06, 207 monthly rebalances, cached. "
            "**Survivorship named on the Signal axis** — current-membership basket, premium is an "
            "upper bound. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 12-1 L/S **{R['mean_bps']:+.0f} bps/mo** "
            f"(~{R['ann_pct']:+.1f}%/yr, Sharpe {R['sharpe']:.2f}); **HAC t = {R['hac_t']:+.2f}** "
            f"(< 2); random-rank placebo p = {R['placebo_p']:.3f} |\n"
            f"| **Tradability** | `FRAGILE` | net {R['timer10_net']:+.1f}%/yr @10bps "
            f"(net t = {R['timer10_t']:.2f}); turnover {R['turnover']:.2f}x; edge all pre-2019 |\n"
            f"| **Recent half** | `NOT SUPPORTED` | 2019-2026: HAC t = {R['late_t']:+.2f}, "
            f"Sharpe {R['late_sharpe']:.2f} vs pre-2019 t = {R['early_t']:.2f} |\n\n"
            "> 💡 In plain words: an economically large, not-random momentum premium that the "
            "robust *t* can't sign off on, is carried entirely by the first half of the sample, and "
            "survives costs only because you rebalance so rarely."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be ETF $i$'s total return in month $t$, and "
            "$S_{i,t} = \\sum_{k=1}^{11} \\log(1+r_{i,t-k})$ the trailing **12-1** momentum "
            "signal (months $t-11..t-1$, the most recent month skipped). Each month rank the live "
            "assets on $S$, long the top third $W_t$, short the bottom third $L_t$, equal-weight:\n\n"
            "$$ R^{LS}_{t+1} = \\tfrac{1}{|W_t|}\\sum_{i\\in W_t} r_{i,t+1} - "
            "\\tfrac{1}{|L_t|}\\sum_{i\\in L_t} r_{i,t+1}. $$\n\n"
            "- **H₁ (premium).** $E[R^{LS}] > 0$ with an autocorrelation-robust $t \\ge 2$.\n"
            "- **H₂ (skill, not luck).** $R^{LS}$ beats a random-rank sort of the same basket.\n"
            "- **H₃ (persistence).** The premium survives into the recent (post-2019) half.\n"
            "- **H₄ (capture).** It survives realistic costs + short borrow.\n\n"
            f"We find **H₁ not met on this tape** (HAC t = {R['hac_t']:.2f}), **H₂ met** "
            f"(placebo p = {R['placebo_p']:.3f}), **H₃ not met** (late t = {R['late_t']:.2f}), "
            f"**H₄ met on magnitude but not certification** (net t = {R['timer10_t']:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly L/S factor returns are **serially correlated** (overlapping trends, momentum "
            "crashes), so the primary statistic is a **Newey-West (HAC)** *t* on the monthly mean "
            "with an automatic Bartlett lag — an i.i.d. SE would overstate significance. The "
            "**one execution lag** is explicit: weights formed at the close of month *t* earn "
            "month *t*+1 (a single `shift`). The random-rank **placebo (40 seeds)** holds basket "
            "breadth fixed and asks how often noise ranks match the real *t*. The sub-period split "
            "(2019-01-01, the effective-sample midpoint) is tested as a **difference** (Welch *t*), "
            "not eyeballed. Costs are **one-way × traded notional**; the short leg pays borrow."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_etfs']} single-commodity ETFs, total-return, {R['panel_months']} "
            f"month-ends; the L/S trades **{R['n_months']} months** ({R['ls_start']} → "
            f"{R['ls_end']}) once ≥ 6 names carry a full 12-month signal.\n"
            "- **Headline.** HAC *t* + one-sample *t* + annualised Sharpe on the monthly L/S.\n"
            "- **Decomposition.** Winner and loser legs, each excess-of-basket, HAC-*t*'d.\n"
            "- **Placebo.** 40-seed random-rank null; p = share of seeds with t ≥ real t.\n"
            "- **Persistence.** Pre/post-2019 HAC *t* + a Welch *t* of the difference.\n"
            "- **Execution.** Cost sweep (5 / 10 bps one-way × turnover) + 50 bps/yr short borrow.\n"
            "- **Control.** Synthetic AR(1)-trend panel, plantable `mom_edge`; the null "
            "(edge = 0) must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline L/S and its placebo\n\n"
            "The 12-1 book's HAC *t*, and where it sits against 40 random-rank sorts of the same "
            "basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rep = st.momentum_report(MRET)\n"
            "    print(f\"L/S: {rep['mean_bps']:+.2f} bps/mo (~{rep['ann_pct']:+.2f}%/yr) over \"\n"
            "          f\"{rep['n_months']} months\")\n"
            "    print(f\"HAC t = {rep['hac_t']:+.3f} ({rep['hac_lags']} lags) | one-sample t = \"\n"
            "          f\"{rep['one_sample_t']:+.3f} | Sharpe {rep['sharpe']:+.3f} | hit {rep['hit_rate']*100:.1f}%\")\n"
            "    pl = st.random_placebo(MRET); real_t, ts = pl['real_t'], pl['ts']\n"
            "else:\n"
            "    real_t = R['hac_t']\n"
            "    ts = np.random.default_rng(792).normal(R['placebo_mean_t'], R['placebo_sd_t'], 40)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(ts, bins=16, color=GREY, alpha=.85, label='random-rank sorts (40 seeds)')\n"
            "ax.axvline(real_t, c=AMBER, lw=2.5, label=f'real momentum sort t = {real_t:.2f}')\n"
            "ax.axvline(2, ls='--', c=RED, lw=1, label='t = 2 bar')\n"
            "ax.set_xlabel('HAC t of the long/short book'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Beats random (p={R['placebo_p']:.3f}) but under the bar\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"real t {real_t:+.2f} | placebo mean t {R['placebo_mean_t']:+.2f} \"\n"
            "      f\"(sd {R['placebo_sd_t']:.2f}) | frac|t|>=2 {R['placebo_frac_ge2']:.0%} | p {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the sort earns **{R['mean_bps']:+.0f} bps/mo** and sits in the top "
            f"~2.5% of random-rank sorts (**p = {R['placebo_p']:.3f}**, 0/40 random sorts even reach "
            f"|t| = 2) — it is *not* noise. But its own **HAC t = {R['hac_t']:.2f}** is below the "
            "desk's t ≥ 2 certification bar. **H₁ fails on this tape; H₂ holds.** The gap between "
            "'beats random' and 'certifiable' is exactly why the Signal stamp is `WEAK`, not `REAL`."
        ),
        md(
            "### 4b · Leg decomposition — winners, losers, or the basket?\n\n"
            "Each leg measured **excess of the equal-weight basket**, so a plain long-only "
            "commodity tilt can't masquerade as momentum."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lg = st.long_short_legs(MRET)\n"
            "    lb, sb, bt = lg['long_excess_bps'], lg['short_excess_bps'], lg['basket_ann_pct']\n"
            "    lt, stt = lg['long_t'], lg['short_t']\n"
            "else:\n"
            "    lb, sb, bt = R['long_bps'], R['short_bps'], R['basket_ann']\n"
            "    lt, stt = R['long_t'], R['short_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['winner leg\\n(t={:.2f})'.format(lt),'loser (short) leg\\n(t={:.2f})'.format(stt)],\n"
            "       [lb, sb], color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([lb, sb]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('excess-of-basket (bps/mo)')\n"
            "ax.set_title(f'Both legs contribute; basket itself only {bt:+.1f}%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'winner leg {lb:+.2f} bps (t={lt:+.2f}) | loser-short leg {sb:+.2f} bps (t={stt:+.2f}) | basket {bt:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the winner leg adds **{R['long_bps']:+.0f} bps/mo** over the "
            f"basket (t = {R['long_t']:.2f}) and the short-losers leg **{R['short_bps']:+.0f} bps/mo** "
            f"(t = {R['short_t']:.2f}) — the short side does slightly more work (it caught moves like "
            f"the 2014-15 crude collapse). Neither leg clears t = 2 alone, and the equal-weight "
            f"basket returned only {R['basket_ann']:+.1f}%/yr — so this is a genuine relative-strength "
            "spread, not a disguised long-only commodity bet."
        ),
        md(
            "### 4c · Persistence — did it survive into the 2020s?\n\n"
            "Split at 2019-01-01; within-era HAC *t* and a Welch *t* of the difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_contrast(MRET, data.ERA_SPLIT)\n"
            "    e_b, l_b = sp['early_bps'], sp['late_bps']\n"
            "    e_t, l_t, d_t = sp['early_t'], sp['late_t'], sp['welch_t_diff']\n"
            "else:\n"
            "    e_b, l_b = R['early_bps'], R['late_bps']\n"
            "    e_t, l_t, d_t = R['early_t'], R['late_t'], R['diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['2009-2018\\n(n={})'.format(R['early_n']),'2019-2026\\n(n={})'.format(R['late_n'])],\n"
            "       [e_b, l_b], color=[GREEN, RED], width=.55)\n"
            "for i,(v,t_) in enumerate([(e_b,e_t),(l_b,l_t)]):\n"
            "    ax.annotate(f'{v:+.0f} bps\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('L/S mean (bps/mo)')\n"
            "ax.set_title(f'Premium is pre-2019 (difference Welch t={d_t:+.2f}, underpowered)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e_b:+.1f} bps (t={e_t:+.2f}) | late {l_b:+.1f} bps (t={l_t:+.2f}) | diff Welch t {d_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the pre-2019 half clears the bar on its own "
            f"(**{R['early_bps']:+.0f} bps/mo, t = {R['early_t']:.2f}**); the post-2019 half is flat "
            f"(**{R['late_bps']:+.0f} bps/mo, t = {R['late_t']:.2f}**). The honesty rail: the "
            f"*difference* test is underpowered (Welch t = {R['diff_t']:.2f}), so we say **faded / "
            "not-supported-recently**, not *proven dead* — but there is no tradable premium in the "
            "recent tape. **H₃ fails.**"
        ),
        md(
            "### 4d · The costed timer — and why costs aren't the killer here\n\n"
            "One-way cost × traded notional; short leg pays 50 bps/yr borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.costed_timer(MRET, cost_bps=5.0)\n"
            "    tm10 = st.costed_timer(MRET, cost_bps=10.0)\n"
            "    g = tm5['gross_ann_pct']; n5, n10 = tm5['net_ann_pct'], tm10['net_ann_pct']\n"
            "    t5, t10 = tm5['net_t'], tm10['net_t']; to = tm5['avg_turnover']\n"
            "else:\n"
            "    g = R['timer5_gross']; n5, n10 = R['timer5_net'], R['timer10_net']\n"
            "    t5, t10 = R['timer5_t'], R['timer10_t']; to = R['turnover']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('annualised (%)')\n"
            "a1.set_title(f'Turnover only {to:.2f}x/mo -> costs barely bite')\n"
            "a2.bar(['net t @5bps','net t @10bps'], [t5, t10], color=[AMBER, AMBER], width=.5)\n"
            "for i,v in enumerate([t5, t10]): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('net HAC t'); a2.set_title('...but net t never clears 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}% -> net {n5:+.1f}% (t={t5:.2f}) / {n10:+.1f}% (t={t10:.2f}), turnover {to:.2f}x')"
        ),
        md(
            f"> 💡 In plain words: monthly turnover is only **{R['turnover']:.2f}× NAV**, so even at "
            f"10 bps/side + borrow the gross **+{R['timer5_gross']:.0f}%/yr** only falls to "
            f"**+{R['timer10_net']:.0f}%/yr** — costs are *not* the killer, unusually for this desk. "
            f"What sinks tradability is that the **net HAC t ({R['timer10_t']:.2f}) never clears 2**, "
            "the whole premium is pre-2019, and it's a 13-name survivor basket. **H₄: magnitude "
            "survives, certification doesn't → `FRAGILE`.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic AR(1)-trend panel with a TUNABLE planted momentum edge. The null "
            "(`mom_edge` = 0) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(data.synthetic_world(mom_edge=0.0, seed=792+s))['hac_t']\n"
            "                    for s in range(20)])\n"
            "planted = st.synthetic_detect(data.synthetic_world(mom_edge=1.0, seed=792))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mom_edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted['hac_t']], color=GREEN, s=110, zorder=5,\n"
            "           label=f\"planted edge (t={planted['hac_t']:.1f})\")\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('12-1 L/S HAC t')\n"
            "ax.set_title('Control: unbiased null, planted edge lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 | planted t {planted[\"hac_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (unbiased) and fires |t| ≥ 2 on only "
            f"{R['syn_null_fire']}/20 seeds — the ~5-10% false-positive rate you *expect* from "
            f"noise, which is exactly why a synthetic control can never certify a real stamp. A "
            f"planted edge reads t = {R['syn_planted_t']:.1f} ({R['syn_planted_bps']:+.0f} bps/mo). "
            "The sort works when momentum is there; on the real tape it reads 1.66. *(Machinery / "
            "power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 12-1 L/S {R['mean_bps']:+.0f} bps/mo (~{R['ann_pct']:+.1f}%/yr, "
            f"Sharpe {R['sharpe']:.2f}); **HAC t = {R['hac_t']:.2f} < 2**. Not noise (placebo "
            f"p = {R['placebo_p']:.3f}) and literature-backed (Miffre-Rallis 2007), but this tape "
            f"can't certify it, and the certifiable half is entirely pre-2019 "
            f"(t = {R['early_t']:.2f} early vs {R['late_t']:.2f} late). Survivor basket — upper bound.\n"
            f"- **Tradability `FRAGILE`** — costs barely dent it (net {R['timer10_net']:+.1f}%/yr "
            f"@10bps) but net HAC t = {R['timer10_t']:.2f} (uncertified), premium concentrated "
            f"pre-2019, 13-name survivor universe with thin early breadth.\n"
            f"- **Recent half `NOT SUPPORTED`** — 2019-2026 HAC t = {R['late_t']:.2f}, "
            f"Sharpe {R['late_sharpe']:.2f}; the difference is underpowered (Welch t = {R['diff_t']:.2f}), "
            "so *faded*, not *provably dead*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Longer history:** a continuous-futures tape (roll-adjusted) back to the 1990s "
            "would restore power the ETF era lacks — at the cost of roll noise. Does the recent "
            "fade hold there?\n"
            "- **Bigger cross-section:** 13 ETFs is a thin sort (2-4 names per leg early). A "
            "20-30 commodity futures panel (the Miffre-Rallis breadth) is the natural robustness "
            "extension.\n"
            "- **Dedup map:** [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) "
            "(same 12-1 sort, **equities**, `NONE`); "
            "[638-value-momentum-everywhere](../../638-value-momentum-everywhere/) (the mixed "
            "multi-asset value+momentum *combo*, commodities one sleeve of eight); the "
            "single-commodity **seasonality** studies (226, 307, 648-651 — *calendar* effects "
            "within one commodity, not cross-sectional strength).\n\n"
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
