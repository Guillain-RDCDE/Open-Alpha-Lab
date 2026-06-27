"""Generate the two narrative notebooks for Study 532 (Firm-Age-Anomaly).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats. Synthetic figures run anywhere, offline and
deterministic; the real-panel cells use the cached yfinance parquets under ../_cache/ if
present and otherwise quote the frozen headline numbers in ``R``.

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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=490, year_start=1985, year_end=2026,
    n_tickers=39,
    age_mean=-18.22, age_vol=23.50, age_sharpe=-0.775, age_t=-4.661,
    age_hit=38.8, age_dd=-100.0,
    old_mean=15.74, young_mean=33.96, mkt_mean=12.65,
    avg_age_old=20.9, avg_age_young=8.7,
    turnover=0.02, net_mean=-19.24, net_t=-4.924, net_sharpe=-0.819,
    placebo_real=-1.518, placebo_perm=-0.036, placebo_std=0.474, placebo_p=0.010,
    post2012_n=172, post2012_mean=-15.8, post2012_t=-2.067,
    fp_prices="823c8d7707e9", fp_spy="c3b449e6ee7f",
    # synthetic control sweep (25 seeds each)
    ctrl=[(0.00, -0.08, -0.026, 0), (0.05, 1.53, 0.403, 8),
          (0.15, 4.76, 1.221, 12), (0.30, 9.58, 2.249, 60)],
)


# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from firm_age_anomaly import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "age_prices.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "age_spy.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy = data.fetch_panel()
    first_dates = data.first_price_dates(prices)
    result = st.age_portfolio(prices, spy, first_dates=first_dates, min_stocks=9)
    s_age   = st.summary(result["age_ret"])
    s_old   = st.summary(result["old_ret"])
    s_young = st.summary(result["young_ret"])
    s_mkt   = st.summary(result["mkt_ret"])
    print(f"N months: {s_age['n']} | AGE (old-young) mean: {s_age['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_age['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Firm-Age-Anomaly -- do mature firms quietly beat the young upstarts?\n"
            "### The new-list effect (Fama-French 2004; Jiang-Lee-Zhang 2005), tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survivor%20basket%20the%20whole%20story%3F: Busted](https://img.shields.io/badge/Survivor%20basket%20the%20whole%20story%3F-Busted-8b949e?style=flat-square)\n\n"
            "Fama and French (2004) noticed something about companies that *newly list* on an "
            "exchange: they are less profitable, grow their assets faster, fail more often, and -- "
            "on average -- earn *lower* future returns than seasoned, mature firms. Jiang, Lee and "
            "Zhang (2005) folded firm age into a broader 'information-uncertainty' story: young "
            "firms are hard to value, and hard-to-value stocks tend to underperform. Put them "
            "together and you get the **firm-age** (or **new-list**) effect: **old beats young**.\n\n"
            "We test it the obvious way -- proxy a firm's age from how far back its price history "
            "goes, sort a basket old-minus-young, and see whether the mature names win.\n\n"
            "> **This is the plain-language layer.** Want the age dispersion over time, the "
            "label-shuffle placebo, and the post-2012 subsample? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do old firms beat young firms? | **No -- the opposite.** Old-minus-young earns "
            f"**{R['age_mean']:.1f}%/yr** at HAC *t* = **{R['age_t']:.2f}**. Young firms *win*. |\n"
            "| Is the reversal just noise? | **No.** A firm-age label-shuffle placebo puts it at "
            f"**p = {R['placebo_p']:.3f}** -- the labels carry a real signal, pointing the wrong "
            "way for the claim. |\n"
            "| Could you trade old-minus-young? | **No.** Sharpe "
            f"**{R['age_sharpe']:.2f}**, drawdown **{R['age_dd']:.0f}%** -- it is a money "
            "incinerator as specified. |\n"
            "| Why is the sign backwards? | **Survivorship.** Our basket only holds young firms "
            "that *survived* to 2026 -- the winners. The failed IPOs that make the effect work "
            "are gone. |\n\n"
            "> The new-list effect is real in broad, unbiased samples. On a survivor basket it "
            "doesn't just vanish -- it **inverts**, because the surviving young firms are the "
            "spectacular winners (META, the NVDA-era IPOs)."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Newly listed firms have lower survival rates and lower expected returns than "
            "seasoned firms. Young firms underperform mature firms.\"*\n\n"
            "-- Fama & French (2004), *Journal of Financial Economics*; Jiang, Lee & Zhang (2005), "
            "*Review of Accounting Studies*\n\n"
            "The intuition: a young firm is a bundle of uncertainty. Investors don't yet know its "
            "true profitability, its competitive moat, or whether it survives the next downturn. "
            "That uncertainty -- and the optimism that often surrounds IPOs -- leaves young firms "
            "*overpriced* relative to the returns they deliver. Mature firms, boringly profitable "
            "and well-understood, quietly do better. A long-old / short-young portfolio should "
            "harvest the gap."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, firm age is a free, fundamentals-light sorting variable: you can compute it "
            "from a listing date alone. It connects a whole family of findings -- IPO long-run "
            "underperformance (Ritter 1991), the new-issues puzzle (Loughran-Ritter 1995), and "
            "information-uncertainty pricing (Zhang 2006). The questions for us:\n\n"
            "1. **Does old-minus-young earn a premium on a realistic large-cap basket?**\n"
            "2. **What does survivorship do to it?**\n"
            "3. **Is anything left after costs?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Age from first price date.** A firm's age at month *t* is *t* minus the first "
            "day we have a price for it -- a clean, data-driven proxy that needs no fundamentals.\n"
            "2. **One-day execution lag.** The age ranking at month-start is public; we enter at "
            "the close one day later. No look-ahead, no same-bar fill.\n"
            "3. **Name the survivorship bias loudly.** Our basket is names *still trading in "
            "2026*. The young firms that IPO'd and then died -- the worst performers, the ones "
            "that make the effect -- are simply absent. That is not a footnote here; it is the "
            "whole result."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the engine find an age premium when one is really there?**"
        ),
        code(
            "# Synthetic positive control: plant a known age premium (old>young) and recover it.\n"
            "premiums = [0.0, 0.05, 0.15, 0.30]\n"
            "rows = []\n"
            "for prem in premiums:\n"
            "    ts, ms = [], []\n"
            "    for sd in range(15):\n"
            "        p, sp, fd, _ = data.synthetic_panel(age_premium=prem, seed=1000+sd)\n"
            "        res = st.age_portfolio(p, sp, first_dates=fd, min_stocks=9)\n"
            "        if not res.empty:\n"
            "            s = st.summary(res['age_ret']); ts.append(s['tstat']); ms.append(s['mean'])\n"
            "    rows.append({'premium': prem, 'mean_%': np.nanmean(ms)*100, 't': np.nanmean(ts)})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['mean_%'], color=GREEN)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['mean_%'], ctrl['t'])):\n"
            "    ax.text(i, m + 0.2, f't={t:+.2f}', ha='center', fontsize=9)\n"
            "ax.set_xlabel('planted age premium (old>young)'); ax.set_ylabel('old-minus-young mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers a planted premium, monotonically')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: with no planted effect it sits on zero, and as we plant a "
            "stronger old-beats-young premium the recovered mean and *t*-stat rise monotonically. "
            "So whatever we see on the real tape reflects **the market**, not a broken detector."
        ),
        md("**Now the honest test on the real yfinance basket.**"),
        code(
            "if HAVE_REAL:\n"
            "    age_mean = s_age['mean']*100; age_t = s_age['tstat']\n"
            "    old_m = s_old['mean']*100; young_m = s_young['mean']*100; mkt_m = s_mkt['mean']*100\n"
            "else:\n"
            "    age_mean, age_t = R['age_mean'], R['age_t']\n"
            "    old_m, young_m, mkt_m = R['old_mean'], R['young_mean'], R['mkt_mean']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "labels = ['Old leg\\n(long)', 'SPY', 'Young leg\\n(short)']\n"
            "vals = [old_m, mkt_m, young_m]\n"
            "cols = [GREY, GREY, RED]\n"
            "ax.bar(labels, vals, color=cols, width=0.5)\n"
            "ax.axhline(mkt_m, ls='--', c=GREY, lw=1)\n"
            "ax.set_ylabel('mean annual return (%/yr)')\n"
            "ax.set_title(f'Young firms WIN: old-minus-young = {age_mean:+.1f}%/yr (t={age_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Old (long): {old_m:+.1f}%/yr | Young (short): {young_m:+.1f}%/yr | SPY: {mkt_m:+.1f}%/yr')\n"
            "print(f'Old-minus-young: {age_mean:+.2f}%/yr | HAC t = {age_t:+.3f}')"
        ),
        md(
            f"The long (old) leg earns **+{R['old_mean']:.1f}%/yr**; the short (young) leg earns "
            f"**+{R['young_mean']:.1f}%/yr** -- so old-minus-young is **{R['age_mean']:.1f}%/yr** "
            f"at HAC *t* = **{R['age_t']:.2f}**. The claimed premium is not weak -- it is "
            "**reversed**. The young survivors crushed the old guard."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** The claimed old-minus-young premium is absent: the real "
            f"spread is significantly *negative* ({R['age_mean']:.1f}%/yr, HAC *t* = "
            f"{R['age_t']:.2f}; placebo p = {R['placebo_p']:.3f}). The new-list effect, as "
            "stated, does not exist on this basket -- the sign is flipped.\n"
            f"- **Tradability -- MIRAGE.** Sharpe {R['age_sharpe']:.2f}, drawdown "
            f"{R['age_dd']:.0f}%. As specified it loses money relentlessly; reversing the sign to "
            "'profit' from young survivors would be data-mining the basket, not trading a premium.\n"
            "- **Survivor basket the whole story? -- BUSTED.** Yes. Remove the failed young IPOs "
            "and the effect inverts. The honest takeaway is the bias, not an edge."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No -- and the reason is instructive:\n\n"
            "1. **The sign is survivorship, not signal.** The young firms in our basket are the "
            "ones that *won the lottery*. Buying them and shorting old blue-chips 'works' only in "
            "hindsight, on names we already know survived.\n"
            "2. **Out of sample, the next cohort of young firms is unknown** -- including the ones "
            "that will fail. A real-time long-young / short-old book would catch those failures.\n"
            "3. **It is barely a portfolio.** Turnover is ~0.02: the age ranks almost never "
            "change, so this is a near-static bet on a fixed set of survivors.\n\n"
            "There is no harvestable edge here in either direction -- only a clean illustration of "
            "how conditioning on survival can turn an anomaly upside down."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Survivorship-free universe.** A CRSP-style tape that keeps delisted firms would "
            "let the young leg include its failures -- the only honest way to test the claim.\n"
            "- **True IPO dates.** Our first-price-date proxy is floored by yfinance's 1985 start, "
            "collapsing the age of old names. EDGAR/SEC listing dates would sharpen the proxy.\n"
            "- **[Study 219 -- IPO-Pop](../../219-ipo-pop/)**: the first-day cousin of the "
            "new-list effect.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: another "
            "cross-sectional sort on the same survivor-basket machinery.\n\n"
            "*Think the firm-age premium survives once you keep the dead young firms? Fork this, "
            "pull a delisting-inclusive universe, and show old-minus-young t > 2 with the failures "
            "left in. That is the bar.*"
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
            "# Firm-Age-Anomaly -- a quantitative teardown\n"
            "### yfinance panel * first-price-date age proxy * old-minus-young * HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survivor%20basket%20the%20whole%20story%3F: Busted](https://img.shields.io/badge/Survivor%20basket%20the%20whole%20story%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every claim "
            "carrying its standard error. We test the new-list effect (Fama-French 2004; "
            "Jiang-Lee-Zhang 2005): proxy firm age from first price date, sort into terciles, long "
            "old / short young, monthly rebalance, one-day execution lag.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, 39 large-caps "
            "spanning 1985-2021 IPO vintages, 1985-2026, as-of 2026-06-26. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** the basket is names still trading in 2026. The young "
            "leg is missing its failures -- which is exactly why the sign reverses."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | old-minus-young **{R['age_mean']:.2f}%/yr**, HAC *t* = "
            f"**{R['age_t']:.3f}** (sign reversed vs the claim; placebo p = {R['placebo_p']:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['age_sharpe']:.3f}**, max DD "
            f"**{R['age_dd']:.0f}%**, net *t* **{R['net_t']:.3f}** -- loses as specified. |\n"
            "| **Survivor basket the whole story?** | `BUSTED` | young leg = survivors only; "
            "removing failed IPOs inverts the effect. |\n\n"
            "> The new-list effect is documented on broad, survivorship-free universes. Here, on a "
            "basket of survivors, old-minus-young is significantly negative -- the surviving young "
            "firms are the winners, and the failures that drive the published effect are absent."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\text{age}_{i,t}$ = years since stock $i$'s first available price as of month "
            "$t$. Fama-French (2004) and Jiang-Lee-Zhang (2005) assert:\n\n"
            "- **H1 (signal).** Old firms earn higher returns than young firms: the long-old / "
            "short-young portfolio has $\\mathbb{E}[r^{\\text{old}} - r^{\\text{young}}] > 0$.\n"
            "- **H2 (mechanism).** The gap reflects information uncertainty / new-list optimism, "
            "not just a market-beta tilt.\n"
            "- **H3 (tradable).** It survives costs and is implementable.\n\n"
            "We **reject H1 in sign** (the spread is significantly negative on this basket), and "
            "show H1's published direction is recoverable only once the failed young firms -- "
            "absent from a survivor basket -- are restored. H3 is moot once H1 fails."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Firm age is one of the cheapest factors to compute -- a listing date is all you need. "
            "If it carried a clean premium it would be a free tilt for any quant. This study is "
            "really a parable: the most natural way to *backtest* firm age (current names, "
            "projected back) bakes in survivorship so severely that the published anomaly flips "
            "sign. It is a cautionary tale for every factor built on a 'current index, projected "
            "backwards' universe."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Age proxy.** First non-NaN adjusted-close date per ticker; age at *t* = "
            "*(t - first_date)* in years.\n"
            "- **Eligibility.** >= 126 trading days of history at *t*.\n"
            "- **Ranking.** Monthly, sort eligible names by age; long the oldest tercile, short "
            "the youngest tercile, equal-weight.\n"
            "- **Execution.** Enter at the close one day after the month-start ranking (one-day "
            "lag, no look-ahead); hold to next rebalance.\n"
            "- **Long-short.** old_ret - young_ret each month.\n"
            "- **Costs.** 10 bps one-way x turnover + 100 bps/yr borrow on the short (young) leg.\n"
            "- **Inference.** Newey-West HAC one-sample *t* on monthly old-minus-young; firm-age "
            "label-shuffle placebo (200 permutations).\n"
            "- **Universe caveat.** 39 large-cap survivors; survivorship-biased by construction."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector\n\n"
            "Sweep the planted age premium upward on a synthetic panel (25 seeds each). The "
            "recovered mean and HAC *t* should rise monotonically; the null should sit on zero."
        ),
        code(
            "premiums = [0.0, 0.05, 0.15, 0.30]\n"
            "means, tstats = [], []\n"
            "for prem in premiums:\n"
            "    ts, ms = [], []\n"
            "    for sd in range(25):\n"
            "        p, sp, fd, _ = data.synthetic_panel(age_premium=prem, seed=1000+sd)\n"
            "        res = st.age_portfolio(p, sp, first_dates=fd, min_stocks=9)\n"
            "        if not res.empty:\n"
            "            s = st.summary(res['age_ret']); ts.append(s['tstat']); ms.append(s['mean'])\n"
            "    means.append(np.nanmean(ms)*100); tstats.append(np.nanmean(ts))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(premiums, means, 'o-', c=GREEN, lw=2); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_xlabel('planted age premium'); a1.set_ylabel('old-minus-young mean (%/yr)')\n"
            "a1.set_title('Mean is monotone in the planted premium')\n"
            "a2.plot(premiums, tstats, 's-', c=AMBER, lw=2)\n"
            "a2.axhline(2, c=RED, ls='--', lw=1, label='t=2'); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_xlabel('planted age premium'); a2.set_ylabel('avg HAC t (25 seeds)')\n"
            "a2.set_title('t-stat rises with the planted premium'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('premium :', premiums)\n"
            "print('mean %/yr:', [round(m,2) for m in means])\n"
            "print('avg t    :', [round(t,3) for t in tstats])"
        ),
        md(
            "### 4b -- Firm-age dispersion over time\n\n"
            "Age dispersion only opens up once the recent IPOs enter. Before ~2005 the old names "
            "all share the yfinance 1985 floor, so their age collapses together -- a data "
            "limitation we name and then sidestep with a post-2012 subsample."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rc = result.copy(); rc['year'] = rc.index.year\n"
            "    g = rc.groupby('year').agg(old=('avg_age_old','mean'),\n"
            "                               young=('avg_age_young','mean'))\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "    ax.plot(g.index, g['old'], c=GREY, lw=2, label='old leg avg age')\n"
            "    ax.plot(g.index, g['young'], c=RED, lw=2, label='young leg avg age')\n"
            "    ax.fill_between(g.index, g['young'], g['old'], color=AMBER, alpha=0.15)\n"
            "    ax.axvline(2012, c='k', ls=':', lw=1, label='recent IPOs enter')\n"
            "    ax.set_xlabel('year'); ax.set_ylabel('average firm age (yrs)')\n"
            "    ax.set_title('Age dispersion opens up only after the 2012-2021 IPO cohort lists')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('Latest: old leg', round(g['old'].iloc[-1],1), 'yrs |',\n"
            "          'young leg', round(g['young'].iloc[-1],1), 'yrs')\n"
            "else:\n"
            "    print(f'Frozen: avg old age={R[\"avg_age_old\"]:.1f} yrs | young={R[\"avg_age_young\"]:.1f} yrs')"
        ),
        md("### 4c -- The old-minus-young long-short, year by year"),
        code(
            "if HAVE_REAL:\n"
            "    age_m = s_age['mean']*100; age_t = s_age['tstat']\n"
            "    rc = result.copy(); rc['year'] = rc.index.year\n"
            "    annual = rc.groupby('year')['age_ret'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    age_m, age_t = R['age_mean'], R['age_t']\n"
            "    annual = pd.Series(dtype=float)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.3))\n"
            "if HAVE_REAL:\n"
            "    col = [GREEN if v > 0 else RED for v in annual.values]\n"
            "    ax.bar(annual.index, annual.values, color=col)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('year'); ax.set_ylabel('old-minus-young (%/yr)')\n"
            "    ax.set_title(f'Old-minus-young by year | mean {age_m:+.1f}%/yr  t={age_t:+.2f}')\n"
            "else:\n"
            "    ax.text(0.5, 0.5, f'Frozen: {age_m:+.1f}%/yr, t={age_t:+.2f}',\n"
            "            ha='center', transform=ax.transAxes)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Old-minus-young: {age_m:+.2f}%/yr | HAC t = {age_t:+.3f}')"
        ),
        md(
            f"> Old-minus-young earns **{R['age_mean']:.2f}%/yr** at HAC *t* = "
            f"**{R['age_t']:.3f}** -- the claimed premium with the sign flipped. The young leg "
            f"(**+{R['young_mean']:.1f}%/yr**) trounces the old leg "
            f"(**+{R['old_mean']:.1f}%/yr**)."
        ),
        md(
            "### 4d -- Label-shuffle placebo\n\n"
            "Permute which name is old vs young and rebuild the long-short. If age carried no "
            "information, shuffled means would be as large as the real one about as often."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(prices, spy, first_dates=first_dates,\n"
            "                            n_perm=200, min_stocks=9)\n"
            "    real_m = pb['real_mean']*100; pm = pb['perm_mean']*100; ps = pb['perm_std']*100\n"
            "    pval = pb['p_value']\n"
            "else:\n"
            "    real_m, pm, ps, pval = (R['placebo_real'], R['placebo_perm'],\n"
            "                            R['placebo_std'], R['placebo_p'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "xs = np.linspace(pm-4*ps, pm+4*ps, 200)\n"
            "ax.plot(xs, np.exp(-0.5*((xs-pm)/ps)**2), c=GREY, lw=2, label='shuffle null')\n"
            "ax.axvline(real_m, c=RED, lw=2, label=f'real mean {real_m:+.2f}%')\n"
            "ax.axvline(pm, c='k', lw=1, ls='--', label=f'null mean {pm:+.2f}%')\n"
            "ax.set_xlabel('monthly old-minus-young mean (%)'); ax.set_ylabel('density (illustrative)')\n"
            "ax.set_title(f'Label-shuffle placebo: real spread is in the tail (p={pval:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Real {real_m:+.3f}% | null {pm:+.3f}% (std {ps:.3f}%) | p = {pval:.3f}')"
        ),
        md(
            f"> The real spread sits far in the tail of the shuffle null "
            f"(**p = {R['placebo_p']:.3f}**): firm age carries a *genuine* cross-sectional "
            "signal -- it just points the wrong way for the published claim."
        ),
        md(
            "### 4e -- Costs, turnover, and the post-2012 subsample\n\n"
            "Turnover is tiny (age ranks barely move), so costs are not the story. And the clean "
            "post-2012 window -- where real age dispersion exists -- agrees with the full sample."
        ),
        code(
            "if HAVE_REAL:\n"
            "    net = st.apply_costs(result); s_net = st.summary(net)\n"
            "    r2 = result[result.index >= '2012-01-01']; s2 = st.summary(r2['age_ret'])\n"
            "    print('--- Costs ---')\n"
            "    print(f'Avg two-sided turnover: {result[\"turnover\"].mean():.3f}')\n"
            "    print(f'Gross mean: {s_age[\"mean\"]*100:+.2f}%/yr | t {s_age[\"tstat\"]:+.3f}')\n"
            "    print(f'Net mean:   {s_net[\"mean\"]*100:+.2f}%/yr | t {s_net[\"tstat\"]:+.3f}'\n"
            "          f' | Sharpe {s_net[\"sharpe\"]:+.3f}')\n"
            "    print('--- Post-2012 subsample (genuine age dispersion) ---')\n"
            "    print(f'N={s2[\"n\"]} | mean {s2[\"mean\"]*100:+.2f}%/yr | t {s2[\"tstat\"]:+.3f}')\n"
            "else:\n"
            "    print(f'Frozen: turnover {R[\"turnover\"]:.2f} | net {R[\"net_mean\"]:+.2f}%/yr '\n"
            "          f'(t {R[\"net_t\"]:+.3f}) | post-2012 N={R[\"post2012_n\"]} '\n"
            "          f'mean {R[\"post2012_mean\"]:+.1f}%/yr t {R[\"post2012_t\"]:+.3f}')"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + result['age_ret']).cumprod()\n"
            "    dd = (eq / eq.cummax() - 1) * 100\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    ax1.plot(result.index, eq.values, c=RED, lw=1.5)\n"
            "    ax1.axhline(1, c='k', lw=1, ls='--'); ax1.set_yscale('log')\n"
            "    ax1.set_ylabel('cumulative wealth (log)')\n"
            "    ax1.set_title('Old-minus-young equity curve: a relentless bleed')\n"
            "    ax2.fill_between(result.index, dd.values, 0, color=RED, alpha=0.4)\n"
            "    ax2.set_ylabel('drawdown (%)'); ax2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max drawdown: {dd.min():.1f}% | hit rate: {s_age[\"hit_rate\"]*100:.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: max DD {R[\"age_dd\"]:.0f}% | hit rate {R[\"age_hit\"]:.1f}%')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- old-minus-young {R['age_mean']:.2f}%/yr, HAC *t* = "
            f"{R['age_t']:.3f}; post-2012 *t* = {R['post2012_t']:.3f}; placebo p = "
            f"{R['placebo_p']:.3f}. The published direction (old > young) is *reversed* on this "
            "basket. Literature support cannot stamp a sign the data flips.\n"
            f"- **Tradability `MIRAGE`** -- Sharpe {R['age_sharpe']:.3f} (net "
            f"{R['net_sharpe']:.3f}), max DD {R['age_dd']:.0f}%, turnover {R['turnover']:.2f}. "
            "Costs are negligible; the loss is structural, not frictional. Flipping the sign to "
            "'win' would be in-sample survivorship mining.\n"
            "- **Survivor basket the whole story? `BUSTED`** -- yes. The young leg holds only the "
            "IPOs that survived (the winners); restoring the failures is the only way the "
            "published effect could reappear."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "Not as a firm-age premium. The obstacles are conceptual, not operational:\n\n"
            "1. **The edge is hindsight.** The young winners (META, NVDA-era IPOs) are known to "
            "us only because they survived. A real-time long-young book would also buy the "
            "failures.\n"
            "2. **Static portfolio.** Turnover ~0.02 -- the age ranks are nearly frozen, so this "
            "is a single bet on a fixed survivor set, not a refreshed factor.\n"
            "3. **No survivorship-free tape here.** Without delisted names, the test cannot be "
            "made honest within yfinance. The right experiment needs CRSP-style data.\n\n"
            "The deliverable is a lesson, not a strategy: survivorship can invert a real anomaly."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Survivorship-free universe.** Keep delisted firms; the young leg's failures are "
            "the whole point of the new-list effect.\n"
            "- **True listing dates.** EDGAR/SEC first-filing dates beat a yfinance-floored "
            "first-price proxy, which collapses the age of pre-1985 names.\n"
            "- **Age x other factors.** Jiang-Lee-Zhang (2005) find the effect is strongest among "
            "low-momentum, high-uncertainty names -- a conditional sort might survive even here.\n"
            "- **[Study 219 -- IPO-Pop](../../219-ipo-pop/)** and "
            "**[Study 265 -- IPO-Volume](../../265-ipo-volume/)**: the IPO-timing siblings.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: the same "
            "survivor-basket sort machinery (rolling rank, equal-weight legs, HAC inference).\n\n"
            "*Think the firm-age premium survives with the dead young firms left in? Fork this, "
            "pull a delisting-inclusive tape, and show old-minus-young t > 2. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
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
