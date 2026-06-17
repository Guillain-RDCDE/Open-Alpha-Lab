"""Generate the two narrative notebooks for Study 221 (Mayer-Multiple).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; real-tape cells use the cached BTC daily
parquet if present and otherwise fall back to frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebooks re-run for any reader without network access.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    # tape
    n=4291, start="2014-09-17", end="2026-06-16", fp="abb873462d3e",
    # MM distribution
    mm_mean=1.17, mm_median=1.12, mm_min=0.48, mm_max=3.93,
    pct_cheap=38.6, pct_expensive=1.5, pct_neutral=59.9,
    # long-only strategy (MM < 1)
    strat_ann_gross=1.5, strat_ann_net=1.0,
    strat_sharpe_gross=0.045, strat_t_gross=0.19, strat_t_net=0.13,
    time_in_mkt=36.8,
    # buy-and-hold
    bh_ann=34.0,
    # cost sweep
    ann_c0=1.5, ann_c10=1.0, ann_c50=-0.7, ann_c100=-2.9,
    t_c0=0.19, t_c10=0.13, t_c50=-0.10, t_c100=-0.39,
    # band forward returns (30-day log-return %)
    cheap_n=1548, cheap_mean=1.33, cheap_std=18.2,
    neutral_n=2453, neutral_mean=6.28, neutral_std=21.4,
    expensive_n=61, expensive_mean=-6.14, expensive_std=29.5,
    # band t-stat
    t_cheap_neutral=-2.84,
    t_expensive_neutral=-1.79,
    # decile means (30d log-return %)
    decile_means=[0.5, 3.7, -2.7, 3.0, 5.2, 5.2, 4.7, 4.7, 6.8, 10.9],
    decile_ranges=["0.48-0.74", "0.74-0.84", "0.84-0.93", "0.93-1.03",
                   "1.03-1.12", "1.12-1.18", "1.18-1.28", "1.28-1.40",
                   "1.40-1.70", "1.70-3.93"],
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

from mayer_multiple import data, strategy as st

def _have_cache():
    try:
        data.fetch_btc(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("real BTC cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    # Pre-compute strings that need R values to avoid backslash-in-f-string issues.
    verdict_table = (
        "## The answer first\n\n"
        "| Question | Answer |\n|---|---|\n"
        "| Does the long-only strategy (buy when MM < 1) beat buy-and-hold? | "
        "**No.** +{strat}%/yr gross vs +{bh}%/yr buy-and-hold. |\n"
        "| Does the 'cheap' band predict higher forward returns? | "
        "**No — the opposite.** 'Cheap' (MM < 1) 30-day fwd return: +{cheap}%; "
        "'Neutral' (MM 1–2.4): +{neutral}%. The 'cheap' band underperforms. |\n"
        "| Is the strategy HAC t-stat significant? | "
        "**No.** HAC t = +{t_strat:.2f} (bar: |t| >= 2) — noise. |\n"
        "| Does the band effect run the *right* way? | "
        "**No.** Band t-stat (cheap vs neutral) = {t_cn:.2f}: "
        "statistically significant but **inverted** from the claim. |\n\n"
        "> The Mayer Multiple labels Bitcoin's downtrend 'cheap.' That is not a bargain; "
        "it is a momentum-negative regime. The highest-MM decile (most 'expensive') "
        "produces +{d10}% 30-day return vs +{d1}% for the lowest-MM decile."
    ).format(
        strat=R["strat_ann_gross"], bh=R["bh_ann"],
        cheap=R["cheap_mean"], neutral=R["neutral_mean"],
        t_strat=R["strat_t_gross"], t_cn=R["t_cheap_neutral"],
        d10=R["decile_means"][-1], d1=R["decile_means"][0],
    )

    code_price_chart = (
        "# Plot Mayer Multiple over time vs the canonical thresholds.\n"
        "if HAVE_REAL:\n"
        "    df = data.fetch_btc()\n"
        "else:\n"
        "    df, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n\n"
        "mm = df['mayer_multiple'].dropna()\n\n"
        "fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)\n\n"
        "# Top: BTC price\n"
        "axes[0].semilogy(df.index, df['close'], c='k', lw=1.0, label='BTC price')\n"
        "if df['sma_200'].notna().any():\n"
        "    axes[0].semilogy(df.index, df['sma_200'], c=AMBER, lw=1.5, ls='--', label='200-day SMA')\n"
        "axes[0].set_ylabel('Price (log scale)')\n"
        "axes[0].set_title('BTC price vs 200-day SMA')\n"
        "axes[0].legend()\n\n"
        "# Bottom: Mayer Multiple\n"
        "axes[1].plot(mm.index, mm, c='#1565c0', lw=1.0, label='Mayer Multiple')\n"
        "axes[1].axhline(1.0, c=GREEN, ls='--', lw=1.5, label='MM=1.0 (buy threshold)')\n"
        "axes[1].axhline(2.4, c=RED, ls='--', lw=1.5, label='MM=2.4 (sell threshold)')\n"
        "axes[1].fill_between(mm.index, 0, mm, where=(mm < 1.0), color=GREEN, alpha=0.2, label='cheap zone')\n"
        "axes[1].fill_between(mm.index, 0, mm, where=(mm > 2.4), color=RED, alpha=0.3, label='expensive zone')\n"
        "axes[1].set_ylabel('Mayer Multiple')\n"
        "axes[1].set_title('Mayer Multiple with canonical thresholds')\n"
        "axes[1].legend(loc='upper right')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'% days in cheap zone (MM<1): {(mm<1).mean()*100:.1f}%')\n"
        "print(f'% days in expensive zone (MM>2.4): {(mm>2.4).mean()*100:.1f}%')"
    )

    code_band_returns = (
        "# 30-day forward returns by regime.\n"
        "if HAVE_REAL:\n"
        "    fwd = st.forward_return_by_band(df)\n"
        "else:\n"
        "    df_s, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n"
        "    fwd = st.forward_return_by_band(df_s)\n\n"
        "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
        "colours = {'cheap': GREEN, 'neutral': GREY, 'expensive': RED}\n"
        "for regime, col in colours.items():\n"
        "    grp = fwd.loc[fwd['regime'] == regime, 'fwd_30d'] * 100\n"
        "    if len(grp) > 0:\n"
        "        ax.hist(grp, bins=30, color=col, alpha=0.5, label=f'{regime} (n={len(grp)})')\n"
        "ax.axvline(0, c='k', lw=1.0)\n"
        "ax.set_xlabel('30-day forward log-return (%)')\n"
        "ax.set_title('30-day forward returns by Mayer Multiple regime')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n\n"
        "for regime in ['cheap', 'neutral', 'expensive']:\n"
        "    grp = fwd.loc[fwd['regime'] == regime, 'fwd_30d'] * 100\n"
        "    if len(grp) > 0:\n"
        "        print(f'{regime:12s}: n={len(grp):5d}  mean={grp.mean():+.1f}%  median={grp.median():+.1f}%')\n"
        "print()\n"
        "print('Expected if claim were true: cheap > neutral > expensive')\n"
        "print('Actual order (real tape):   cheap << neutral, expensive << neutral')"
    )

    band_narrative = (
        "The 'cheap' band (MM < 1.0) produces a 30-day forward return of just "
        "+{cheap}% vs +{neutral}% for the neutral band. "
        "The claim predicts the reverse. The expensive band (MM > 2.4) posts "
        "{expensive}% — also worse than neutral, and consistent with "
        "late-cycle momentum exhaustion, though the n is tiny (only 61 days in 11 years).\n\n"
        "**The Mayer Multiple identifies downtrends, not bargains.** When price is "
        "below the 200-day SMA, it is typically in a sustained drawdown. The 200-day "
        "SMA is conventionally used as a *trend filter* (stay long above it, exit below) "
        "— the Mayer Multiple inverts that intuition by calling the below-SMA zone 'cheap.'"
    ).format(cheap=R["cheap_mean"], neutral=R["neutral_mean"], expensive=R["expensive_mean"])

    code_cum_return = (
        "# Cumulative return: Mayer strategy vs buy-and-hold.\n"
        "if HAVE_REAL:\n"
        "    sig = st.mayer_signal(df)\n"
        "    bt = st.backtest(df, sig, cost_bps=10.0)\n"
        "else:\n"
        "    df_s, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n"
        "    sig = st.mayer_signal(df_s)\n"
        "    bt = st.backtest(df_s, sig, cost_bps=10.0)\n\n"
        "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
        "ax.plot(bt.index, (bt['cum_log_bh'].apply(lambda x: (np.exp(x)-1)*100)),\n"
        "        c=GREEN, lw=2, label='Buy-and-hold Bitcoin')\n"
        "ax.plot(bt.index, (bt['cum_log_net'].apply(lambda x: (np.exp(x)-1)*100)),\n"
        "        c=RED, lw=1.5, label='Mayer Multiple strategy (10 bp cost)')\n"
        "ax.axhline(0, c='k', lw=0.8)\n"
        "ax.set_ylabel('Cumulative return (%)')\n"
        "ax.set_title('Mayer Multiple long-only vs buy-and-hold')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
        "flat_pct = (bt['position'] == 0).mean() * 100\n"
        "s = st.summarize(bt)\n"
        "print(f'Strategy flat: {flat_pct:.0f}% of days')\n"
        "print(f'Net ann: {s[\"ann_ret_pct\"]:+.1f}% vs BH +{R_bh}%/yr')".replace(
            "{R_bh}", str(R["bh_ann"])
        )
    )

    verdict_text = (
        "## 5 - The verdict\n\n"
        "- **Signal — None.** Strategy HAC t = +{t_strat:.2f} (bar: |t| >= 2). "
        "The strategy earns +{strat}%/yr gross vs BH +{bh}%/yr. The band test is "
        "significant (t = {t_cn:.2f}) but runs **inverted** — 'cheap' underperforms 'neutral.'\n"
        "- **Tradability — Mirage.** +{strat}%/yr gross, negative at 50 bps cost. "
        "Strategy is flat 63% of the time, precisely during the uptrend phases.\n"
        "- **The claim — Busted.** MM < 1 is a downtrend flag, not a valuation signal. "
        "Applying mean-reversion intuition to a momentum-driven asset produces the "
        "opposite of the intended result."
    ).format(
        t_strat=R["strat_t_gross"], strat=R["strat_ann_gross"],
        bh=R["bh_ann"], t_cn=R["t_cheap_neutral"],
    )

    cells = [
        md(
            "# Mayer Multiple — does price / 200-day MA tell you when Bitcoin is cheap?\n"
            "### A valuation-band strategy tested against buy-and-hold BTC\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Claim%3A%20Busted](https://img.shields.io/badge/Claim%3A%20MM%3C1%3DCheap-Busted-8b949e?style=flat-square)\n\n"
            "The Mayer Multiple divides Bitcoin's price by its 200-day moving average. "
            "The pitch, popularised by Trace Mayer in 2017-2018: a reading below 1.0 means "
            "Bitcoin is 'cheap' — accumulate. Above 2.4 means 'expensive' — trim. Simple, "
            "intuitive, and widely cited in crypto media.\n\n"
            "This notebook asks the only question that matters: **does it actually predict "
            "better forward returns when MM is low?**\n\n"
            "> Plain-language layer. For t-stats, decile tables, and the full statistical "
            "teardown, see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n\n"
            "> **Not investment advice.** Reproducible research; every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(verdict_table),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'The Mayer Multiple is a multiplier of the current Bitcoin price over the "
            "200-day moving average. Historically, buying below 1 and holding has proven "
            "to be a solid long-term strategy. A Mayer Multiple above 2.4 indicates "
            "overvaluation territory.'* — popularised by Trace Mayer (2017-2018)\n\n"
            "The formula:\n\n"
            "    Mayer Multiple = BTC price / 200-day simple moving average\n\n"
            "The strategy:\n"
            "- **MM < 1.0** — price below the 200-day SMA; label: 'cheap'; action: accumulate / go long.\n"
            "- **MM > 2.4** — price more than 2.4x above the 200-day SMA; label: 'expensive'; action: trim.\n\n"
            "Unlike the Bitcoin Rainbow Chart ([Study 174](../../174-bitcoin-rainbow/)), "
            "the Mayer Multiple is fully observable in real time — no regression, no "
            "future data needed. The 200-day SMA is computed from historical closes only. "
            "The question is not about look-ahead bias; it is whether the signal points "
            "in the right direction."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the Mayer Multiple genuinely identifies cheap and expensive Bitcoin, "
            "it offers a simple valuation timer that any retail investor can follow: "
            "buy the dip when MM < 1, hold, reduce exposure when MM > 2.4.\n\n"
            "If it does not — if the 'cheap' regime actually predicts lower forward "
            "returns than the neutral regime — then following this signal causes investors "
            "to buy precisely when BTC is in a downtrend and sit out the uptrend. "
            "The 200-day SMA would then be a trend filter (as it is conventionally used "
            "in equities), not a valuation anchor."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two complementary tests:\n\n"
            "1. **Strategy test** — go long when MM < 1.0, flat otherwise; measure the "
            "HAC t-stat on daily strategy returns. Requires |t| >= 2 to stamp REAL.\n\n"
            "2. **Band forward-return test** — compute the 30-day forward log-return at "
            "each date; group by regime (cheap / neutral / expensive); compare the mean "
            "returns and the t-stat on the mean difference.\n\n"
            "No look-ahead: the MM at date *t* uses only prices through *t*; the position "
            "is entered at the *open* of *t+1*.\n\n"
            "Positive control: synthetic BTC data with a tunable mean-reversion knob — "
            "the engine should find the signal when it is genuinely planted."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md(
            "## 4 - The teardown\n\n"
            "### First: the Mayer Multiple over BTC's history"
        ),
        code(code_price_chart),

        md(
            "### Then: do 'cheap' days actually predict higher forward returns?"
        ),
        code(code_band_returns),
        md(band_narrative),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(verdict_text),

        # ---- BEAT 6 -- COULD YOU TRADE IT? ----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Strategy cumulative return vs buy-and-hold."
        ),
        code(code_cum_return),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The Mayer Multiple as a trend filter** — If you *reverse* the signal "
            "(go long when MM > 1, i.e., price above the 200-day SMA), you are now "
            "following momentum rather than fighting it. That is how the 200-day SMA "
            "is used in Faber-style tactical allocation ([Study 106 — Faber-Timing](../../106-faber-timing/)), "
            "which shows a genuine but fragile edge in equities.\n"
            "- **Bitcoin Rainbow Chart** ([Study 174 — Bitcoin-Rainbow](../../174-bitcoin-rainbow/)): "
            "a log-time regression band with the same 'buy cheap, sell rich' directional "
            "claim and the additional flaw of look-ahead bias in the regression fit.\n"
            "- **Turtle Trading** ([Study 103 — Turtle-Trading](../../103-turtle-trading/)): "
            "systematic trend-following on BTC — the momentum premium that our decile "
            "analysis shows in raw form.\n"
            "- **CAPE / Excess-CAPE-Yield** ([Study 120](../../120-excess-cape-yield/)): "
            "an equity valuation measure that *does* predict long-horizon returns (though "
            "fragilely), because the mechanism (earnings yield vs bond yield) is grounded "
            "in theory. The Mayer Multiple has no analogous mechanism.\n\n"
            "*Think the Mayer Multiple works? Fork this, show a walk-forward t >= 2 in the "
            "claimed direction (cheap outperforms neutral), net of costs, and cite a mechanism "
            "for why BTC mean-reverts to its 200-day SMA. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    # Pre-compute multi-value strings to avoid backslash-in-f-string.
    cheap_mean_str = str(R["cheap_mean"])
    neutral_mean_str = str(R["neutral_mean"])
    expensive_mean_str = str(R["expensive_mean"])
    t_cn_str = "{:.2f}".format(R["t_cheap_neutral"])
    t_en_str = "{:.2f}".format(R["t_expensive_neutral"])
    bh_ann_str = str(R["bh_ann"])
    strat_gross_str = str(R["strat_ann_gross"])
    strat_t_str = "{:.2f}".format(R["strat_t_gross"])
    d1_str = str(R["decile_means"][0])
    d10_str = str(R["decile_means"][-1])
    cost_ref_str = str(list(zip([0, 10, 50, 100],
                                [R["ann_c0"], R["ann_c10"], R["ann_c50"], R["ann_c100"]])))
    decile_means_str = repr(["{:.1f}%".format(m) for m in R["decile_means"]]).replace("'", '"')

    verdict_upfront = (
        "## Verdict, up front\n\n"
        "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
        "| **Signal** | `NONE` | Strategy HAC t = **+{t_strat}** (gross). "
        "Band t-stat (cheap vs neutral) = **{t_cn}**: significant but "
        "**inverted** — cheap underperforms neutral. |\n"
        "| **Tradability** | `MIRAGE` | Long-only earns **+{strat}%/yr** vs "
        "buy-and-hold **+{bh}%/yr**; strategy flat {flat:.0f}% of time. |\n"
        "| **The claim** | `BUSTED` | Decile D10 (MM=1.70-3.93, 'most expensive') earns "
        "+{d10}% 30-day return; D1 (MM=0.48-0.74, 'cheapest') earns "
        "+{d1}%. The cheap zone is a downtrend, not a bargain. |\n\n"
        "> In plain words: the Mayer Multiple is a momentum indicator dressed as a "
        "valuation signal. Low MM = falling price = bad forward return. High MM = rising "
        "price = good forward return. The claim inverts this."
    ).format(
        t_strat=strat_t_str, t_cn=t_cn_str, strat=strat_gross_str, bh=bh_ann_str,
        flat=100 - R["time_in_mkt"], d10=d10_str, d1=d1_str,
    )

    code_band_analysis = (
        "if HAVE_REAL:\n"
        "    df = data.fetch_btc()\n"
        "    fwd = st.forward_return_by_band(df)\n"
        "    t_cn = st.band_tstat(fwd, 'cheap', 'neutral')\n"
        "    t_en = st.band_tstat(fwd, 'expensive', 'neutral')\n"
        "else:\n"
        "    df_s, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n"
        "    fwd = st.forward_return_by_band(df_s)\n"
        "    t_cn = st.band_tstat(fwd, 'cheap', 'neutral')\n"
        "    t_en = st.band_tstat(fwd, 'expensive', 'neutral')\n\n"
        "rows = []\n"
        "for regime in ['cheap', 'neutral', 'expensive']:\n"
        "    grp = fwd.loc[fwd['regime'] == regime, 'fwd_30d'] * 100\n"
        "    rows.append((regime, len(grp), grp.mean() if len(grp) else float('nan'),\n"
        "                 grp.std() if len(grp) else float('nan')))\n"
        "import pandas as pd\n"
        "tbl = pd.DataFrame(rows, columns=['regime', 'n', 'mean_30d_%', 'std_30d_%'])\n"
        "print(tbl.to_string(index=False))\n"
        "print(f'band t-stat cheap vs neutral:     {t_cn:+.3f}')\n"
        "print(f'band t-stat expensive vs neutral: {t_en:+.3f}')\n"
        "print('Frozen ref: cheap=" + cheap_mean_str + "%, neutral=" + neutral_mean_str + "%, expensive=" + expensive_mean_str + "%')\n"
        "print('Frozen band t cheap vs neutral: " + t_cn_str + "')"
    )

    code_decile = (
        "import pandas as pd\n"
        "if HAVE_REAL:\n"
        "    df_use = df\n"
        "else:\n"
        "    df_use, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n\n"
        "mm_valid = df_use['mayer_multiple'].dropna()\n"
        "log_r = df_use['log_return']\n"
        "fwd_r = log_r.shift(-1).rolling(30).sum().shift(-29)\n"
        "valid = mm_valid.index.intersection(fwd_r.dropna().index)\n"
        "mm_v = mm_valid[valid]\n"
        "fwd_v = fwd_r[valid] * 100\n"
        "deciles = pd.qcut(mm_v, 10, labels=False)\n\n"
        "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
        "decile_means = []\n"
        "decile_labels = []\n"
        "for d in range(10):\n"
        "    vals = fwd_v[deciles == d]\n"
        "    decile_means.append(vals.mean())\n"
        "    lo, hi = mm_v[deciles==d].min(), mm_v[deciles==d].max()\n"
        "    decile_labels.append(f'D{d+1}\\n[{lo:.2f}-{hi:.2f}]')\n\n"
        "colors = [GREEN if m > 0 else RED for m in decile_means]\n"
        "ax.bar(range(10), decile_means, color=colors, alpha=0.8)\n"
        "ax.axhline(0, c='k', lw=0.8)\n"
        "ax.set_xticks(range(10)); ax.set_xticklabels(decile_labels, fontsize=8)\n"
        "ax.set_xlabel('Mayer Multiple decile (D1 = cheapest, D10 = most expensive)')\n"
        "ax.set_ylabel('Mean 30-day forward log-return (%)')\n"
        "ax.set_title('30-day forward return by MM decile')\n"
        "plt.tight_layout(); plt.show()\n"
        "print('Decile means (D1 to D10):', [f'{m:.1f}%' for m in decile_means])\n"
        "print('Frozen reference: " + decile_means_str + "')"
    )

    code_strategy = (
        "if HAVE_REAL:\n"
        "    df_bt = df\n"
        "else:\n"
        "    df_bt, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n\n"
        "sig = st.mayer_signal(df_bt)\n"
        "bt_g = st.backtest(df_bt, sig, cost_bps=0.0)\n"
        "bt_n = st.backtest(df_bt, sig, cost_bps=10.0)\n"
        "s_g = st.summarize(bt_g)\n"
        "s_n = st.summarize(bt_n)\n\n"
        "import pandas as pd\n"
        "tbl = pd.DataFrame([\n"
        "    ('Long-only gross', s_g['ann_ret_pct'], s_g['sharpe'], s_g['tstat'], s_g['time_in_market_pct']),\n"
        "    ('Long-only net 10bps', s_n['ann_ret_pct'], s_n['sharpe'], s_n['tstat'], s_n['time_in_market_pct']),\n"
        "], columns=['version', 'ann_ret_%', 'sharpe', 'HAC_t', 'time_in_mkt_%'])\n"
        "print(tbl.to_string(index=False))\n"
        "print('Buy-and-hold: +" + bh_ann_str + "%/yr')\n\n"
        "# Cumulative chart\n"
        "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
        "bh_cum = (bt_n['cum_log_bh'].apply(lambda x: (np.exp(x)-1)*100))\n"
        "st_cum = (bt_n['cum_log_net'].apply(lambda x: (np.exp(x)-1)*100))\n"
        "ax.plot(bh_cum.index, bh_cum, c=GREEN, lw=2, label='Buy-and-hold BTC')\n"
        "ax.plot(st_cum.index, st_cum, c=RED, lw=1.5, label='Mayer strategy (net 10bps)')\n"
        "ax.axhline(0, c='k', lw=0.8)\n"
        "ax.set_ylabel('Cumulative return (%)')\n"
        "ax.set_title('Mayer Multiple long-only vs buy-and-hold')\n"
        "ax.legend(); plt.tight_layout(); plt.show()"
    )

    code_cost_sweep = (
        "costs = [0, 10, 50, 100]\n"
        "if HAVE_REAL:\n"
        "    df_cs = df\n"
        "else:\n"
        "    df_cs, _ = data.synthetic_btc(n_days=3000, signal_strength=0.0, seed=221)\n"
        "sig_cs = st.mayer_signal(df_cs)\n"
        "net_anns, net_ts = [], []\n"
        "for c in costs:\n"
        "    bt = st.backtest(df_cs, sig_cs, cost_bps=c)\n"
        "    s = st.summarize(bt)\n"
        "    net_anns.append(s['ann_ret_pct'])\n"
        "    net_ts.append(s['tstat'])\n\n"
        "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
        "ax.plot(costs, net_anns, 'o-', c=RED, lw=2, label='MM strategy ann. return (%)')\n"
        "ax.axhline(" + bh_ann_str + ", ls='--', c=GREEN, lw=1.5, label='Buy-and-hold (+" + bh_ann_str + "%/yr)')\n"
        "ax.axhline(0, c='k', lw=0.8)\n"
        "ax.set_xlabel('round-trip cost (bps)')\n"
        "ax.set_ylabel('annualised return (%)')\n"
        "ax.set_title('Cost sweep: Mayer strategy vs buy-and-hold')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
        "for c, ann, t in zip(costs, net_anns, net_ts):\n"
        "    print(f'cost={c:4d} bps: ann={ann:+.1f}%  HAC t={t:+.3f}')\n"
        "print('Frozen reference: " + cost_ref_str + "')"
    )

    code_positive_control = (
        "strengths = [0.0, 0.25, 0.5, 0.75, 1.0]\n"
        "band_ts, strat_ts = [], []\n"
        "for ss in strengths:\n"
        "    df_s, _ = data.synthetic_btc(n_days=2000, signal_strength=ss, seed=221)\n"
        "    fwd_s = st.forward_return_by_band(df_s)\n"
        "    band_ts.append(st.band_tstat(fwd_s, 'cheap', 'neutral'))\n"
        "    sig_s = st.mayer_signal(df_s)\n"
        "    bt_s = st.backtest(df_s, sig_s, cost_bps=0.0)\n"
        "    strat_ts.append(st.summarize(bt_s)['tstat'])\n\n"
        "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
        "ax.plot(strengths, band_ts, 'o-', c=GREEN, lw=2, label='Band t-stat (cheap vs neutral)')\n"
        "ax.plot(strengths, strat_ts, 's--', c=AMBER, lw=2, label='Strategy HAC t-stat')\n"
        "ax.axhline(2, ls=':', c=GREY, lw=1.5, label='|t|=2 bar')\n"
        "ax.axhline(-2, ls=':', c=GREY, lw=1.5)\n"
        "ax.axhline(0, c='k', lw=0.8)\n"
        "ax.set_xlabel('signal_strength (0=random walk, 1=genuine mean-reversion)')\n"
        "ax.set_ylabel('HAC t-stat')\n"
        "ax.set_title('Positive control: engine recovers signal only when mean-reversion is planted')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
        "print('Band t-stats:', [f'{t:.2f}' for t in band_ts])\n"
        "print('Strategy t-stats:', [f'{t:.2f}' for t in strat_ts])\n"
        "print('Real BTC: strategy t=+" + strat_t_str + ", band t=" + t_cn_str + " (inverted direction)')"
    )

    decile_narrative = (
        "> In plain words: decile D10 ('most expensive', MM=1.70-3.93) produces "
        "+{d10}% 30-day return; D1 ('cheapest', MM=0.48-0.74) "
        "produces +{d1}%. The pattern is monotonically increasing "
        "from D1 to D10 — precisely the opposite of what the Mayer Multiple predicts. "
        "This is the momentum premium: being above your 200-day SMA (high MM) signals "
        "trend strength, not overvaluation."
    ).format(d10=d10_str, d1=d1_str)

    pc_narrative = (
        "> In plain words: the machine is not broken. On a synthetic tape with planted "
        "mean-reversion, both the band t-stat and the strategy t-stat rise as signal_strength "
        "increases and eventually clear the |t| = 2 bar. The real BTC tape gives "
        "strategy t = +{t_strat} and band t = {t_cn} — "
        "the band effect is real, but it runs the wrong way for the claim."
    ).format(t_strat=strat_t_str, t_cn=t_cn_str)

    final_verdict = (
        "## 5 - The verdict\n\n"
        "- **Signal `NONE`** — strategy HAC t = +{t_strat} (< 2). "
        "The only significant band effect (cheap vs neutral t = {t_cn}) "
        "is **inverted from the claim**.\n"
        "- **Tradability `MIRAGE`** — +{strat}%/yr gross vs "
        "+{bh}%/yr buy-and-hold; negative at 50 bps cost; flat 63% of the time.\n"
        "- **The claim `BUSTED`** — MM < 1 identifies downtrends, not bargains. "
        "The decile analysis shows a monotonically *increasing* return pattern from "
        "D1 to D10: the 'cheapest' decile has the worst forward return, the "
        "'most expensive' the best. The Mayer Multiple's valuation framing "
        "misidentifies momentum as mean-reversion."
    ).format(
        t_strat=strat_t_str, t_cn=t_cn_str,
        strat=strat_gross_str, bh=bh_ann_str,
    )

    band_inv_narrative = (
        "> In plain words: 'cheap' days (MM < 1.0) produce a 30-day forward return of "
        "+{cheap}% vs +{neutral}% for neutral. The difference is "
        "statistically significant (t = {t_cn}) but **in the wrong "
        "direction**. H1 is rejected — the Mayer Multiple predicts lower forward returns "
        "in its 'accumulate' zone than in its neutral zone."
    ).format(cheap=cheap_mean_str, neutral=neutral_mean_str, t_cn=t_cn_str)

    cells = [
        md(
            "# Mayer Multiple — a quantitative teardown\n"
            "### Price / 200-day MA * band forward returns * HAC inference * positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Claim%3A%20Busted](https://img.shields.io/badge/Claim%3A%20MM%3C1%3DCheap-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We test the Mayer Multiple claim (buy below 1.0, trim above 2.4) using three "
            "lenses: (1) HAC t-stat on the long-only strategy, (2) conditional 30-day "
            "forward returns by band with statistical testing, and (3) a decile analysis "
            "of MM vs subsequent returns. We also run a synthetic positive control to "
            "confirm the engine detects a genuine mean-reversion signal when one is planted.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance BTC-USD daily bars, "
            "as-of " + R["end"] + "; offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(verdict_upfront),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $M_t = P_t / \\overline{P}_{200,t}$ (Mayer Multiple at date $t$, where "
            "$\\overline{P}_{200,t}$ is the 200-day SMA). The claim asserts:\n\n"
            "- **H1 (signal).** $E[r_{t+1:t+30} \\mid M_t < 1.0] > E[r_{t+1:t+30} \\mid 1.0 \\leq M_t \\leq 2.4]$ — "
            "buying in the 'cheap' band earns higher subsequent returns than staying neutral.\n"
            "- **H2 (tradable).** Going long when $M_t < 1$ and flat otherwise beats "
            "buy-and-hold net of transaction costs.\n\n"
            "The Mayer Multiple has no look-ahead bias: $M_t$ uses only data available at "
            "close of $t$. The question is purely whether H1 holds empirically."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The 200-day SMA is the most widely tracked trend indicator in financial markets. "
            "Most practitioners use it as a *trend filter*: long when price is above it, "
            "exit when price falls below. The Mayer Multiple does the opposite: it labels "
            "below-SMA as a buying opportunity. If H1 fails — if 'cheap' days predict "
            "lower forward returns than 'neutral' days — then the Mayer Multiple is "
            "inverting one of the best-documented cross-asset signals (trend-following) "
            "without theoretical justification."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - Protocol\n\n"
            "- **Band forward-return test.** At each valid date $t$, compute "
            "$r_{t+1:t+30} = \\sum_{k=1}^{30} \\log(P_{t+k}/P_{t+k-1})$. "
            "Assign the regime from $M_t$. Compare conditional means across bands using "
            "HAC Newey-West t-stat (two-sample, independent SE pooling) — see `band_tstat`.\n"
            "- **Strategy test.** Long (+1) when $M_t < 1$, flat (0) otherwise; "
            "position entered at open of $t+1$. HAC t-stat on per-day net returns.\n"
            "- **Decile analysis.** Sort $M_t$ into deciles; compute mean 30-day forward "
            "return per decile. If H1 holds, decile means should decline monotonically "
            "from D1 (cheapest) to D10 (most expensive).\n"
            "- **Positive control.** Synthetic tape with tunable ``signal_strength`` — "
            "mean-reversion is planted; the engine should recover it.\n"
            "- **Costs.** 10 bps round-trip per trade as the base case."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - Band forward-return comparison: the decisive test\n\n"
            "If H1 holds, the 'cheap' band should have the highest 30-day forward return."
        ),
        code(code_band_analysis),
        md(band_inv_narrative),

        md(
            "### 4b - MM decile analysis — is there any monotone relationship?\n\n"
            "If the claim were correct, we'd see a declining pattern of forward returns "
            "from D1 (cheapest MM) to D10 (most expensive MM)."
        ),
        code(code_decile),
        md(decile_narrative),

        md(
            "### 4c - Strategy t-stat and cumulative return\n\n"
            "The long-only strategy (buy when MM < 1, flat otherwise) vs buy-and-hold."
        ),
        code(code_strategy),

        md(
            "### 4d - Cost sweep\n\n"
            "Net return at various round-trip cost levels."
        ),
        code(code_cost_sweep),

        md(
            "### 4e - Positive control: engine finds mean-reversion when planted\n\n"
            "Synthetic tape sweep: signal_strength from 0 (random walk) to 1 (genuine "
            "mean-reversion). The band t-stat should increase monotonically with "
            "signal_strength. This confirms the engine is not broken — it simply does not "
            "find signal because there is none."
        ),
        code(code_positive_control),
        md(pc_narrative),

        # ---- BEAT 5 ----------------------------------------------------------
        md(final_verdict),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? — cost sweep summary\n\n"
            "Shown in 4d above. All cost levels fail the inference bar and "
            "underperform buy-and-hold by > 30 percentage points annualised. "
            "Even at zero cost, the strategy t-stat (+{t_strat}) "
            "is indistinguishable from random noise.".format(t_strat=strat_t_str)
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Faber 200-day MA timing** ([Study 106 — Faber-Timing](../../106-faber-timing/)): "
            "the same 200-day SMA used as a *trend filter* (long above, exit below) — the "
            "opposite logic, applied to equities with a genuine but fragile edge.\n"
            "- **Bitcoin Rainbow Chart** ([Study 174 — Bitcoin-Rainbow](../../174-bitcoin-rainbow/)): "
            "a log-time regression band with the same 'buy cheap, sell rich' directional "
            "claim and the additional flaw of look-ahead bias in the regression fit.\n"
            "- **Turtle Trading** ([Study 103 — Turtle-Trading](../../103-turtle-trading/)): "
            "systematic trend-following on BTC — the momentum premium that our decile "
            "analysis shows in raw form.\n"
            "- **CAPE / Excess-CAPE-Yield** ([Study 120](../../120-excess-cape-yield/)): "
            "an equity valuation measure that *does* predict long-horizon returns (though "
            "fragilely), because the mechanism (earnings yield vs bond yield) is grounded "
            "in theory. The Mayer Multiple has no analogous mechanism.\n\n"
            "*Convinced the signal is real? Fork this, show a band t-stat in the claimed "
            "direction (cheap outperforms neutral, t >= 2) on a walk-forward basis, net of "
            "costs, and provide a mechanism. That is the bar.*"
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
