"""Generate the two narrative notebooks for Study 513 (Size-Effect / Banz 1981 SMB).

    python notebooks/build_notebooks.py

Synthetic cells run anywhere, offline and deterministic; real-basket cells use the cached
yfinance parquets under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (the single source of truth mirroring docs/results.md, as-of 2026-06-26).
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


# Frozen real-basket headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=304, year_start=2001, year_end=2026, n_tickers=40,
    fp_prices="54a77c8e6180", fp_spy="5898b297d67e", fp_caps="17f05782d1ad",
    small_mean=13.28, small_sharpe=0.815,
    large_mean=14.36, large_sharpe=1.010,
    mkt_mean=9.72, mkt_sharpe=0.642,
    dn_mean=-1.08, dn_sharpe=-0.090, dn_t=-0.483, dn_hit=43.8, dn_dd=-57.8,
    bn_mean=-1.76, bn_sharpe=-0.146, bn_t=-0.755, beta_ratio=0.967,
    turnover=0.4, net_mean=-2.08, net_t=-0.933,
    placebo_t=-0.483, placebo_p=0.613,
    jan_mean=-1.01, jan_n=25, rest_mean=-0.01, rest_n=279, jan_welch_t=-1.681, jan_welch_p=0.103,
    early_mean=1.76, early_t=0.556, early_n=149,
    late_mean=-3.81, late_t=-1.245, late_n=155,
    sc_t_signal=2.73, sc_frac_signal=0.72, sc_t_null=0.81, sc_frac_null=0.08,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from size_effect import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    c = os.path.join(study, "_cache")
    return all(os.path.exists(os.path.join(c, f)) for f in
               ("size_prices.parquet", "size_spy.parquet", "size_caps.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy, caps = data.fetch_panel()
    book = st.add_beta_neutral(st.smb_portfolio(prices, spy, caps, min_stocks=10))
    book = data.drop_partial_last_month(book)
    s_dn = st.summary(book["smb_dn"]); s_bn = st.summary(book["smb_bn"])
    print(f"N months: {s_dn['n']} | SMB dollar-neutral: {s_dn['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_dn['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Size-Effect -- do small-caps really beat large-caps?\n"
            "### Banz (1981), the original cross-sectional anomaly, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![January concentration%3F: Busted](https://img.shields.io/badge/January--concentration%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 1981 Rolf Banz noticed something the CAPM could not explain: **small companies "
            "earned higher returns than big ones**, even after adjusting for risk. It became the "
            "founding anomaly of factor investing -- Fama and French later packaged it as "
            "**SMB (Small Minus Big)**, one of their three factors. The premium was famously "
            "concentrated in **January**.\n\n"
            "We test that claim on a 40-name basket spanning mega-caps (Apple, Microsoft) down "
            "to genuine small/mid-caps (Sensient, WD-40, J&J Snack...), ranking by market cap "
            "each month, buying the small half and shorting the big half -- naming the "
            "survivorship bias up front.\n\n"
            "> **This is the plain-language layer.** Want the cap-ranking mechanics, the "
            "label-shuffle placebo and the decay slice? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do small-caps beat big-caps? | **No, the reverse.** The small-minus-large book "
            f"earns **{R['dn_mean']:+.2f}%/yr** at HAC *t* = **{R['dn_t']:+.2f}** -- wrong sign, "
            "statistically zero. |\n"
            f"| Can a coin-flip beat it? | **Yes.** A label-shuffle placebo reproduces the real "
            f"*t* **{R['placebo_p']*100:.0f}%** of the time (*p* = {R['placebo_p']:.2f}). |\n"
            f"| Is there a January pop? | **Busted.** January was the *worst* month for the "
            f"spread (**{R['jan_mean']:+.2f}%/mo** vs ~0 the rest of the year). |\n"
            "| Survivorship problem? | **Yes, named.** Failed small-caps -- the natural long -- "
            "are gone, biasing the premium *up*; it is *still* negative. |\n\n"
            "> Banz's size effect is real in the 1936-1975 broad-universe literature, but on "
            "this survivor basket over the mega-cap decade 2001-2026 it has vanished and "
            "inverted."
        ),

        # ---- synthetic positive control ----
        md(
            "## First, prove the engine works -- a synthetic world with a *real* size premium\n\n"
            "Before trusting a flat real-tape result, we plant a known 6%/yr small-cap premium "
            "in a fake market and check the **same sort/long-short/t-stat pipeline** finds it. "
            "We average over 25 random seeds so no single lucky draw can carry the claim."
        ),
        code(
            "sc = st.synthetic_control(size_premium=0.06, n_seeds=25)\n"
            "print(f\"Planted 6%/yr premium -> mean HAC t = {sc['mean_t_signal']:+.2f} \"\n"
            "      f\"(t>2 in {sc['frac_signal_t_gt2']*100:.0f}% of seeds)\")\n"
            "print(f\"Null (0% premium)      -> mean HAC t = {sc['mean_t_null']:+.2f} \"\n"
            "      f\"(|t|>2 in {sc['frac_null_t_gt2']*100:.0f}% of seeds)\")\n"
            "print('\\nThe engine recovers a real premium and stays quiet under the null:'\n"
            "      ' the flat REAL result is about the data, not a broken harness.')"
        ),

        # ---- the real legs ----
        md(
            "## The real basket -- small vs big, 2001-2026\n\n"
            "Small-caps did *not* win. Over this large-cap decade the big half out-earned the "
            "small half on both return and Sharpe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_small = st.summary(book['small_ret']); s_large = st.summary(book['large_ret'])\n"
            "    s_mkt = st.summary(book['mkt_ret'])\n"
            "    labels = ['Small leg', 'Large leg', 'SPY (market)']\n"
            "    means = [s_small['mean']*100, s_large['mean']*100, s_mkt['mean']*100]\n"
            "    shps  = [s_small['sharpe'], s_large['sharpe'], s_mkt['sharpe']]\n"
            "else:\n"
            "    labels = ['Small leg', 'Large leg', 'SPY (market)']\n"
            "    means = [R['small_mean'], R['large_mean'], R['mkt_mean']]\n"
            "    shps  = [R['small_sharpe'], R['large_sharpe'], R['mkt_sharpe']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))\n"
            "cols = [RED, GREEN, GREY]\n"
            "a1.bar(labels, means, color=cols); a1.set_title('Return (%/yr)'); a1.set_ylabel('%/yr')\n"
            "a2.bar(labels, shps, color=cols); a2.set_title('Sharpe')\n"
            "for ax in (a1, a2):\n"
            "    ax.tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Small {means[0]:+.1f}%/yr (Sharpe {shps[0]:+.2f}) vs '\n"
            "      f'Large {means[1]:+.1f}%/yr (Sharpe {shps[1]:+.2f}) -- big won.')"
        ),

        # ---- the busted January ----
        md(
            "## The January pop -- busted\n\n"
            "Banz's premium classically showed up in **January** (tax-loss rebound). Here "
            "January was the spread's *worst* month."
        ),
        code(
            "if HAVE_REAL:\n"
            "    jan = st.january_split(book)\n"
            "    jm, rm, wp = jan['jan_mean']*100, jan['rest_mean']*100, jan['welch_p']\n"
            "else:\n"
            "    jm, rm, wp = R['jan_mean'], R['rest_mean'], R['jan_welch_p']\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar(['January', 'Other 11 months'], [jm, rm], color=[RED, GREY])\n"
            "ax.axhline(0, c='k', lw=0.8); ax.set_ylabel('mean SMB spread (%/mo)')\n"
            "ax.set_title(f'January small-cap pop? Busted (Welch p={wp:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'January {jm:+.2f}%/mo vs other months {rm:+.2f}%/mo -- the wrong way round.')"
        ),

        # ---- verdict ----
        md(
            "## The honest verdict\n\n"
            f"- **Signal -- NONE.** SMB = {R['dn_mean']:+.2f}%/yr, HAC *t* = {R['dn_t']:+.2f}, "
            f"placebo *p* = {R['placebo_p']:.2f}. Wrong sign, statistically zero.\n"
            f"- **Tradability -- MIRAGE.** Net of a {R['net_mean']:+.2f}%/yr book after short "
            "borrow -- nothing to trade.\n"
            "- **January concentration? -- BUSTED.** The signature January pop is absent and "
            "reversed.\n"
            "- **Survivorship -- Named.** Failed small-caps are gone, biasing the premium up; "
            "it is *still* negative.\n\n"
            "Banz (1981) is a genuine historical finding on a broad 1936-1975 universe. On a "
            "small survivor basket over the mega-cap-led 2001-2026 era, replicated honestly "
            "with real costs, the size premium simply isn't there -- the expected, on-brand "
            "outcome for a pointed academic factor."
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
            "# Size-Effect -- the quant teardown\n"
            "### Cap-ranking, dollar- vs beta-neutral spreads, placebo, decay\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![January concentration%3F: Busted](https://img.shields.io/badge/January--concentration%3F-Busted-8b949e?style=flat-square)\n\n"
            "Protocol: each month-end rank a 40-name survivor basket by market cap "
            "(`cap_t = cap_now * price_t / price_now`, shares-constant), long the small half / "
            "short the large half, **one execution lag** (signal month *m* -> return month "
            "*m+1*). Dollar-neutral and beta-neutral books, costs + short borrow, a "
            "label-shuffle placebo, and an early-vs-late decay slice.\n\n"
            "> Companion plain-language notebook: "
            "**[01_for_the_curious.ipynb](01_for_the_curious.ipynb)**."
        ),
        code(BOOT),

        md(
            "## The spreads -- gross, beta-neutral, and net of costs\n\n"
            "Hedging out market beta (ratio ~0.97 -- small and large were near-equal beta on "
            "this basket) does not rescue the book; costs only deepen the hole."
        ),
        code(
            "if HAVE_REAL:\n"
            "    net = st.apply_costs(book['smb_dn'], book['turnover'], 10.0, 100.0)\n"
            "    rows = [('Dollar-neutral, gross', st.summary(book['smb_dn'])),\n"
            "            ('Beta-neutral, gross', st.summary(book['smb_bn'])),\n"
            "            ('Dollar-neutral, net', st.summary(net))]\n"
            "    for name, s in rows:\n"
            "        print(f'{name:28s} mean {s[\"mean\"]*100:+6.2f}%/yr  '\n"
            "              f'Sharpe {s[\"sharpe\"]:+.3f}  HAC t {s[\"tstat\"]:+.3f}')\n"
            "    print(f'Avg beta-hedge ratio: {book[\"beta_ratio\"].mean():.3f}  | '\n"
            "          f'avg turnover: {book[\"turnover\"].mean()*100:.1f}%/mo')\n"
            "else:\n"
            "    print(f'Dollar-neutral gross  mean {R[\"dn_mean\"]:+.2f}%/yr  HAC t {R[\"dn_t\"]:+.3f}')\n"
            "    print(f'Beta-neutral   gross  mean {R[\"bn_mean\"]:+.2f}%/yr  HAC t {R[\"bn_t\"]:+.3f}')\n"
            "    print(f'Dollar-neutral net    mean {R[\"net_mean\"]:+.2f}%/yr  HAC t {R[\"net_t\"]:+.3f}')"
        ),

        md(
            "## Equity curve & drawdown -- the small-minus-large book\n\n"
            "A self-financing SMB book on this basket is a slow grind *down*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + book['smb_dn']).cumprod()\n"
            "    dd = eq / eq.cummax() - 1\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios': [2, 1]})\n"
            "    a1.plot(eq.index, eq.values, c=RED, lw=1.4)\n"
            "    a1.axhline(1.0, c='k', lw=0.8, ls=':'); a1.set_ylabel('growth of $1')\n"
            "    a1.set_title('Dollar-neutral SMB (long small, short large)')\n"
            "    a2.fill_between(dd.index, dd.values*100, 0, color=RED, alpha=0.4)\n"
            "    a2.set_ylabel('drawdown (%)'); a2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Terminal multiple: {eq.iloc[-1]:.2f}x | max DD {dd.min()*100:.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: dollar-neutral {R[\"dn_mean\"]:+.2f}%/yr, max DD {R[\"dn_dd\"]:.1f}%')"
        ),

        md(
            "## The label-shuffle placebo\n\n"
            "Re-assign the small/large labels at random each month (preserving the 20/20 split) "
            "and recompute the HAC *t*, 300 times. If the size *label* carries information, the "
            "real *t* should sit in the tail of this null. It does not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t_real, p = st.placebo_pvalue(prices, spy, caps, n_perm=200, min_stocks=10)\n"
            "    print(f'Real DN HAC t = {t_real:+.3f}')\n"
            "    print(f'Placebo p (|t_null| >= |t_real|) = {p:.3f}')\n"
            "else:\n"
            "    print(f'Frozen: real t = {R[\"placebo_t\"]:+.3f}, placebo p = {R[\"placebo_p\"]:.3f}')\n"
            "print('\\nThe size label is indistinguishable from a random coin-flip label.')"
        ),

        md(
            "## Decay within sample -- early vs late\n\n"
            "Data begins 2001, so we split the sample at its midpoint. The small tilt that was "
            "mildly positive early flips negative in the mega-cap-led second half -- the "
            "well-documented post-publication / post-1980 size decay (Schwert 2003)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dec = st.decade_split(book, cut='2013-07-01')\n"
            "    print(f\"Early 2001-2013: {dec['pre_mean_ann']*100:+.2f}%/yr  \"\n"
            "          f\"HAC t {dec['pre_t']:+.3f}  (n={dec['pre_n']})\")\n"
            "    print(f\"Late  2013-2026: {dec['post_mean_ann']*100:+.2f}%/yr  \"\n"
            "          f\"HAC t {dec['post_t']:+.3f}  (n={dec['post_n']})\")\n"
            "else:\n"
            "    print(f\"Early 2001-2013: {R['early_mean']:+.2f}%/yr  HAC t {R['early_t']:+.3f}\")\n"
            "    print(f\"Late  2013-2026: {R['late_mean']:+.2f}%/yr  HAC t {R['late_t']:+.3f}\")"
        ),

        md(
            "## Survivorship & limitations\n\n"
            "- **Survivorship bias (named).** The basket is names still trading in 2026. Failed "
            "small-caps -- the natural *long* leg -- are absent, biasing the size premium "
            "**upward**. It is *still* negative here, so the true premium is no better.\n"
            "- **Cap reconstruction.** `cap_t = cap_now * price_t / price_now` holds shares "
            "outstanding constant (ignores buybacks/issuance). Adequate for *ranking* a stable "
            "basket into small/large halves; it would matter more for a value-weighted book.\n"
            "- **Basket size.** 40 names is a thin cross-section vs the thousands Banz used; "
            "the median split gives only 20-a-side. This is a *replication on retail-pullable "
            "data*, not the CRSP universe.\n"
            "- **Quality control.** Asness et al. (2018) show the size premium re-emerges only "
            "after controlling for quality (junk small-caps drag it). We do **not** control for "
            "quality -- a documented reason the raw effect looks dead.\n\n"
            "*Think size survives once you add the missing small-caps and a quality screen? "
            "Fork this, expand to the Russell 2000 with delisted names, sort within quality "
            "buckets, and show *t* > 2 net of costs. That is the bar.*"
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
