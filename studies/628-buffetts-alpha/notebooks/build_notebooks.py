"""Generate the two narrative notebooks for Study 628 (Buffett's Alpha).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached monthly frame
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance BRK-A / spliced
# market / ^IRX, 1980-04-30 -> 2026-06-30, 555 months, as-of 2026-06-30).
R = dict(
    start="1980-04-30", end="2026-06-30", n=555, years=46.2,
    splice_pre=154, splice_spy=401,
    wealth_brk=2880, wealth_mkt=220, brk_ann=18.80, mkt_ann=12.37,
    alpha_ann=9.46, t_alpha=3.47, beta=0.70, r2=0.24, ir=0.50, nw_lags=5,
    brk_exc=15.64, mkt_exc=8.88, brk_sharpe=0.72, mkt_sharpe=0.59, brk_vol=21.6,
    fkp=dict(alpha=12.36, t=3.40, beta=0.68, n=381, brk_sharpe=0.72, mkt_sharpe=0.44),
    # decade table: (decade, alpha, t, beta, brk_exc, mkt_exc, months)
    decades=[("1980s", 23.39, 3.55, 0.82, 31.25, 9.58, 117),
             ("1990s", 5.07, 0.76, 0.92, 16.70, 12.68, 120),
             ("2000s", 6.01, 1.28, 0.47, 4.88, -2.41, 120),
             ("2010s", 5.15, 1.28, 0.60, 12.89, 12.90, 120),
             ("2020s*", 2.06, 0.34, 0.71, 11.40, 13.13, 78)],
    roll=dict(n_windows=436, first="1990-03-31", peak=20.34, peak_at="1990-03-31",
              latest=1.83, latest_t=0.46, latest_beta=0.75,
              last_sig="2002-12-31", share_sig=25.0),
    fade=dict(early_alpha=12.97, early_t=3.48, d_alpha=-10.98, d_t=-2.38,
              d_beta=0.08, d_beta_t=0.65, split="2010-12-31"),
    last15=dict(alpha=2.66, t=0.89, beta=0.75, n=180, brk_sharpe=0.74, mkt_sharpe=0.90,
                wealth_brk=6.45, wealth_mkt=7.37, start="2011-07-31"),
    flite=dict(start="2013-08-31", n=155, capm_alpha=1.54, capm_t=0.46, capm_beta=0.75,
               r2_capm=0.41, full_alpha=1.43, full_t=0.44, r2_full=0.51,
               b_mkt=-0.17, t_mkt=-0.37, b_qual=0.30, t_qual=0.68,
               b_usmv=0.89, t_usmv=3.77),
    # synthetic (20 seeds): (planted %/yr, mean recovered, mean t, share |t|>=2 %)
    syn=[(0.0, 0.27, 0.11, 10), (6.0, 6.27, 2.56, 70)],
    fingerprint="96d18b13a5da",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Any_alpha_left%3F: Busted](https://img.shields.io/badge/Any_alpha_left%3F-Busted-8b949e?style=flat-square)\n\n"
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

from buffetts_alpha import data, strategy as st

HAVE_REAL = data.have_real()
DF = data.load_real() if HAVE_REAL else None
print("real tape cached:", HAVE_REAL,
      "| months:", (0 if DF is None else len(DF)),
      "| span:", ("-" if DF is None else f"{DF.index.min().date()} -> {DF.index.max().date()}"))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buffett's alpha — the most famous edge in the world, audited 🎩\n"
            "### Was Warren Buffett's outperformance *real*? And is there any of it left?\n\n"
            + BADGES +
            "Every market legend on this desk so far has been a folk tale, a coin flip, or an "
            "accounting trick. This one is different: **Warren Buffett's Berkshire Hathaway** is "
            "the single most famous investment track record alive, and in 2013 three quants "
            "(Frazzini, Kabiller & Pedersen — 'FKP') published *Buffett's Alpha*, claiming his "
            "40-year edge is **statistically real**, that it comes from buying **safe, "
            "high-quality stocks with cheap insurance-money leverage** — and that it has been "
            "**fading**.\n\n"
            "We audit all three claims on free public data: every month of Berkshire's stock "
            "since 1980 against the US market.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the fade regression and the "
            "factor loadings? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **One honesty note up front.** Berkshire is on this bench *because it won* — "
            "we're grading the most successful survivor of thousands of 1980s conglomerates. "
            "The stats certify *his* track record, not your odds of finding the next one."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Was the alpha real? | **Yes.** $1 in Berkshire in 1980 became **$2,880** vs the "
            "market's **$220** — and the statistics say luck can't explain it (the quants' *t* "
            "is **3.5**, well past the desk bar of 2). |\n"
            "| Where did it live? | Overwhelmingly the **1980s** (+23%/yr of pure skill). Every "
            "decade since is statistically indistinguishable from a passive index fund. |\n"
            "| Is any left? | **No.** Over the last 15 years Berkshire *lagged* the S&P 500 "
            "($6.45 vs $7.37 per dollar) with zero measurable alpha. |\n\n"
            "The chart below is the whole legend in one picture — note the **log scale**: each "
            "gridline is 10× the one below."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w_brk = (1 + DF['brk']).cumprod()\n"
            "    w_mkt = (1 + DF['mkt']).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(w_brk.index, w_brk, color=GREEN, lw=1.8,\n"
            "            label=f\"Berkshire (BRK-A)  ->  ${R['wealth_brk']:,}\")\n"
            "    ax.plot(w_mkt.index, w_mkt, color=GREY, lw=1.8,\n"
            "            label=f\"US market (total return)  ->  ${R['wealth_mkt']:,}\")\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_title('$1 invested in April 1980 — Berkshire vs the market (log scale)')\n"
            "    ax.set_ylabel('value of $1 (log)')\n"
            "    ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing — headline: $1 -> $2,880 (BRK) vs $220 (market)')"
        ),

        md(
            "## What 'alpha' actually means here\n\n"
            "'Beating the market' isn't enough — you can beat it by simply borrowing and taking "
            "more risk. **Alpha** is the outperformance *left over* after accounting for how much "
            "market risk you took (**beta**). Buffett's numbers are strange in the best way:\n\n"
            f"- his beta is only **{R['beta']:.2f}** — Berkshire moves *less* than the market,\n"
            f"- yet he out-earned it by **{R['brk_exc'] - R['mkt_exc']:+.1f} points a year** over "
            f"{R['years']:.0f} years,\n"
            f"- leaving an alpha of **+{R['alpha_ann']:.1f}%/yr** that risk cannot explain.\n\n"
            "That is exactly FKP's signature: **safe stocks, levered with cheap insurance float** "
            "— less risk than the index, more return.\n\n"
            "> 🔬 **For the quants.** Full-sample excess-vs-excess CAPM, Newey-West HAC: alpha "
            f"+{R['alpha_ann']:.2f}%/yr, *t* = {R['t_alpha']:.2f}, beta {R['beta']:.2f}, "
            f"n = {R['n']}. On FKP's own era (→ 2011): alpha +{R['fkp']['alpha']:.2f}%/yr, "
            f"*t* = {R['fkp']['t']:.2f} — their '*t* > 3' replicates."
        ),

        md(
            "## Where the magic lived — decade by decade\n\n"
            "Split the 46 years into decades and the legend gets a timestamp: the skill is "
            "**front-loaded**. The 1980s alone carry a staggering +23%/yr of alpha; from the "
            "1990s on, every bar could be zero (the error bars — the honest 'could be luck' "
            "range — cross the axis)."
        ),
        code(
            "decs = R['decades']\n"
            "labels = [d[0] for d in decs]; alphas = [d[1] for d in decs]; ts = [d[2] for d in decs]\n"
            "colors = [GREEN if t >= 2 else GREY for t in ts]\n"
            "fig, ax = plt.subplots()\n"
            "bars = ax.bar(labels, alphas, color=colors)\n"
            "for b, t in zip(bars, ts):\n"
            "    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f't={t:.1f}',\n"
            "            ha='center', fontsize=9, color='#444')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_title('Berkshire CAPM alpha by decade (%/yr) — green = statistically real (t >= 2)')\n"
            "ax.set_ylabel('alpha, %/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('* 2020s = partial decade (78 months)')"
        ),

        md(
            "## The fade, filmed in slow motion\n\n"
            "A fairer camera than decades: at every month we look back at the **previous 10 "
            "years** and ask 'was there alpha in this window?'. The curve starts near +20%/yr "
            "and dies on the axis. The last 10-year window that *statistically* certified alpha "
            f"ended in **{R['roll']['last_sig'][:4]}** — over twenty years ago."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ra = st.rolling_alpha(DF, window=120)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(ra.index, ra['alpha_ann_pct'], color=GREEN, lw=1.6, label='rolling 10-y alpha (%/yr)')\n"
            "    sig = ra[ra['t_alpha'] >= 2]\n"
            "    ax.scatter(sig.index, sig['alpha_ann_pct'], s=12, color=GREEN, zorder=3,\n"
            "               label='window clears t >= 2')\n"
            "    ax.axhline(0, color='k', lw=.8)\n"
            "    ax.set_title('The decay curve — 10-year rolling alpha, 1990-2026')\n"
            "    ax.set_ylabel('alpha, %/yr'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"peak {ra['alpha_ann_pct'].max():+.1f}%/yr | latest \"\n"
            "          f\"{ra['alpha_ann_pct'].iloc[-1]:+.1f}%/yr | last significant window ends \"\n"
            "          f\"{ra[ra['t_alpha'] >= 2].index.max().date()}\")\n"
            "else:\n"
            "    print('cache missing — peak +20.3%/yr (1990), latest +1.8%/yr, last t>=2 window ends 2002-12')"
        ),

        md(
            "## The last 15 years — buying the museum, not the artist\n\n"
            f"Since mid-2011: alpha **+{R['last15']['alpha']:.1f}%/yr** with *t* = "
            f"{R['last15']['t']:.2f} — statistically **zero**. Berkshire's risk-adjusted score "
            f"(Sharpe **{R['last15']['brk_sharpe']:.2f}**) is now *below* the index's "
            f"(**{R['last15']['mkt_sharpe']:.2f}**), and a dollar in Berkshire grew to "
            f"**${R['last15']['wealth_brk']:.2f}** vs **${R['last15']['wealth_mkt']:.2f}** in the "
            "S&P 500. What does Berkshire trade like today? A **minimum-volatility index fund** — "
            "in the quants' regression it tracks the min-vol ETF (USMV) more decisively than "
            "anything else. Exactly what FKP predicted: the style is copyable, the leverage "
            "advantage shrank with size, and $1T of assets can't dart in and out of bargains.\n\n"
            "> 💡 Buffett himself has said it for years in the shareholder letters: *'size is the "
            "anchor of performance.'* The tape agrees with him."
        ),

        md(
            "## Takeaway\n\n"
            "1. **The legend is real.** Buffett's alpha is the rare one that survives a hostile "
            "audit: +9.5%/yr of genuine, risk-adjusted skill over 46 years (*t* = 3.5).\n"
            "2. **It's history.** The skill lives in the 1980s; no 10-year stretch since 2002 "
            "certifies alpha, and the fade itself is statistically significant.\n"
            "3. **You can't buy it.** BRK-B costs a click — but what the click buys is a giant, "
            "well-run, low-beta conglomerate that has *lagged* the index for 15 years. The alpha "
            "belongs to the man, the era, and the float — not to today's shareholder.\n\n"
            "*Selection note (repeated on purpose): we graded the winner. Thousands of 1980s "
            "conglomerates went to zero un-audited.*\n\n"
            "---\n"
            "*Study 628 of [Open-Alpha-Lab](../../../README.md). Numbers: "
            "[docs/results.md](../docs/results.md) (as-of 2026-06-30, fingerprint "
            "`96d18b13a5da`). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Buffett's Alpha — the quant teardown 🎩\n\n"
            + BADGES +
            "**Claim under test** (Frazzini, Kabiller & Pedersen, *Buffett's Alpha*, FAJ 2018): "
            "Berkshire's four-decade CAPM alpha is real (*t* > 3), is explained by quality + "
            "low-beta exposure financed with cheap insurance-float leverage, and has been "
            "fading.\n\n"
            "**Design.** BRK-A monthly total return (no dividend since 1967 ⇒ adjusted price = "
            "total return) vs a spliced US-market total return (^GSPC + Shiller dividend yield "
            "→ 1993-01, SPY thereafter; labeled per row), both in excess of the previous "
            "month-end ^IRX/12. Newey-West HAC everywhere; sub-period contrasts get a HAC *t* "
            "on the *difference*; the factor attribution uses investable ETF proxies (QUAL, "
            "USMV) and is labeled **factor-lite**. As-of **2026-06-30**, fingerprint "
            "`96d18b13a5da`.\n\n"
            "**Named biases:** selection-on-success (the fund is studied because it won); our "
            "tape misses 1976-80 (biases the full-sample alpha *down* vs FKP); QUAL/USMV exist "
            "only from 2011-13 (the attribution cannot reach the high-alpha era)."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · Data stamp\n\n"
            "> 💡 **In plain words.** One monthly table: Berkshire's return, the market's "
            "return (with dividends), and the T-bill rate you could have locked at the start of "
            "each month. Everything below is regressions on these three columns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    from quantlab import repro\n"
            "    print(repro.data_stamp('ba_monthly', DF, cols=['brk','mkt','rf'], asof=data.AS_OF))\n"
            "    print(DF['mkt_src'].value_counts().to_string())\n"
            "    print(f\"expected fingerprint: {R['fingerprint']}\")\n"
            "else:\n"
            "    print('cache missing — quoting frozen numbers from R throughout')"
        ),

        md(
            "## 2 · Full-sample CAPM — the headline\n\n"
            "Excess-vs-excess OLS with Newey-West (Bartlett) HAC standard errors, rule-of-thumb "
            "lags. The desk bar is **HAC *t* ≥ 2 on the real tape**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.capm(DF)\n"
            "    cf = st.capm(DF, end='2011-12-31')\n"
            "    print(f\"FULL 1980-04 -> 2026-06 (n={c['n']}, NW lags={c['lags']}):\")\n"
            "    print(f\"  alpha {c['alpha_ann_pct']:+.2f}%/yr  HAC t = {c['t_alpha']:+.2f}   \"\n"
            "          f\"beta {c['beta']:.2f} (t={c['t_beta']:.1f})   R2={c['r2']:.2f}   IR={c['ir']:.2f}\")\n"
            "    print(f\"  excess: BRK {c['brk_exc_ann_pct']:+.2f}%/yr vs mkt {c['mkt_exc_ann_pct']:+.2f}%/yr | \"\n"
            "          f\"Sharpe {c['brk_sharpe']:.2f} vs {c['mkt_sharpe']:.2f} (excess-vs-excess)\")\n"
            "    print(f\"FKP-era -> 2011-12 (n={cf['n']}):\")\n"
            "    print(f\"  alpha {cf['alpha_ann_pct']:+.2f}%/yr  HAC t = {cf['t_alpha']:+.2f}   beta {cf['beta']:.2f}  \"\n"
            "          f\"Sharpe {cf['brk_sharpe']:.2f} vs {cf['mkt_sharpe']:.2f}\")\n"
            "else:\n"
            "    print(f\"frozen: alpha +{R['alpha_ann']}%/yr t={R['t_alpha']} beta {R['beta']} | \"\n"
            "          f\"FKP-era +{R['fkp']['alpha']}%/yr t={R['fkp']['t']}\")"
        ),
        md(
            "> 💡 **In plain words.** Over 46 years Buffett beat the T-bill by ~15.6%/yr while "
            "the market managed ~8.9%/yr — and he did it with *less* market risk (beta 0.70). "
            "The +9.5%/yr left over is 3.5 standard errors from zero: not luck. FKP's *t* > 3 "
            "replicates on free data (*t* = 3.40 on their era).\n\n"
            "**Scatter + fit:** the intercept IS the claim."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yb = (DF['brk'] - DF['rf']) * 100; xm = (DF['mkt'] - DF['rf']) * 100\n"
            "    c = st.capm(DF)\n"
            "    fig, ax = plt.subplots(figsize=(7.5, 6))\n"
            "    ax.scatter(xm, yb, s=8, alpha=.4, color=GREY)\n"
            "    xs = np.linspace(xm.min(), xm.max(), 50)\n"
            "    ax.plot(xs, c['alpha_m']*100 + c['beta']*xs, color=GREEN, lw=2,\n"
            "            label=f\"fit: alpha {c['alpha_ann_pct']:+.1f}%/yr, beta {c['beta']:.2f}\")\n"
            "    ax.plot(xs, xs, color=RED, lw=1, ls='--', label='beta = 1, zero alpha')\n"
            "    ax.set_xlabel('market excess return, %/month'); ax.set_ylabel('BRK excess return, %/month')\n"
            "    ax.set_title('555 months of Berkshire vs the market'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()"
        ),

        md("## 3 · The decay — decades, rolling windows, and a t-test on the fade itself"),
        code(
            "if HAVE_REAL:\n"
            "    dt = st.decade_table(DF)\n"
            "    print(dt.round(2).to_string())\n"
            "else:\n"
            "    for d in R['decades']: print(d)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ra = st.rolling_alpha(DF, window=120)\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)\n"
            "    axes[0].plot(ra.index, ra['alpha_ann_pct'], color=GREEN, lw=1.5)\n"
            "    axes[0].axhline(0, color='k', lw=.8)\n"
            "    axes[0].set_ylabel('alpha, %/yr'); axes[0].set_title('Rolling 10-year CAPM alpha (top) and its HAC t (bottom)')\n"
            "    axes[1].plot(ra.index, ra['t_alpha'], color=GREY, lw=1.5)\n"
            "    axes[1].axhline(2, color=RED, lw=1, ls='--', label='t = 2 (the desk bar)')\n"
            "    axes[1].axhline(0, color='k', lw=.8)\n"
            "    axes[1].set_ylabel('HAC t'); axes[1].legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"peak {ra['alpha_ann_pct'].max():+.2f}%/yr at {ra['alpha_ann_pct'].idxmax().date()} | \"\n"
            "          f\"latest {ra['alpha_ann_pct'].iloc[-1]:+.2f}%/yr (t={ra['t_alpha'].iloc[-1]:+.2f}) | \"\n"
            "          f\"last t>=2 window ends {ra[ra['t_alpha'] >= 2].index.max().date()} | \"\n"
            "          f\"share t>=2: {(ra['t_alpha'] >= 2).mean()*100:.1f}%\")"
        ),
        md(
            "> 💡 **In plain words.** No ten-year stretch ending after 2002 can certify alpha on "
            "its own. And the *change* is not noise — putting a post-2010 dummy in one regression "
            "gives the fade its own *t*-stat:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fd = st.subperiod_fade(DF, split='2010-12-31')\n"
            "    print(f\"early alpha (-> 2010-12): {fd['alpha_early_ann_pct']:+.2f}%/yr (t={fd['t_alpha_early']:+.2f})\")\n"
            "    print(f\"alpha change after 2010 : {fd['d_alpha_ann_pct']:+.2f} pp/yr  HAC t = {fd['t_d_alpha']:+.2f}\")\n"
            "    print(f\"beta  change after 2010 : {fd['d_beta']:+.2f} (t={fd['t_d_beta']:+.2f}) — the fade is alpha, not risk\")\n"
            "else:\n"
            "    print(f\"frozen: d_alpha {R['fade']['d_alpha']} pp/yr, t {R['fade']['d_t']}\")"
        ),

        md(
            "## 4 · Third axis — any alpha left in the last 15 years?\n\n"
            "The out-of-sample continuation of FKP (their tape ends 2011)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c15 = st.capm(DF, start='2011-07-01')\n"
            "    d15 = DF[DF.index >= '2011-07-01']\n"
            "    print(f\"2011-07 -> 2026-06 (n={c15['n']}):\")\n"
            "    print(f\"  alpha {c15['alpha_ann_pct']:+.2f}%/yr  HAC t = {c15['t_alpha']:+.2f}   beta {c15['beta']:.2f}\")\n"
            "    print(f\"  Sharpe: BRK {c15['brk_sharpe']:.2f} vs mkt {c15['mkt_sharpe']:.2f} (excess-vs-excess)\")\n"
            "    print(f\"  $1 -> BRK ${float((1+d15['brk']).prod()):.2f} vs market ${float((1+d15['mkt']).prod()):.2f}\")\n"
            "else:\n"
            "    print(f\"frozen: alpha +{R['last15']['alpha']}%/yr t={R['last15']['t']}, \"\n"
            "          f\"$ {R['last15']['wealth_brk']} vs {R['last15']['wealth_mkt']}\")"
        ),
        md(
            "> 💡 **In plain words.** Zero alpha, a worse Sharpe than the index, and a lagging "
            "dollar. **BUSTED** — nothing is left for today's buyer. That is *also* a "
            "confirmation of FKP: they called the fade in 2013."
        ),

        md(
            "## 5 · Factor-lite attribution — the FKP mechanism, ETF-proxy edition\n\n"
            "FKP explain Buffett with BAB (low-beta) + QMJ (quality). Those academic long-short "
            "factors aren't freely available here, so we use the **investable proxies** USMV "
            "(min-vol) and QUAL (quality), excess of T-bills, on their common window (2013-08 →). "
            "Honest limitation: this window post-dates the alpha, so the test is of the *style "
            "signature*, not a re-run of FKP's full attribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fl = st.factor_lite(DF)\n"
            "    print(f\"window {fl['start']} -> {fl['end']} (n={fl['n']})\")\n"
            "    print(f\"CAPM alone   : alpha {fl['capm_alpha_ann_pct']:+.2f}%/yr (t={fl['capm_t_alpha']:+.2f})  \"\n"
            "          f\"beta {fl['capm_beta']:.2f}  R2={fl['r2_capm']:.2f}\")\n"
            "    print(f\"+ QUAL + USMV: alpha {fl['full_alpha_ann_pct']:+.2f}%/yr (t={fl['full_t_alpha']:+.2f})  R2={fl['r2_full']:.2f}\")\n"
            "    print(f\"loadings: mkt {fl['b_mkt']:+.2f} (t={fl['t_mkt']:+.2f}) | QUAL {fl['b_qual']:+.2f} \"\n"
            "          f\"(t={fl['t_qual']:+.2f}) | USMV {fl['b_usmv']:+.2f} (t={fl['t_usmv']:+.2f})\")\n"
            "else:\n"
            "    print(f\"frozen: USMV loading {R['flite']['b_usmv']} (t={R['flite']['t_usmv']})\")"
        ),
        md(
            "> 💡 **In plain words.** Modern Berkshire prices like a **min-vol index fund**: the "
            "USMV loading (+0.89, *t* = 3.77) soaks up the market beta and lifts R² from 0.41 to "
            "0.51. The style FKP identified is now literally purchasable for 15 bps — which is "
            "precisely why the alpha is gone."
        ),

        md(
            "## 6 · Synthetic control — the machinery is faithful\n\n"
            "Seeded (market, manager) worlds with AR(1) idiosyncratic noise and a **planted** "
            "CAPM alpha, averaged over **20 seeds**. The null must stay quiet; the planted alpha "
            "must light up. *(Machinery proof only — never market evidence.)*"
        ),
        code(
            "for a in (0.0, 0.005):\n"
            "    ts, als = [], []\n"
            "    for s in range(628, 648):\n"
            "        c = st.capm(data.synthetic_world(alpha_m=a, seed=s))\n"
            "        ts.append(c['t_alpha']); als.append(c['alpha_ann_pct'])\n"
            "    ts, als = np.array(ts), np.array(als)\n"
            "    print(f\"planted {a*12*100:4.1f}%/yr: mean recovered {als.mean():+.2f}%/yr  \"\n"
            "          f\"mean HAC t {ts.mean():+.2f}  share |t|>=2: {np.mean(np.abs(ts)>=2)*100:.0f}%\")"
        ),

        md(
            "## 7 · Verdict\n\n"
            "| Axis | Stamp | The number that decides |\n|---|---|---|\n"
            "| Signal | **REAL** | full-sample HAC *t* = **3.47** (alpha +9.46%/yr, beta 0.70); "
            "FKP-era *t* = 3.40 — with the **selection-on-success** caveat named |\n"
            "| Tradability | **FRAGILE** | fade = **−10.98 pp/yr** post-2010 (HAC *t* = −2.38); "
            "no 10-y window clears *t* ≥ 2 since 2002; buying BRK today buys beta 0.75, alpha ≈ 0 |\n"
            "| Any alpha left? | **BUSTED** | last 15y: +2.66%/yr, *t* = 0.89; Sharpe 0.74 vs "
            "0.90; $6.45 vs $7.37 per dollar |\n\n"
            "---\n"
            "*Reproduce: `python examples/verify.py`. Sources: [docs/references.md]"
            "(../docs/references.md). As-of 2026-06-30, fingerprint `96d18b13a5da`. Not "
            "investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    return nb


if __name__ == "__main__":
    for fname, builder in [("01_for_the_curious.ipynb", build_curious),
                           ("02_for_the_quants.ipynb", build_quants)]:
        path = os.path.join(HERE, fname)
        nbf.write(builder(), path)
        print("wrote", path)
