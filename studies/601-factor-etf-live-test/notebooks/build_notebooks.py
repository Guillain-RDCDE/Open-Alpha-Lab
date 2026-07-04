"""Generate the two narrative notebooks for Study 601 (Factor ETFs — Live Test).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the study's cached
yfinance tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). Heavy pieces (the 10,000-draw permutation placebos, the
2,000-draw vol bootstraps) are re-run LIGHT in-notebook with the canonical numbers quoted
from ``R``. The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance tape, monthly
# stats sliced to 2026-06-30, fingerprint fd36c691c5cb).
R = dict(
    asof="2026-07-03", last_month="2026-06-30", fingerprint="fd36c691c5cb",
    funds=dict(
        USMV=dict(start="2011-11", n=176, beta=0.689, se=0.041, t_b1=-7.52, r2=74.0,
                  alpha=0.69, t_a=0.47, vr=0.799, vr_lo=0.741, vr_hi=0.876,
                  vol=11.2, spy_vol=14.0, up=71.9, down=65.9,
                  cagr=11.44, spy_cagr=14.94, sh=0.887, spy_sh=0.959,
                  dd=-19.1, w=4.90, spy_w=7.71, act=-28.9, t_act=-1.94),
        MTUM=dict(start="2013-05", n=158, beta=0.979, se=0.051, t_b1=-0.40, r2=75.4,
                  alpha=2.65, t_a=1.10, vr=1.129, vr_lo=1.002, vr_hi=1.223,
                  vol=16.3, spy_vol=14.4, up=99.9, down=81.8,
                  cagr=16.80, spy_cagr=14.38, sh=0.934, spy_sh=0.887,
                  dd=-30.2, w=7.72, spy_w=5.86, act=19.9, t_act=0.93),
        VLUE=dict(start="2013-05", n=158, beta=1.077, se=0.058, t_b1=1.34, r2=76.6,
                  alpha=-0.96, t_a=-0.32, vr=1.231, vr_lo=1.073, vr_hi=1.317,
                  vol=17.7, spy_vol=14.4, up=103.3, down=106.9,
                  cagr=13.81, spy_cagr=14.38, sh=0.722, spy_sh=0.887,
                  dd=-29.0, w=5.49, spy_w=5.86, act=0.2, t_act=0.01),
        QUAL=dict(start="2013-08", n=155, beta=0.993, se=0.021, t_b1=-0.36, r2=96.1,
                  alpha=-0.23, t_a=-0.28, vr=1.013, vr_lo=0.985, vr_hi=1.055,
                  vol=14.7, spy_vol=14.5, up=98.2, down=98.7,
                  cagr=13.75, spy_cagr=14.14, sh=0.834, spy_sh=0.867,
                  dd=-27.8, w=5.28, spy_w=5.52, act=-2.7, t_act=-0.41),
    ),
    spy_dd=-23.9,
    # exposure delivery: loading, NW t, 2-factor R2, split means (bps), diff, Welch t, placebo p
    style=dict(
        MTUM=dict(name="sector 12-1 WML", load=0.326, t=7.45, r2=81.9,
                  pos=96.2, neg=-64.5, diff=160.7, welch=4.59, p=0.0000),
        VLUE=dict(name="IWD-IWF value spread", load=0.517, t=9.99, r2=85.6,
                  pos=131.2, neg=-84.5, diff=215.7, welch=6.06, p=0.0000),
        QUAL=dict(name="SPHQ-SPY quality spread", load=0.384, t=8.76, r2=97.4,
                  pos=20.7, neg=-26.4, diff=47.1, welch=3.63, p=0.0003),
    ),
    # alpha NW t at lags 3/6/12
    alpha_lags=dict(USMV=(0.45, 0.47, 0.47), MTUM=(1.11, 1.10, 1.24),
                    VLUE=(-0.36, -0.32, -0.31), QUAL=(-0.29, -0.28, -0.27)),
    # style-proxy sanity: mean bps/mo, NW t
    proxies=dict(WML=(5.5, 0.28), VMG=(-4.4, -0.22), QMB=(-4.8, -0.42)),
    # synthetic control
    syn=dict(null=dict(beta=1.019, t_b1=1.82, load=0.010, t_l=0.56, a=-0.56, t_a=-1.02),
             planted=dict(beta=0.717, t_b1=-10.54, load=0.510, t_l=27.37, a=2.44, t_a=4.43)),
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Beat SPY outright?: Busted](https://img.shields.io/badge/Beat_SPY_outright%3F-Busted-8b949e?style=flat-square)\n\n"
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

from factor_etf_live_test import data, strategy as st

FUNDS = data.FUNDS
HAVE_REAL = data.have_real()
if HAVE_REAL:
    TAPE = data.load_tape()
    MRET = data.monthly_total_returns(TAPE[[c for c in TAPE.columns if c != "^IRX"]])
    RF = data.monthly_rf(TAPE["^IRX"])
    SPY = MRET["SPY"].dropna()
    WML = data.sector_momentum_spread(MRET[data.SECTORS].dropna())
    VMG = data.value_spread(MRET)
    QMB = data.quality_spread(MRET)
    SPREADS = {"MTUM": WML, "VLUE": VMG, "QUAL": QMB}

    def fund_frame(tk):
        r = MRET[tk].dropna()
        idx = r.index.intersection(SPY.index).intersection(RF.dropna().index)
        return r.loc[idx], SPY.loc[idx], RF.loc[idx]
else:
    TAPE = MRET = RF = SPY = WML = VMG = QMB = None
    SPREADS = {}
print("real cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    F = R["funds"]
    cells = [
        md(
            "# The factor zoo went retail: did the ETFs deliver? 🏷️\n"
            "### USMV, MTUM, VLUE, QUAL vs the S&P 500 — thirteen live years, in plain English\n\n"
            + BADGES +
            "Between 2011 and 2013, BlackRock did something academics had promised for decades: it "
            "put the famous **factor premia** — low-volatility, momentum, value, quality — into "
            "ordinary ETFs anyone can buy for 0.15%/yr. The pitch was simple: *these patterns beat "
            "the market in a hundred years of data; now you can own them with one ticket.*\n\n"
            "That was a decade and a half ago. These funds now hold tens of billions of dollars. "
            "So — did each one actually **deliver**? We audit the four flagship funds against SPY, "
            "since each fund's first day, on total returns, net of their own fees.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West regressions, bootstraps and placebos? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Dedup note.** The desk already graded the *academic* versions of these claims "
            "([330-low-volatility-anomaly](../../330-low-volatility-anomaly/), "
            "[242-quality-minus-junk](../../242-quality-minus-junk/)). This study asks a different "
            "question: did the **shipped products** do what the label says? Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did each fund deliver its promised *exposure*? | **Yes — decisively.** USMV really is "
            f"~20% less volatile than SPY (beta {F['USMV']['beta']:.2f}); MTUM really moves with "
            "momentum, VLUE with value, QUAL with quality — all at *t*-statistics of 7 to 10. |\n"
            "| Did any deliver *alpha* (market-beating risk-adjusted return)? | **No.** Alphas sit "
            f"between {F['VLUE']['alpha']:+.1f}% and {F['MTUM']['alpha']:+.1f}%/yr — all statistically "
            "zero. |\n"
            "| Did any simply beat SPY? | **No — busted.** Three of four *lagged* SPY; MTUM finished "
            f"ahead (+{F['MTUM']['cagr']-F['MTUM']['spy_cagr']:.1f} pp/yr) but well within noise. |\n"
            "| So what did buyers actually get? | A **risk profile**, faithfully delivered — not an "
            "edge. USMV holders got smaller crashes (down-capture 66%) and 3.5 pp/yr less return. |"
        ),

        md(
            "## Four races against the index they were sold to beat\n\n"
            "Growth of $1, each fund vs SPY over the *same* window (every fund's own life). "
            "Total returns, net of fees:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)\n"
            "    for ax, tk in zip(axes.ravel(), FUNDS):\n"
            "        r, sp, f = fund_frame(tk)\n"
            "        ax.plot((1+sp).cumprod(), color=GREY, lw=1.6, label='SPY')\n"
            "        ax.plot((1+r).cumprod(), color=GREEN if tk=='MTUM' else AMBER, lw=1.6, label=tk)\n"
            "        ax.set_yscale('log'); ax.legend(fontsize=9)\n"
            "        ax.set_title(f\"{tk}: ${R['funds'][tk]['w']:.2f} vs SPY ${R['funds'][tk]['spy_w']:.2f}\", fontsize=11)\n"
            "    fig.suptitle('growth of $1 since each inception (log scale, total return)', y=1.02)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    for tk in FUNDS:\n"
            "        d = R['funds'][tk]\n"
            "        print(f\"{tk}: CAGR {d['cagr']:.2f}% vs SPY {d['spy_cagr']:.2f}%  (gap {d['cagr']-d['spy_cagr']:+.2f} pp/yr)\")\n"
            "else:\n"
            "    print('cache missing — see docs/results.md for the frozen numbers')\n"
        ),
        md(
            "Only **MTUM** ends ahead of SPY — and the quants will show even that gap is "
            "statistical noise. **USMV**, the largest factor ETF in the world, trails by "
            "**3.5 percentage points a year**. Is that a scandal? No — and that's the whole "
            "point of this study. Keep reading."
        ),

        md(
            "## Half one of the promise: the exposure — delivered, beautifully\n\n"
            "Each fund's label makes a checkable claim. USMV's is the easiest: *you will be less "
            "volatile than the market.* Watch its rolling volatility against SPY's:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r, sp, f = fund_frame('USMV')\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(sp.rolling(12).std()*np.sqrt(12)*100, color=GREY, lw=1.6, label='SPY')\n"
            "    ax.plot(r.rolling(12).std()*np.sqrt(12)*100, color=GREEN, lw=1.6, label='USMV')\n"
            "    ax.set_ylabel('rolling 12-month volatility (%/yr)')\n"
            "    ax.set_title('USMV really is the calmer fund — in every regime')\n"
            "    ax.legend(); plt.show()\n"
            "    d = R['funds']['USMV']\n"
            "    print(f\"full-window vol: USMV {d['vol']:.1f}% vs SPY {d['spy_vol']:.1f}%  \"\n"
            "          f\"(ratio {d['vr']:.2f}, CI [{d['vr_lo']:.2f}, {d['vr_hi']:.2f}])\")\n"
            "    print(f\"beta {d['beta']:.3f}; in SPY's down months USMV only falls {d['down']:.0f}% as far\")\n"
        ),
        md(
            "USMV runs at **~80% of SPY's volatility** (a 20% reduction — the low end of the "
            "promised 20–30%), catches only **66%** of the market's down months, and its worst "
            f"peak-to-trough fall was **{F['USMV']['dd']:.0f}%** vs SPY's {R['spy_dd']:.0f}%. "
            "The label is accurate.\n\n"
            "The other three labels are checkable too: in months when the *momentum* style won "
            "(winners kept winning), MTUM should beat SPY; when *value* won, VLUE should; when "
            "*quality* won, QUAL should. Do they?"
        ),
        code(
            "sty = R['style']\n"
            "fig, ax = plt.subplots()\n"
            "x = np.arange(3); w = 0.36\n"
            "pos = [sty[t]['pos'] for t in ('MTUM','VLUE','QUAL')]\n"
            "neg = [sty[t]['neg'] for t in ('MTUM','VLUE','QUAL')]\n"
            "ax.bar(x-w/2, pos, w, color=GREEN, label='months the factor WON')\n"
            "ax.bar(x+w/2, neg, w, color=RED, label='months the factor LOST')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['MTUM\\n(momentum)', 'VLUE\\n(value)', 'QUAL\\n(quality)'])\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_ylabel('fund return minus SPY (bps/month)')\n"
            "ax.set_title('each fund beats SPY exactly when its factor wins — the exposure is real')\n"
            "ax.legend(); plt.show()\n"
            "for t in ('MTUM','VLUE','QUAL'):\n"
            "    s = sty[t]\n"
            "    print(f\"{t}: {s['pos']:+.0f} bps/mo when its factor won, {s['neg']:+.0f} when it lost \"\n"
            "          f\"(gap {s['diff']:+.0f} bps, t = {s['welch']:.1f})\")\n"
        ),
        md(
            "Textbook behaviour. Each fund out-runs SPY precisely in its factor's good months and "
            "lags in the bad ones, at *t*-statistics between 3.6 and 6.1 (regression loadings: 7 to "
            "10). **The wrappers work.** Whatever you think of factor investing, the engineering "
            "delivered exactly what it printed on the tin.\n\n"
            "> 🔬 **For the quants:** the loadings come from a two-factor Newey-West regression of "
            "fund excess return on [market, realized style spread]; the splits carry 10,000-draw "
            "permutation placebos (*p* ≤ 0.0003). Notebook 02 has the full grid."
        ),

        md(
            "## Half two of the promise: the alpha — missing\n\n"
            "The academic papers didn't just say these factors *exist* — they said they **pay**: "
            "higher risk-adjusted returns than the market. That's the CAPM alpha. Here it is for "
            "the four live funds, with the statistical bar drawn where the desk always draws it:"
        ),
        code(
            "F = R['funds']\n"
            "fig, ax = plt.subplots()\n"
            "names = list(F)\n"
            "alphas = [F[t]['alpha'] for t in names]\n"
            "tstats = [F[t]['t_a'] for t in names]\n"
            "cols = [AMBER if a > 0 else RED for a in alphas]\n"
            "bars = ax.bar(names, alphas, color=cols)\n"
            "for b, a, t in zip(bars, alphas, tstats):\n"
            "    ax.text(b.get_x()+b.get_width()/2, a + (0.12 if a>=0 else -0.3),\n"
            "            f'{a:+.2f}%/yr\\n(t = {t:+.2f})', ha='center', fontsize=10)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_ylabel('CAPM alpha (%/yr, Newey-West)')\n"
            "ax.set_ylim(-2.2, 4.2)\n"
            "ax.set_title('the promised alpha: every t-statistic far below 2')\n"
            "plt.show()\n"
            "print('the desk bar: REAL needs a robust t >= 2 on the real tape — none comes close')\n"
        ),
        md(
            "Nothing clears the bar, or comes near it. MTUM's +2.65%/yr *looks* like something "
            "until you see *t* = 1.10 — thirteen years of monthly data cannot tell it from luck. "
            "And here's the deeper reason: the **factors themselves didn't pay** over this window. "
            "Our momentum spread earned ~5 bps/month, the value spread ~−4, the quality spread ~−5 "
            "— all statistically zero. The funds faithfully tracked styles that went nowhere.\n\n"
            "That's the honest reading of the great smart-beta experiment: the products worked; "
            "the *premia* — measured after the papers were published, after the launch wave, after "
            "everyone piled in — did not show up."
        ),

        md(
            "## The verdict, in plain words\n\n"
            "- **Exposure: delivered.** Every label is accurate, at overwhelming statistical "
            "strength. If you want 20% less volatility (USMV) or a genuine momentum tilt (MTUM), "
            "the wrapper gives you exactly that for 0.15%/yr.\n"
            "- **Alpha: not delivered.** All four CAPM alphas are statistically zero; the style "
            "spreads themselves paid nothing over the live window.\n"
            "- **Beating SPY: busted.** None did at significance — three of four lagged outright. "
            "USMV's −3.5 pp/yr is not a failure of the fund; it's what beta 0.69 costs in a "
            "fifteen-year bull market. Buyers got the smoother ride they paid for — many just "
            "thought they were buying outperformance too.\n"
            "- **Caveat:** these are the four flagship *survivors* of the launch wave — the "
            "average factor-ETF experience is worse.\n"
            "**Signal: Real (exposure). Tradability: Fragile. Beat SPY outright: Busted.**\n\n"
            "> 📓 Full statistics, robustness grids and the honest fine print: "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** · frozen numbers: "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R['fingerprint'] + "`).\n\n"
            "*Research & education, not investment advice.*"
        ),
    ]
    return new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    F = R["funds"]
    cells = [
        md(
            "# Factor ETFs — Live Test: the quant teardown 🏷️\n\n"
            + BADGES +
            "**Claim under test:** USMV/MTUM/VLUE/QUAL promised academic factor exposure in a "
            "0.15%/yr ETF wrapper — a decade-plus later, did each deliver (a) the exposure, "
            "(b) the alpha?\n\n"
            "**Dedup guard:** [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) and "
            "[242-quality-minus-junk](../../242-quality-minus-junk/) grade the *academic "
            "cross-sections*; the unit under test here is the **live product**, net of its own "
            "fee, since its own inception.\n\n"
            "**Method skeleton.** Monthly total returns (yfinance auto-adjusted), sliced to the "
            f"last complete month ({R['last_month']}); excess = minus prior-month-end ^IRX/12. "
            "Per fund: CAPM with Newey-West (lags 6; 3/12 in robustness) — alpha *t* and "
            "beta-vs-one *t*; paired moving-block bootstrap CI on the realized vol ratio; "
            "up/down capture; a **two-factor NW regression** on [market excess, realized style "
            "spread] for exposure delivery; a spread-sign month split (Welch *t* + 10,000-draw "
            "permutation placebo). The sector 12-1 WML proxy is formed on months *t−12…t−2* — "
            "known at the end of *t−1*, applied to month *t*: exactly ONE month of lag. No other "
            "trading rule exists in this study (the funds are buy-and-hold products).\n\n"
            f"Frozen headline run: [`docs/results.md`](../docs/results.md), as-of {R['asof']}, "
            f"fingerprint `{R['fingerprint']}`."
        ),
        code(BOOT_CELL),

        md("## 0 · Data stamp (cache-first, deterministic)"),
        code(
            "if HAVE_REAL:\n"
            "    try:\n"
            "        from quantlab import repro\n"
            "        panel = pd.concat([MRET[['SPY'] + FUNDS], RF.rename('rf')], axis=1)\n"
            "        print('fingerprint:', repro.fingerprint(panel), '(frozen:', R['fingerprint'] + ')')\n"
            "    except Exception as e:\n"
            "        print('quantlab.repro unavailable:', e)\n"
            "    for tk in FUNDS:\n"
            "        s = MRET[tk].dropna()\n"
            "        print(f'{tk}: {s.index.min().date()} -> {s.index.max().date()}  ({len(s)} complete months)')\n"
            "else:\n"
            "    print('cache missing — the frozen numbers in R carry the notebook')\n"
        ),
        md(
            "> 💡 **In plain words:** the fingerprint is a hash of the exact monthly panel behind "
            "the published verdict — if your rerun prints the same 12 characters, you are holding "
            "byte-for-byte the same data."
        ),

        md(
            "## 1 · Per-fund CAPM (excess-vs-excess, Newey-West)\n\n"
            "The two headline statistics per fund: **alpha** (the premium the papers promised) and "
            "**beta vs 1** (the risk profile the label promised)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for tk in FUNDS:\n"
            "        r, sp, f = fund_frame(tk)\n"
            "        cp = st.capm(r - f, sp - f, lags=6)\n"
            "        rows.append([tk, len(r), cp['beta'], cp['se_beta'], cp['t_beta_vs1'],\n"
            "                     cp['alpha_ann']*100, cp['t_alpha'], cp['r2']*100])\n"
            "    print(pd.DataFrame(rows, columns=['fund','n','beta','NW se','t(b vs 1)',\n"
            "          'alpha %/yr','NW t(a)','R2 %']).round(3).to_string(index=False))\n"
            "else:\n"
            "    for tk, d in R['funds'].items():\n"
            "        print(tk, 'beta', d['beta'], 't(b vs 1)', d['t_b1'], 'alpha', d['alpha'], 't', d['t_a'])\n"
        ),
        md(
            f"USMV: beta **{F['USMV']['beta']:.3f}**, *t*(β<1) = **{F['USMV']['t_b1']:+.2f}** — the "
            "low-vol profile is delivered at overwhelming significance. MTUM/QUAL are ~market-beta "
            "(0.98/0.99); VLUE runs hot (1.08). Alphas: all |*t*| ≤ 1.24.\n\n"
            "> 💡 **In plain words:** beta is 'how much of the market's movement the fund carries'. "
            "USMV carries 69% of it — exactly the min-vol pitch. Alpha is 'return you can't explain "
            "by carrying the market' — and nobody has any."
        ),

        md(
            "## 2 · USMV's mechanical promise — vol ratio with a paired block bootstrap\n\n"
            "The vol reduction must show on the tape *with a CI* (sampling noise exists even for "
            "mechanical claims). Paired moving-block bootstrap (block 6), light draws in-notebook; "
            "canonical numbers from the 2,000-draw run in `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for tk in FUNDS:\n"
            "        r, sp, f = fund_frame(tk)\n"
            "        vr = st.vol_ratio_ci(r, sp, n_draws=500, seed=601)\n"
            "        cap = st.up_down_capture(r, sp)\n"
            "        d = R['funds'][tk]\n"
            "        print(f\"{tk}: vol ratio {vr['obs']:.3f} (light CI [{vr['lo']:.3f}, {vr['hi']:.3f}]; \"\n"
            "              f\"canonical [{d['vr_lo']:.3f}, {d['vr_hi']:.3f}])  \"\n"
            "              f\"capture up {cap['up']*100:.1f}% / down {cap['down']*100:.1f}%\")\n"
        ),
        md(
            f"USMV's realized vol ratio is **{F['USMV']['vr']:.3f}** (CI "
            f"[{F['USMV']['vr_lo']:.3f}, {F['USMV']['vr_hi']:.3f}]) — a **20% vol reduction** "
            "(CI 12–26%), the *low edge* of the promised 20–30%. Down-capture 65.9%. Note VLUE is "
            "**more** volatile than SPY (ratio 1.23) — 'value' in live form was a higher-beta, "
            "rougher ride."
        ),

        md(
            "## 3 · Exposure delivery — two-factor loadings + spread-sign splits\n\n"
            "Fund excess on [SPY excess, realized style spread], NW lags 6. The spread loading is "
            "the exposure-delivery statistic. Splits: active return (fund − SPY) in factor-won vs "
            "factor-lost months, Welch *t*, permutation placebo (light 1,000 draws in-notebook; "
            "canonical 10,000-draw *p* from `results.md`)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for tk in ('MTUM', 'VLUE', 'QUAL'):\n"
            "        r, sp, f = fund_frame(tk)\n"
            "        sl = st.style_loading(r - f, sp - f, SPREADS[tk], lags=6)\n"
            "        spl = st.spread_sign_split(r - sp, SPREADS[tk], n_draws=1000, seed=601)\n"
            "        d = R['style'][tk]\n"
            "        print(f\"{tk}: loading {sl['loading']:+.3f}  NW t {sl['t_loading']:+.2f}  \"\n"
            "              f\"R2 {sl['r2']*100:.1f}%   split diff {spl['diff_bps']:+.1f} bps \"\n"
            "              f\"Welch t {spl['welch_t']:+.2f}  placebo p {spl['p_placebo']:.4f} \"\n"
            "              f\"(canonical {d['p']:.4f})\")\n"
        ),
        md(
            "All three loadings clear the bar by a mile: MTUM **+0.326 (t 7.45)**, VLUE "
            "**+0.517 (t 9.99)**, QUAL **+0.384 (t 8.76)**; splits at Welch *t* 3.6–6.1, placebos "
            "≤ 0.0003. Combined with USMV's *t*(β<1) = −7.52, **exposure delivery is REAL** on the "
            "real tape.\n\n"
            "> 💡 **In plain words:** we checked whether each fund actually moves with the style on "
            "its label, using a factor we can compute ourselves with no look-ahead. All four do, "
            "unambiguously.\n\n"
            "*Proxy caveats:* the quality spread uses SPHQ (independent provider), which switched "
            "to the S&P 500 Quality index in 2016; the WML proxy is sector-level (coarser than "
            "MSCI's stock-level momentum), which if anything *understates* MTUM's true loading."
        ),

        md("## 4 · Alpha delivery — NW-lag robustness (the axis that fails)"),
        code(
            "if HAVE_REAL:\n"
            "    for tk in FUNDS:\n"
            "        r, sp, f = fund_frame(tk)\n"
            "        ts = [st.capm(r - f, sp - f, lags=lg) for lg in (3, 6, 12)]\n"
            "        print(f\"{tk}: alpha {ts[1]['alpha_ann']*100:+.2f}%/yr   NW t = \"\n"
            "              + '  '.join(f\"{t['t_alpha']:+.2f} (lags {lg})\" for t, lg in zip(ts, (3,6,12))))\n"
            "else:\n"
            "    for tk, tt in R['alpha_lags'].items():\n"
            "        print(tk, 'NW t at lags 3/6/12:', tt)\n"
        ),
        md(
            "No lag choice rescues anyone. MTUM peaks at *t* = 1.24 (lags 12). The literature bar "
            "— *REAL needs a robust t ≥ 2 on the real tape* — is not approached.\n\n"
            "Why? The realized factors themselves paid nothing over the live window: sector WML "
            f"**{R['proxies']['WML'][0]:+.1f}** bps/mo (NW *t* {R['proxies']['WML'][1]:+.2f}), "
            f"IWD−IWF **{R['proxies']['VMG'][0]:+.1f}** (*t* {R['proxies']['VMG'][1]:+.2f}), "
            f"SPHQ−SPY **{R['proxies']['QMB'][0]:+.1f}** (*t* {R['proxies']['QMB'][1]:+.2f}). "
            "Faithful trackers of flat factors produce zero alpha — McLean-Pontiff decay, live."
        ),

        md("## 5 · Third axis — did ANY beat SPY outright?"),
        code(
            "if HAVE_REAL:\n"
            "    for tk in FUNDS:\n"
            "        r, sp, f = fund_frame(tk)\n"
            "        sf, ss = st.ann_stats(r, f), st.ann_stats(sp, f)\n"
            "        act = r - sp\n"
            "        print(f\"{tk}: CAGR {sf['cagr']*100:.2f}% vs SPY {ss['cagr']*100:.2f}%  \"\n"
            "              f\"Sharpe {sf['sharpe']:.3f} vs {ss['sharpe']:.3f}  \"\n"
            "              f\"active {act.mean()*1e4:+.1f} bps/mo  NW t {st.nw_tstat(act, lags=6):+.2f}\")\n"
        ),
        md(
            "**Busted.** None beats SPY at significance; three of four lagged on CAGR. The "
            "sharpest number on the board is actually USMV's **−1.94** — the closest thing to a "
            "*significant* result on this axis is a factor fund significantly **losing** the "
            "outright race (while, to be fair, winning on its own risk-profile terms: Sharpe 0.887 "
            "vs 0.959 is a photo finish at 80% of the vol).\n\n"
            "> 💡 **In plain words:** if what you wanted was 'more money than the S&P', none of "
            "these funds gave it to you, and the biggest one gave you visibly less. What USMV gave "
            "you instead is nearly the same reward-per-risk with smaller crashes — which is what "
            "its label, read carefully, actually promised."
        ),

        md(
            "## 6 · Synthetic control — planted-parameter recovery (machinery proof only)\n\n"
            "Deterministic joint (market, spread, fund) world; the estimators must stay quiet on "
            "the null (β=1, loading=0, α=0) and recover planted parameters. Never cited in support "
            "of a stamp."
        ),
        code(
            "for label, kw in [('null    (b=1.0, s=0.0, a=+0%)', dict(beta=1.0, loading=0.0, alpha_ann=0.0)),\n"
            "                  ('planted (b=0.7, s=0.5, a=+3%)', dict(beta=0.7, loading=0.5, alpha_ann=0.03))]:\n"
            "    w = data.synthetic_world(n_months=168, seed=601, **kw)\n"
            "    sl = st.style_loading(w['fund'], w['mkt'], w['spread'], lags=6)\n"
            "    cp = st.capm(w['fund'], w['mkt'], lags=6)\n"
            "    print(f\"{label}: beta {cp['beta']:.3f} (t vs 1 {cp['t_beta_vs1']:+.2f})  \"\n"
            "          f\"loading {sl['loading']:+.3f} (t {sl['t_loading']:+.2f})  \"\n"
            "          f\"alpha {sl['alpha_ann']*100:+.2f}%/yr (t {sl['t_alpha']:+.2f})\")\n"
        ),
        md(
            "Null stays below the bar on all three estimators; the planted world is recovered "
            "(β 0.717, loading +0.510, α +2.44%/yr at *t* 4.43). The pipeline can detect exactly "
            "the effects it failed to find on the real tape — the nulls are informative."
        ),

        md(
            "## Verdict\n\n"
            "- **Signal — REAL (exposure delivery).** USMV *t*(β<1) = −7.52 with a 20% vol "
            "reduction (CI 12–26%); style loadings *t* = +7.45 / +9.99 / +8.76; splits Welch *t* "
            "3.6–6.1, placebo *p* ≤ 0.0003. All on the live tape, net of fees. Caveat: flagship-"
            "survivor selection (these are the four biggest surviving wrappers of the launch "
            "wave).\n"
            "- **Tradability — FRAGILE.** Access is as good as it gets (0.15%/yr, penny spreads, "
            "huge AUM) — but the harvestable *premium* is absent: alphas −1.0% to +2.7%/yr, all "
            "|*t*| ≤ 1.24 at every lag; the style spreads themselves paid ~zero. You can cheaply "
            "buy a risk profile; you cannot buy the promised edge.\n"
            "- **\"Did any beat SPY outright?\" — BUSTED.** None at significance; three of four "
            "lagged. MTUM's +2.42 pp/yr reads *t* = 0.93.\n\n"
            f"Frozen numbers: [`docs/results.md`](../docs/results.md) (as-of {R['asof']}, "
            f"fingerprint `{R['fingerprint']}`) · sources: "
            "[`docs/references.md`](../docs/references.md).\n\n"
            "*Research & education, not investment advice.*"
        ),
    ]
    return new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)
