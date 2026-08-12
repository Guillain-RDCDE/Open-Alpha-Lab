"""Generate the two narrative notebooks for Study 895 (Defensive Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic in well under two minutes. Real-tape cells read
the study's cached yfinance tape under ../_cache/ when present and otherwise quote the frozen
headline numbers in ``R`` (a mirror of docs/results.md). The heavy pieces (the 2,000-draw
bootstrap) are re-run LIGHT in-notebook with the canonical numbers quoted from ``R``; the
synthetic control runs anywhere with no network.
"""

from __future__ import annotations

import os

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (fingerprint 7d4dcacd42bd,
# 158-month MTUM-inception window, excess-of-cash, as-of 2026-06-30).
R = dict(
    asof="2026-06-30", fingerprint="7d4dcacd42bd", n=158,
    win="2013-05 -> 2026-06",
    # per-strategy: cagr%, vol%, exSharpe, maxdd%, wealth
    strat=dict(
        MTUM=dict(cagr=16.80, vol=16.3, sh=0.941, dd=-30.2, w=7.72),
        USMV=dict(cagr=10.40, vol=11.5, sh=0.775, dd=-19.1, w=3.68),
        QUAL=dict(cagr=13.75, vol=14.7, sh=0.841, dd=-27.8, w=5.28),
        SPY=dict(cagr=14.38, vol=14.4, sh=0.895, dd=-23.9, w=5.86),
        BLEND=dict(cagr=13.70, vol=13.1, sh=0.924, dd=-22.2, w=5.42),
        BLEND_NET=dict(cagr=13.69, vol=13.1, sh=0.924, dd=-22.2, w=5.42),
        VOLW=dict(cagr=12.98, vol=12.9, sh=0.875, dd=-21.9, w=4.41),
    ),
    # headline race vs MTUM
    race=dict(
        fifty=dict(n=158, sh_bl=0.924, sh_mt=0.941, adv=-0.016, diff=-26.4,
                   t_nw=-1.88, t_1s=-2.21, lo=-0.139, hi=0.164, p=0.573, net_adv=-0.017),
        volw=dict(n=146, sh_bl=0.875, sh_mt=0.919, adv=-0.044, diff=-32.4,
                  t_nw=-1.73, t_1s=-2.10, lo=-0.175, hi=0.188, p=0.524, net_adv=-0.045),
    ),
    # crash-window drawdowns: MTUM, 50/50, USMV
    crash=dict(
        full=(-30.2, -22.2, -19.1),
        covid=(-17.9, -18.5, -19.1),
        bear22=(-21.2, -17.7, -14.1),
        q4_18=(-15.4, -11.6, -7.6),
    ),
    # era cut (50/50 - MTUM)
    era=dict(early=dict(win="2013-05..2019-11", n=79, adv=0.082, diff=-11.6, t=-1.06),
             late=dict(win="2019-12..2026-06", n=79, adv=-0.102, diff=-41.2, t=-1.62)),
    # calendar years: MTUM, USMV, 50/50, SPY
    cal={
        2013: (17.1, 7.5, 12.2, 17.5), 2014: (14.6, 16.3, 15.5, 13.5),
        2015: (8.9, 5.4, 7.2, 1.2), 2016: (5.0, 10.6, 7.8, 12.0),
        2017: (37.5, 18.9, 27.9, 21.7), 2018: (-1.7, 1.3, -0.1, -4.6),
        2019: (27.3, 27.7, 27.5, 31.2), 2020: (29.9, 5.6, 17.3, 18.3),
        2021: (13.4, 20.8, 17.3, 28.7), 2022: (-18.3, -9.4, -13.8, -18.2),
        2023: (9.1, 10.3, 9.8, 26.2), 2024: (32.9, 15.7, 24.2, 24.9),
        2025: (22.1, 7.6, 15.0, 17.7), 2026: (37.2, 3.2, 19.6, 10.1),
    },
    turnover=dict(fifty=1.03, volw=3.20),
    syn=dict(null=dict(adv=0.000, dd_bl=-27.7, dd_mt=-27.7),
             planted=dict(adv=0.242, t=2.56, diff=36.8, dd_bl=-48.0, dd_mt=-67.6)),
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
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
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#3b6fb0"

from def_momentum import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    MRET = data.monthly_total_returns(PRICES)
    CASH = MRET[data.CASH]
    SLEEVES = MRET[data.SLEEVES].dropna()
    COMMON = SLEEVES.index
    FIFTY = st.fixed_blend(SLEEVES, 0.5)
    VOLW = st.vol_weighted_blend(SLEEVES, lookback=12, lag=1)
    def ex(col):
        return st.excess(MRET, col).reindex(COMMON).dropna()
else:
    PRICES = MRET = CASH = SLEEVES = COMMON = FIFTY = VOLW = None
print("real cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    S = R["strat"]
    rc = R["race"]["fifty"]
    cr = R["crash"]
    cells = [
        md(
            "# Momentum without the crashes? 🛟\n"
            "### Blending the momentum ETF with the min-vol ETF — a smoother ride, or just less of everything?\n\n"
            + BADGES +
            "**Momentum** is the market's best-known winning streak: buy what's been going up. It "
            "works for years — and then, in a violent reversal, it *crashes* (1932, 2009). The "
            "famous fix: bolt on a **min-volatility** sleeve — the calmest stocks in the market — "
            "to soften the falls. The pitch writes itself: *momentum's returns, without momentum's "
            "crashes.*\n\n"
            "We test the shipped version of that idea: a **50/50 blend of MTUM (momentum) and USMV "
            "(min-vol)**, rebalanced monthly, against momentum alone, min-vol alone, quality and "
            "the S&P 500 — everything measured **above cash**, net of fees, since MTUM's 2013 "
            "launch.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the bootstrap CI and the "
            "costed series? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Dedup note.** The desk already graded the momentum-*crash* mechanism "
            "([508-momentum-crashes](../../508-momentum-crashes/)), the low-vol anomaly "
            "([330](../../330-low-volatility-anomaly/)) and the *single* factor wrappers "
            "([601](../../601-factor-etf-live-test/)). This study asks the **blend** question those "
            "leave open. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the blend give a **higher Sharpe** (better return-per-unit-risk) than momentum? | "
            f"**No.** The blend's excess-of-cash Sharpe is **{S['BLEND']['sh']:.3f}** vs momentum's "
            f"**{S['MTUM']['sh']:.3f}** — a dead heat (advantage **{rc['adv']:+.3f}**, and the "
            "error bars straddle zero). |\n"
            "| Did it give **smaller drawdowns**? | **On average, yes** — worst fall "
            f"**{S['BLEND']['dd']:.1f}%** vs momentum's **{S['MTUM']['dd']:.1f}%**. But it "
            "**failed in the 2020 crash** (see below). |\n"
            "| So what did you actually get? | A **calmer ride at a matching cut in return** — CAGR "
            f"**{S['BLEND']['cagr']:.1f}%** vs momentum's **{S['MTUM']['cagr']:.1f}%**. Less crash, "
            "less return, **same Sharpe**. |\n"
            "| Was it a *free* lunch (diversification magic)? | **No.** The blend sits right on the "
            "straight line between its two ingredients — you dialled risk down, nothing more. |"
        ),

        md(
            "## The picture that tells the whole story\n\n"
            "Plot each strategy as a dot: risk (volatility) across, reward (return above cash) up. "
            "A *free lunch* would put the blend **above** the line joining its two sleeves. Watch "
            "where it actually lands:"
        ),
        code(
            "fig, ax = plt.subplots()\n"
            "pts = {'MTUM': (R['strat']['MTUM']['vol'], R['strat']['MTUM']['cagr'], GREEN),\n"
            "       'USMV': (R['strat']['USMV']['vol'], R['strat']['USMV']['cagr'], BLUE),\n"
            "       '50/50 blend': (R['strat']['BLEND']['vol'], R['strat']['BLEND']['cagr'], AMBER),\n"
            "       'SPY': (R['strat']['SPY']['vol'], R['strat']['SPY']['cagr'], GREY)}\n"
            "# the straight line between the two sleeves\n"
            "mx, my = pts['MTUM'][0], pts['MTUM'][1]\n"
            "ux, uy = pts['USMV'][0], pts['USMV'][1]\n"
            "ax.plot([ux, mx], [uy, my], '--', color='k', lw=1, alpha=.5, label='line between the two sleeves')\n"
            "for name,(x,y,c) in pts.items():\n"
            "    ax.scatter([x],[y], s=160, color=c, zorder=3, edgecolor='white')\n"
            "    ax.annotate(name, (x,y), textcoords='offset points', xytext=(8,6), fontsize=10)\n"
            "ax.set_xlabel('risk — annualised volatility (%)'); ax.set_ylabel('reward — CAGR (%)')\n"
            "ax.set_title('the blend lands ON the line between its sleeves — no free lunch')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "print(f\"blend Sharpe {R['strat']['BLEND']['sh']:.3f}  sits between USMV \"\n"
            "      f\"{R['strat']['USMV']['sh']:.3f} and MTUM {R['strat']['MTUM']['sh']:.3f}\")\n"
        ),
        md(
            "The amber blend dot sits **right on the dashed line** between momentum and min-vol. "
            "That is the whole result in one image: mixing the two just **slides you down the "
            "line** toward less risk and less return. There is no bulge above the line — no "
            "diversification bonus lifting your risk-adjusted return. You picked a point on a "
            "risk dial; you did not find an edge."
        ),

        md(
            "## \"Without the crashes\" — sometimes true, and it failed when it mattered most\n\n"
            "Here is the worst drop each strategy took in three sell-offs on the tape. Green shading "
            "means the blend helped; red means it didn't:"
        ),
        code(
            "labels = ['full window', '2020 COVID', '2022 bear', '2018 Q4']\n"
            "keys = ['full', 'covid', 'bear22', 'q4_18']\n"
            "mt = [R['crash'][k][0] for k in keys]\n"
            "bl = [R['crash'][k][1] for k in keys]\n"
            "uv = [R['crash'][k][2] for k in keys]\n"
            "x = np.arange(len(labels)); w = 0.27\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(x-w, mt, w, color=GREEN, label='MTUM (momentum)')\n"
            "ax.bar(x,   bl, w, color=AMBER, label='50/50 blend')\n"
            "ax.bar(x+w, uv, w, color=BLUE, label='USMV (min-vol)')\n"
            "ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('worst drawdown in the window (%)')\n"
            "ax.set_title('the blend cushions 2022 and 2018 — but NOT the fast 2020 crash')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "print(f\"2020 COVID: MTUM {R['crash']['covid'][0]:.1f}%  blend {R['crash']['covid'][1]:.1f}%  \"\n"
            "      f\"USMV {R['crash']['covid'][2]:.1f}%  <- min-vol was the WORST of the three\")\n"
        ),
        md(
            "Look at **2020 COVID** — the fastest crash on the whole tape, exactly the kind of "
            f"violent reversal the overlay is supposed to defend against. Min-vol fell "
            f"**{cr['covid'][2]:.1f}%**, *more* than momentum's **{cr['covid'][0]:.1f}%**, and "
            f"dragged the blend to **{cr['covid'][1]:.1f}%** — slightly **worse** than just holding "
            "momentum. The 'seat belt' unbuckled in the crash it was sold for. It *did* help in the "
            "slower 2022 grind and the 2018 wobble — so the protection is real but **unreliable**, "
            "not a rule you can lean on.\n\n"
            "> 🧭 **And the big one is missing.** The canonical momentum crash was **2008-09** — but "
            "the MTUM ETF didn't exist until 2013, so it's **entirely outside** what we can test. "
            "The thesis's headline exhibit never makes it onto the tape."
        ),

        md(
            "## Every year, the blend is just the middle child\n\n"
            "Calendar-year returns. By construction the blend is the average of its two sleeves — it "
            "never wins outright, never loses outright, and never surprises:"
        ),
        code(
            "yrs = sorted(R['cal'])\n"
            "mt = [R['cal'][y][0] for y in yrs]; uv = [R['cal'][y][1] for y in yrs]\n"
            "bl = [R['cal'][y][2] for y in yrs]\n"
            "x = np.arange(len(yrs)); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(11,4.6))\n"
            "ax.bar(x-w/2, mt, w, color=GREEN, alpha=.85, label='MTUM')\n"
            "ax.bar(x+w/2, uv, w, color=BLUE, alpha=.85, label='USMV')\n"
            "ax.plot(x, bl, 'o-', color=AMBER, lw=1.8, label='50/50 blend')\n"
            "ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([str(y) for y in yrs], rotation=45)\n"
            "ax.set_ylabel('total return (%)'); ax.set_title('the blend (amber) is always the middle of its two sleeves')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
        ),

        md(
            "## The verdict, in plain words\n\n"
            "- **The calmer ride is real.** Lower volatility "
            f"({S['BLEND']['vol']:.1f}% vs {S['MTUM']['vol']:.1f}%), a shallower worst-case fall on "
            f"average ({S['BLEND']['dd']:.1f}% vs {S['MTUM']['dd']:.1f}%). If your goal is simply to "
            "shake less, the blend delivers.\n"
            "- **The *edge* is not.** You paid for that calm with a matching cut in return "
            f"({S['BLEND']['cagr']:.1f}% vs {S['MTUM']['cagr']:.1f}% CAGR). Risk-adjusted, it's a "
            f"**wash** — Sharpe advantage {rc['adv']:+.3f}, error bars across zero.\n"
            "- **The crash hedge is unreliable.** Min-vol was *useless* in the 2020 crash, and the "
            "one crash the thesis is built on (2009) is off the tape entirely.\n"
            "- **It's a risk dial, not a discovery.** Anyone wanting less momentum risk can simply "
            "hold less momentum and more cash — the min-vol sleeve adds nothing you couldn't get "
            "for free.\n\n"
            "**Signal: Mixed (real risk cut, no Sharpe edge, unreliable hedge). Tradability: Mirage "
            "(a beta dial dressed as an edge).**\n\n"
            "> 📓 Full statistics, the bootstrap CI, the era cut and the honest fine print: "
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
    rc = R["race"]
    cells = [
        md(
            "# Defensive Momentum — the quant teardown 🛟\n\n"
            + BADGES +
            "**Claim under test:** a 50/50 (or inverse-vol) blend of MTUM (momentum) and USMV "
            "(min-vol) delivers a **higher excess-of-cash Sharpe** and **shallower momentum-crash "
            "drawdowns** than MTUM alone, net of costs.\n\n"
            "**Dedup guard:** [508-momentum-crashes](../../508-momentum-crashes/) grades the crash "
            "*mechanism*; [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) the low-vol "
            "anomaly; [601-factor-etf-live-test](../../601-factor-etf-live-test/) the *single* live "
            "wrappers; [237-residual-momentum](../../237-residual-momentum/) a *different* crash fix. "
            "This study grades the **blend**.\n\n"
            "**Method skeleton.** Monthly total returns (yfinance auto-adjusted), sliced to the last "
            f"complete month ({R['asof']}); excess = minus the BIL cash leg. Blend window = MTUM's "
            f"2013-05 inception → 2026-06 ({R['n']} months). Two blends: fixed **50/50** monthly-"
            "rebalanced, and **inverse-trailing-12m-vol** with weights formed at *t−1* (one lag, no "
            "look-ahead). The excess-vs-excess **Sharpe advantage** vs MTUM, the **Newey-West** *t* "
            "on the monthly return difference, a moving-block **bootstrap CI** on the advantage; "
            "max drawdown + named crash windows; an **era cut**; a **costed** net series (one-way "
            "cost × realized turnover; long-only, no borrow).\n\n"
            f"Frozen headline run: [`docs/results.md`](../docs/results.md), as-of {R['asof']}, "
            f"fingerprint `{R['fingerprint']}`."
        ),
        code(BOOT_CELL),

        md("## 0 · Data stamp (cache-first, deterministic)"),
        code(
            "if HAVE_REAL:\n"
            "    try:\n"
            "        from quantlab import repro\n"
            "        panel = MRET[data.TICKERS].loc[COMMON]\n"
            "        print('fingerprint:', repro.fingerprint(panel), '(frozen:', R['fingerprint'] + ')')\n"
            "    except Exception as e:\n"
            "        print('quantlab.repro unavailable:', e)\n"
            "    print('blend window:', COMMON.min().date(), '->', COMMON.max().date(), f'({len(COMMON)} months)')\n"
            "    for tk in data.TICKERS:\n"
            "        s = MRET[tk].dropna()\n"
            "        print(f'  {tk}: {s.index.min().date()} -> {s.index.max().date()}  ({len(s)} months)')\n"
            "else:\n"
            "    print('cache missing — the frozen numbers in R carry the notebook')\n"
        ),
        md(
            "> 💡 **In plain words:** the fingerprint is a hash of the exact monthly panel behind "
            "the published verdict — a matching 12 characters means you hold byte-for-byte the same "
            "data. Note the blend is **MTUM-inception-limited**: USMV runs to 176 months but the "
            "blend can only start where *both* sleeves exist (2013-05)."
        ),

        md(
            "## 1 · The strategy table (excess-of-cash, same 158-month window)\n\n"
            "CAGR, annualised vol, excess Sharpe and max drawdown for each sleeve, the blends, QUAL "
            "and SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    def row(name, r):\n"
            "        a = st.ann_stats(r.reindex(COMMON).dropna(), CASH)\n"
            "        return [name, a['cagr']*100, a['vol']*100, a['sharpe'], a['maxdd']*100, a['wealth']]\n"
            "    rows = [row('MTUM', MRET['MTUM']), row('USMV', MRET['USMV']), row('QUAL', MRET['QUAL']),\n"
            "            row('SPY', MRET['SPY']), row('50/50 blend', FIFTY['gross']),\n"
            "            row('50/50 net', st.apply_costs(FIFTY, 3.0)),\n"
            "            row('inv-vol blend', VOLW['gross']), row('inv-vol net', st.apply_costs(VOLW, 3.0))]\n"
            "    print(pd.DataFrame(rows, columns=['strategy','CAGR%','vol%','exSharpe','maxDD%','$1'])\n"
            "          .round(3).to_string(index=False))\n"
            "else:\n"
            "    for k,v in R['strat'].items():\n"
            "        print(k, v)\n"
        ),
        md(
            "The blend's excess Sharpe (**0.924**) sits *between* USMV (0.775) and MTUM (0.941) — a "
            "weighted average, not a lift above either. Vol drops to 13.1% (from 16.3%); CAGR drops "
            "to 13.7% (from 16.8%). Risk and return fall together, in lockstep."
        ),

        md(
            "## 2 · The headline race — excess-vs-excess Sharpe advantage vs MTUM\n\n"
            "Both legs excess-of-cash. The Sharpe **advantage** answers 'is the blend a better "
            "risk-adjusted deal?'; the **Newey-West *t*** on the mean monthly return difference "
            "answers 'is that gap distinguishable from zero?'. A moving-block bootstrap (block 6) "
            "puts a CI on the advantage — light draws in-notebook; the canonical 2,000-draw numbers "
            "are quoted from `R`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mt_ex = ex('MTUM')\n"
            "    for label, blend in [('50/50', FIFTY), ('inv-vol', VOLW)]:\n"
            "        bl_ex = (blend['gross'] - CASH).reindex(COMMON).dropna()\n"
            "        idx = bl_ex.index.intersection(mt_ex.index)\n"
            "        race = st.sharpe_advantage(bl_ex.loc[idx], mt_ex.loc[idx])\n"
            "        boot = st.bootstrap_sharpe_adv(bl_ex.loc[idx], mt_ex.loc[idx], n_draws=400, seed=895)\n"
            "        print(f\"{label}: exSharpe {race['sharpe_a']:.3f} vs MTUM {race['sharpe_b']:.3f}  \"\n"
            "              f\"adv {race['sharpe_adv']:+.3f}  diff {race['diff_bps']:+.1f} bps/mo  \"\n"
            "              f\"NW t {race['t_nw']:+.2f}  boot(400) CI [{boot['lo']:+.3f},{boot['hi']:+.3f}]\")\n"
            "    print('\\n(canonical 2,000-draw CIs are in R / docs/results.md)')\n"
            "else:\n"
            "    print('cache missing')\n"
            "for label, k in [('50/50', 'fifty'), ('inv-vol', 'volw')]:\n"
            "    d = R['race'][k]\n"
            "    print(f\"[R] {label}: adv {d['adv']:+.3f}  diff {d['diff']:+.1f} bps  NW t {d['t_nw']:+.2f}  \"\n"
            "          f\"1s t {d['t_1s']:+.2f}  boot CI [{d['lo']:+.3f},{d['hi']:+.3f}]  P(adv>0) {d['p']:.3f}\")\n"
        ),
        code(
            "d = R['race']['fifty']\n"
            "fig, ax = plt.subplots(figsize=(8.5,3.2))\n"
            "# the bootstrap CI on the Sharpe advantage, against the zero line\n"
            "ax.axvspan(d['lo'], d['hi'], color=AMBER, alpha=.25, label='95% bootstrap CI')\n"
            "ax.scatter([d['adv']], [0], s=140, color=AMBER, zorder=3, label=f\"observed adv {d['adv']:+.3f}\")\n"
            "ax.axvline(0, color=RED, lw=1.5, ls='--', label='zero (no advantage)')\n"
            "ax.set_yticks([]); ax.set_xlabel('50/50 blend excess-Sharpe advantage over MTUM')\n"
            "ax.set_title('the Sharpe advantage is a wash — its CI straddles zero')\n"
            "ax.legend(fontsize=9, loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print(f\"P(advantage > 0) under the bootstrap = {d['p']:.3f}  -> a coin flip\")\n"
            "print(f\"but the RETURN give-up is robust: {d['diff']:+.1f} bps/mo, one-sample t {d['t_1s']:+.2f}\")\n"
        ),
        md(
            "The claimed *higher* Sharpe is not there: the 50/50 advantage is **−0.016**, its "
            "bootstrap CI is **[−0.14, +0.16]**, and P(adv>0) ≈ **0.57** — a coin flip. What *is* "
            "robust is the return the blend **surrenders**: −26 bps/mo (one-sample *t* −2.21), the "
            "mechanical cost of holding less momentum. Costs don't enter into it — at 3 bps/side and "
            "~1%/mo turnover the **net** advantage (−0.017) equals the gross."
        ),

        md(
            "## 3 · Crash geometry — the drawdown grid\n\n"
            "Worst peak-to-trough inside each named window. The full-window reduction is real; the "
            "per-crash story is not uniform."
        ),
        code(
            "if HAVE_REAL:\n"
            "    wins = [('full', COMMON.min().strftime('%Y-%m-%d'), COMMON.max().strftime('%Y-%m-%d')),\n"
            "            ('2020 COVID', '2020-01-01', '2020-06-30'),\n"
            "            ('2022 bear', '2022-01-01', '2022-12-31'),\n"
            "            ('2018 Q4', '2018-09-01', '2018-12-31')]\n"
            "    for lab, s, e in wins:\n"
            "        m = st.window_drawdown(MRET['MTUM'].reindex(COMMON), s, e)\n"
            "        b = st.window_drawdown(FIFTY['gross'].reindex(COMMON), s, e)\n"
            "        u = st.window_drawdown(MRET['USMV'].reindex(COMMON), s, e)\n"
            "        flag = 'blend helped' if b > m else 'blend did NOT help'\n"
            "        print(f\"{lab:11s}: MTUM {m*100:6.1f}%  50/50 {b*100:6.1f}%  USMV {u*100:6.1f}%   -> {flag}\")\n"
            "else:\n"
            "    for k in ['full','covid','bear22','q4_18']:\n"
            "        print(k, R['crash'][k])\n"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots()\n"
            "    for name, c in [('MTUM', GREEN), ('50/50 blend', AMBER), ('USMV', BLUE)]:\n"
            "        r = (FIFTY['gross'] if name=='50/50 blend' else MRET[name.split()[0]]).reindex(COMMON).dropna()\n"
            "        ax.plot(st.drawdown_curve(r)*100, color=c, lw=1.5, label=name)\n"
            "    ax.set_ylabel('drawdown (%)'); ax.set_title('underwater curves — the blend is calmer on average, not in every crash')\n"
            "    ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing — see docs/results.md drawdown table')\n"
        ),
        md(
            "The blend's underwater curve is shallower **on average**, but in **2020** it dips "
            "*below* MTUM's — min-vol drew down −19.1%, the worst of the three, in the fastest crash "
            "on the tape. A defensive overlay that abandons you in the sharpest sell-off is not a "
            "dependable crash hedge."
        ),

        md(
            "## 4 · Era cut — is the (absent) edge at least stable? No.\n\n"
            "Split the 158 months in half; recompute the Sharpe advantage in each."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mt_ex = ex('MTUM')\n"
            "    bl_ex = (FIFTY['gross'] - CASH).reindex(COMMON).dropna()\n"
            "    idx = bl_ex.index.intersection(mt_ex.index)\n"
            "    eras = st.era_split(bl_ex.loc[idx], mt_ex.loc[idx])\n"
            "    for nm in ('early','late'):\n"
            "        e = eras[nm]\n"
            "        print(f\"{nm:5s} {e['start']}..{e['end']} ({e['n']}m): Sharpe adv {e['sharpe_adv']:+.3f}  \"\n"
            "              f\"diff {e['diff_bps']:+.1f} bps  NW t {e['t_nw']:+.2f}\")\n"
            "else:\n"
            "    for nm in ('early','late'):\n"
            "        print(nm, R['era'][nm])\n"
        ),
        md(
            "The tiny advantage **flips sign**: mildly positive when momentum was choppy (2013-19, "
            "**+0.082**), clearly negative in the momentum-led 2020s (**−0.102**). Not robust in "
            "sign, let alone magnitude — the opposite of what a real edge does."
        ),

        md(
            "## 5 · Synthetic control — the machinery is faithful\n\n"
            "A deterministic joint (momentum, min-vol, bench, cash) monthly world with a tunable "
            "planted crash/diversification `edge` (`data.synthetic_sleeves`). On the null the two "
            "sleeves are the *same series*, so the blend is exactly a sleeve — the estimator must "
            "return zero. On a planted edge it must recover a positive Sharpe advantage and a "
            "shallower blend drawdown. Runs anywhere, no network."
        ),
        code(
            "for label, edge in [('null (edge=0)', 0.0), ('planted (edge=1)', 1.0)]:\n"
            "    w = data.synthetic_sleeves(edge=edge, seed=895, n_months=160)\n"
            "    d = st.synthetic_detect(w, 0.5)\n"
            "    t = 'undefined (diff exactly 0)' if not np.isfinite(d['t_nw']) else f\"{d['t_nw']:+.2f}\"\n"
            "    print(f\"{label}: Sharpe adv {d['sharpe_adv']:+.3f}  NW t {t}  \"\n"
            "          f\"blend maxDD {d['blend_maxdd']*100:.1f}% vs MTUM {d['mtum_maxdd']*100:.1f}%\")\n"
            "print('\\n(machinery proof only — never cited in support of a stamp)')\n"
        ),
        md(
            "Null: advantage **+0.000**, drawdowns identical — **no false positive**. Planted: "
            "advantage **+0.242** (NW *t* +2.56) with the blend drawdown a full 20 points shallower "
            "than momentum's. The estimator finds a real defensive-momentum edge **when one exists** "
            "— which is exactly why its verdict on the real tape (a wash) is credible."
        ),

        md(
            "## Verdict\n\n"
            "- **Signal — MIXED.** The vol / full-window-drawdown reduction is real and mechanical "
            "(13.1% vs 16.3% vol; −22.2% vs −30.2% maxDD). But the headline **higher Sharpe is "
            "refuted**: 50/50 advantage −0.016, bootstrap CI [−0.14, +0.16] straddling zero, sign "
            "flipping across eras, with a *significant* return give-up (−26 bps/mo, *t* −2.21). The "
            "blend rides the straight line between its sleeves — no diversification convexity — and "
            "the crash protection is inconsistent (min-vol failed in 2020). The 2009 momentum crash "
            "is out of sample.\n"
            "- **Tradability — MIRAGE.** Nothing to bank. Costs are trivial (net advantage −0.017 = "
            "gross), so it's a **no-edge** story, not a cost story. The blend's whole 'benefit' is a "
            "**beta / volatility dial**: less momentum → less crash *and* less return → same Sharpe. "
            "The free lunch the pitch promises dissolves into replicable risk reduction.\n\n"
            "> Frozen numbers: [`docs/results.md`](../docs/results.md) (fingerprint "
            "`" + R['fingerprint'] + "`). *Research & education, not investment advice.*"
        ),
    ]
    return new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})


def main():
    import nbformat as nbf
    for name, nb in [("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())]:
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
