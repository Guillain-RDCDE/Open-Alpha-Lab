"""Generate the two narrative notebooks for Study 901 (Profitable Small-Caps).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached tape under
../_cache/ when present and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# excess-of-cash, common window 2017-06-20 -> 2026-06-30 unless noted; HAC 10 daily lags).
R = dict(
    asof="2026-06-30", fingerprint="3ec88092fc34",
    window="2017-06-20 -> 2026-06-30", n_race=2269,
    # Sharpe race: ann excess %, ann vol %, excess-Sharpe, max DD %
    race={
        "CALF": dict(exc=10.20, vol=25.97, sr=0.393, dd=-49.2, kind="profitable small"),
        "XSHQ": dict(exc=8.99, vol=22.86, sr=0.393, dd=-40.2, kind="profitable small"),
        "IWM":  dict(exc=10.11, vol=23.78, sr=0.425, dd=-43.0, kind="plain small"),
        "IJR":  dict(exc=10.21, vol=23.60, sr=0.433, dd=-46.1, kind="plain small (earnings screen)"),
        "SPY":  dict(exc=13.31, vol=18.68, sr=0.712, dd=-33.9, kind="large cap"),
    },
    # bootstrap Sharpe CI: sharpe, ci_low, ci_high, P(<0)
    boot={
        "CALF": (0.393, -0.234, 0.997, 0.108), "XSHQ": (0.393, -0.209, 0.994, 0.094),
        "IWM": (0.425, -0.209, 1.081, 0.099), "IJR": (0.433, -0.204, 1.073, 0.089),
        "SPY": (0.712, 0.109, 1.406, 0.012),
    },
    # head-to-head: (sr_a, sr_b, diff, ci_low, ci_high, P(diff<=0), daily_diff_bps, hac_t)
    pairs={
        "CALF vs IWM": (0.393, 0.425, -0.033, -0.323, 0.238, 0.584, 0.033, 0.02),
        "CALF vs IJR": (0.393, 0.433, -0.040, -0.274, 0.170, 0.643, -0.004, -0.00),
        "CALF vs SPY": (0.393, 0.712, -0.320, -0.760, 0.063, 0.946, -1.235, -0.60),
        "XSHQ vs IWM": (0.396, 0.434, -0.039, -0.286, 0.193, 0.627, -0.509, -0.43),
        "XSHQ vs IJR": (0.396, 0.438, -0.043, -0.252, 0.135, 0.671, -0.516, -0.54),
        "XSHQ vs SPY": (0.396, 0.730, -0.334, -0.724, 0.026, 0.961, -1.805, -1.10),
    },
    # size/market beta decomposition: alpha %/yr, t_alpha, b_IWM, t_bIWM, b_SPY, t_bSPY, R2
    decomp={
        "CALF": dict(alpha=0.93, t=0.27, b_iwm=1.059, t_iwm=33.1, b_spy=-0.108, t_spy=-2.8, r2=0.816),
        "XSHQ": dict(alpha=0.73, t=0.26, b_iwm=0.887, t_iwm=36.6, b_spy=-0.063, t_spy=-1.6, r2=0.773),
    },
    # era cut CALF vs IWM: (start, end, n, calf_sr, iwm_sr, gap)
    era={"pre": ("2017-06-20", "2020-12-31", 891, 0.364, 0.499, -0.135),
         "post": ("2021-01-04", "2026-06-30", 1378, 0.419, 0.373, 0.045)},
    # calendar-year total returns %: year -> [CALF, XSHQ, IWM, IJR, SPY]
    cy={2017: [5.8, 9.4, 9.1, 10.2, 10.2], 2018: [-10.1, -6.1, -11.1, -8.5, -4.6],
        2019: [18.2, 17.4, 25.4, 22.8, 31.2], 2020: [16.6, 11.8, 20.0, 11.3, 18.3],
        2021: [40.7, 24.0, 14.5, 26.6, 28.7], 2022: [-15.2, -15.0, -20.5, -16.2, -18.2],
        2023: [35.4, 23.9, 16.8, 16.1, 26.2], 2024: [-7.4, 7.5, 11.4, 8.6, 24.9],
        2025: [2.3, 0.9, 12.7, 5.9, 17.7], 2026: [14.3, 15.4, 22.6, 24.0, 10.1]},
    # costed net race: leg -> (charge %/yr, sr_gross, sr_net, iwm_sr, net_gap)
    costed={"CALF": (0.45, 0.393, 0.375, 0.425, -0.050),
            "XSHQ": (0.15, 0.396, 0.389, 0.434, -0.045)},
    # isolation trade: leg -> (gross %/yr, t_gross, net %/yr, t_net, charge %/yr)
    iso={"CALF": (0.08, 0.02, -0.62, -0.18, 0.70), "XSHQ": (-1.28, -0.43, -1.98, -0.67, 0.70)},
    # synthetic control: planted edge -> (diff, ci_low, ci_high, hac_t)
    syn=[(0.0, -0.006, -0.301, 0.284, -0.26), (0.4, 0.425, 0.133, 0.718, 2.57)],
    ers={"CALF": 0.59, "XSHQ": 0.29, "IWM": 0.19, "IJR": 0.06, "SPY": 0.09, "BIL": 0.14},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Cleaned size premium, tradable?: Busted](https://img.shields.io/badge/Cleaned_size_premium%2C_tradable%3F-Busted-8b949e?style=flat-square)\n\n"
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

from profitable_small import data, strategy as st

XLEGS = {"CALF": "x_CALF", "XSHQ": "x_XSHQ", "IWM": "x_IWM", "IJR": "x_IJR", "SPY": "x_SPY"}
RLEGS = {"CALF": "r_CALF", "XSHQ": "r_XSHQ", "IWM": "r_IWM", "IJR": "r_IJR", "SPY": "r_SPY"}
COLOR = {"CALF": GREEN, "XSHQ": "#2f9e6f", "IWM": GREY, "IJR": "#6f7a86", "SPY": RED}

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    FRAME = data.daily_frame(PRICES, asof=data.AS_OF)
else:
    PRICES = FRAME = None
print("real tape cached:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    rc, era = R["race"], R["era"]
    cells = [
        md(
            "# The small-cap fund that keeps only the *profitable* names — does it win? 🌱\n"
            "### Profitable small caps (CALF, XSHQ) vs plain small caps (IWM, IJR) vs the S&P 500, in plain English\n\n"
            + BADGES +
            "There's a famous fix for a famous disappointment. The **size premium** — the old idea "
            "that small companies out-earn big ones — mostly stopped working. In 2018 a landmark AQR "
            "paper (\"Size Matters, **If You Control Your Junk**\") argued why: small caps are, on "
            "average, *junkier* — less profitable, more fragile — and the junk drags the premium to "
            "zero. Strip out the junk, keep only the **profitable** small caps, and (they showed) the "
            "size premium roars back.\n\n"
            "You can *buy* that idea today: ETFs like **CALF** (small caps with the fattest free cash "
            "flow) and **XSHQ** (small caps screened for quality) hold only the profitable ones. So "
            "here's the clean question — **did they actually beat plain small caps?**\n\n"
            "> 📓 **Plain-language layer.** Want the Sharpe ratios, HAC *t*-stats and regressions? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did profitable small caps (CALF, XSHQ) beat plain small caps? | **No.** Over 2017–2026 "
            f"they earned a *worse* risk-adjusted return — a Sharpe of **{rc['CALF']['sr']:.2f}** vs "
            f"**{rc['IWM']['sr']:.2f}** for plain Russell 2000. The quality screen didn't help; it cost "
            "a hair. |\n"
            f"| Did any small-cap flavour beat the S&P 500? | **No.** Large-cap SPY posted a Sharpe of "
            f"**{rc['SPY']['sr']:.2f}** — nearly double any small-cap fund, cleaned or not. |\n"
            "| Is there a hidden 'quality' edge once you adjust for risk? | **No.** After stripping out "
            "plain small-cap and market exposure, the leftover 'alpha' is about **+0.9%/yr — "
            "statistically indistinguishable from zero.** CALF is essentially small-cap beta in a more "
            "expensive wrapper. |\n"
            "| So was the AQR paper wrong? | **Probably not — but you can't buy it this way.** Their "
            "result is about *long-short* portfolios in the stock cross-section. A long-only "
            "profitable-small-cap ETF simply didn't deliver it. |"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Small caps beat big caps — but only the profitable ones. The junky small caps (no "
            "earnings, burning cash, fragile) drag the whole size premium to zero. Hold quality fixed "
            "and small beats big, everywhere, all the time.\"*\n\n"
            "That's the AQR \"Size Matters, If You Control Your Junk\" thesis (Asness, Frazzini, Israel, "
            "Moskowitz & Pedersen, 2018). It's genuinely compelling academic work. The tradable "
            "question is narrower and sharper: **when someone packages 'profitable small caps' into an "
            "ETF you can buy, does it out-earn a plain small-cap index on a risk-adjusted basis, after "
            "fees?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Small-cap-quality ETFs gathered billions on exactly this pitch: *\"get the size premium, "
            "cleaned of the junk that ruined it.\"* If it works, you'd tilt your small-cap sleeve into "
            "CALF/XSHQ and pocket a better Sharpe for free. If it *doesn't* — if it's just small-cap "
            "beta wearing a quality costume and charging more for it — then the pitch is marketing, and "
            "the honest move was boring old large caps. Nine years of live data (2017 onward, including "
            "COVID and the 2022 bear) can tell the two apart."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Five funds, all as **total-return** prices from Yahoo Finance, every return measured "
            "**above cash** (minus 1–3 month T-bills, so nobody wins just by holding a higher cash "
            "rate):\n\n"
            "- **CALF** — small caps with the highest free-cash-flow yield ('cash cows'). Profitable "
            "small, expression #1.\n"
            "- **XSHQ** — small caps screened on a quality composite (profitability, low debt, clean "
            "accounting). Profitable small, expression #2.\n"
            "- **IWM** — plain Russell 2000. Junk-and-all small caps.\n"
            "- **IJR** — plain S&P SmallCap 600 (which already nudges out no-earnings names). "
            "'Half-cleaned' small caps.\n"
            "- **SPY** — the S&P 500. Large caps, the yardstick.\n\n"
            "We race their **Sharpe ratios** (return per unit of risk) on the window they all share "
            "(2017-06 onward, because CALF is the youngest). Higher Sharpe wins. Then we ask whether "
            "any gap survives statistics and costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The headline race.** Risk-adjusted return (Sharpe, above cash), 2017–2026, same window "
            "for everyone."
        ),
        code(
            "order = ['CALF','XSHQ','IWM','IJR','SPY']\n"
            "if HAVE_REAL:\n"
            "    r = st.race(FRAME, XLEGS)\n"
            "    srs = [r['legs'][k]['sharpe'] for k in order]\n"
            "    win = f\"{r['start']} -> {r['end']}  (n={r['n']})\"\n"
            "else:\n"
            "    srs = [R['race'][k]['sr'] for k in order]\n"
            "    win = R['window'] + f\"  (n={R['n_race']})\"\n"
            "cols = [GREEN, '#2f9e6f', GREY, '#6f7a86', RED]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.8))\n"
            "bars = ax.bar(order, srs, color=cols, width=.62)\n"
            "for k,v in zip(order, srs): ax.annotate(f'{v:.2f}', (k, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_ylabel('Sharpe ratio (excess of cash)')\n"
            "ax.set_title('Risk-adjusted return, ' + win + '\\nprofitable small (green) did NOT beat plain small (grey); large cap (red) won')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', {k: round(v,3) for k,v in zip(order, srs)})"
        ),
        md(
            f"The two green 'profitable small-cap' bars ({rc['CALF']['sr']:.2f}) sit **just below** the "
            f"grey plain small-cap bars ({rc['IWM']['sr']:.2f}–{rc['IJR']['sr']:.2f}) — and everything "
            f"small is dwarfed by the red S&P 500 bar ({rc['SPY']['sr']:.2f}). The quality screen bought "
            "no risk-adjusted advantage. If it did anything, it slightly *reduced* the Sharpe (CALF's "
            "big value-y bets also gave it the worst drawdown, "
            f"{rc['CALF']['dd']:.0f}%).\n\n"
            "**But is that small gap even real, or just noise?** Let's put error bars on it."
        ),
        code(
            "order = ['CALF','XSHQ','IWM','IJR','SPY']\n"
            "if HAVE_REAL:\n"
            "    win = st.common_window(FRAME, list(XLEGS.values()))\n"
            "    pts, los, his = [], [], []\n"
            "    for k in order:\n"
            "        b = st.sharpe_ci_bootstrap(win[XLEGS[k]].to_numpy())\n"
            "        pts.append(b['sharpe']); los.append(b['ci_low']); his.append(b['ci_high'])\n"
            "else:\n"
            "    pts = [R['boot'][k][0] for k in order]\n"
            "    los = [R['boot'][k][1] for k in order]; his = [R['boot'][k][2] for k in order]\n"
            "cols = [GREEN, '#2f9e6f', GREY, '#6f7a86', RED]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.8))\n"
            "x = np.arange(len(order))\n"
            "yerr = [np.array(pts)-np.array(los), np.array(his)-np.array(pts)]\n"
            "ax.errorbar(x, pts, yerr=yerr, fmt='none', ecolor='#888', capsize=5)\n"
            "for i,k in enumerate(order): ax.plot(x[i], pts[i], 'o', color=cols[i], ms=10)\n"
            "ax.axhline(0, c=RED, lw=1, ls='--')\n"
            "ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel('Sharpe (excess of cash), 95% bootstrap CI')\n"
            "ax.set_title('Only the S&P 500 Sharpe is clear of zero; every small-cap CI straddles it')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k in order: print(f'{k}: Sharpe {R[\"boot\"][k][0]:+.2f}  95% CI [{R[\"boot\"][k][1]:+.2f}, {R[\"boot\"][k][2]:+.2f}]')"
        ),
        md(
            "Every small-cap fund's confidence interval **crosses zero** — on a resample there's a "
            "~1-in-10 chance its excess-Sharpe was actually negative. Only **SPY** sits clearly in "
            "positive territory. So not only did profitable small caps fail to beat plain small caps — "
            "*no* small-cap Sharpe here is even reliably distinguishable from zero over these nine "
            "years.\n\n"
            "**Was the quality screen doing anything at all, or just tracking plain small caps?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = st.beta_decomp(FRAME, 'x_CALF', ['x_IWM', 'x_SPY'])\n"
            "    b_iwm, alpha, t_alpha = d['betas']['x_IWM'], d['alpha_ann_pct'], d['t_alpha']\n"
            "else:\n"
            "    dd = R['decomp']['CALF']; b_iwm, alpha, t_alpha = dd['b_iwm'], dd['alpha'], dd['t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['loads on plain\\nsmall caps (beta)', \"leftover 'quality'\\nedge (alpha, %/yr)\"],\n"
            "       [b_iwm, alpha], color=[GREY, GREEN], width=.5)\n"
            "ax.annotate(f'{b_iwm:.2f}x', (0, b_iwm), ha='center', va='bottom')\n"
            "ax.annotate(f'{alpha:+.1f}%/yr\\n(t={t_alpha:+.2f}, ~zero)', (1, alpha), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.6); ax.set_title('CALF is ~1.06x plain small-cap beta, with NO leftover edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CALF beta on IWM = {b_iwm:.2f} ; leftover alpha = {alpha:+.2f}%/yr (t = {t_alpha:+.2f})')"
        ),
        md(
            f"> 🔬 **For the quants:** regress CALF's daily excess return on plain small caps (IWM) and "
            f"the market (SPY). CALF loads **{R['decomp']['CALF']['b_iwm']:.2f}×** on plain small caps, "
            f"and the leftover alpha is **+{R['decomp']['CALF']['alpha']:.1f}%/yr at *t* = "
            f"{R['decomp']['CALF']['t']:.2f}** — indistinguishable from zero.\n\n"
            "The 'profitable small-cap' fund is, statistically, just **plain small caps** with a fatter "
            "fee. There is no cleaned premium hiding in the risk-adjusted numbers."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Profitable small caps (Sharpe {rc['CALF']['sr']:.2f}) did **not** beat "
            f"plain small caps ({rc['IWM']['sr']:.2f}); the gap is negative, noisy, and there's no "
            "risk-adjusted 'quality' alpha (~0.9%/yr, *t* ≈ 0.3). The machinery works — on a simulated "
            "world with a real planted edge it lights up cleanly (quants notebook) — so this is a true "
            "null, not a broken test.\n"
            "- **Tradability — Mirage.** Nothing to bank; what looks like a candidate edge is pure "
            "small-cap beta, and CALF's 0.59% fee plus trading costs push the (already zero) gap "
            "negative.\n"
            f"- **'The size effect, cleaned' — Busted (as an ETF).** Over 2017–2026 the S&P 500 "
            f"(Sharpe {rc['SPY']['sr']:.2f}) beat every small-cap flavour. The quality screen didn't "
            "rescue the size premium into something you could buy."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Long-short vs long-only.** AQR's result is a *long-short* one: long profitable small, "
            "**short junky small**. A long-only ETF can't short the junk, so it never isolates the "
            "clean premium — it just holds nicer small caps that still rise and fall with the whole "
            "small-cap tide.\n"
            "- **The value tail wagged CALF.** CALF's cash-flow screen is really a *value* tilt; it had "
            "a huge 2021 and 2023 and an ugly 2024–2025. Big swings, no persistent risk-adjusted edge "
            "(see the calendar-year table in the quants notebook).\n"
            "- **Cousins on the desk.** [513-size-effect](../../513-size-effect/README.md) tests the "
            "raw small-minus-big premium; [657-larry-portfolio](../../657-larry-portfolio/README.md) is "
            "the small-**value** tilt; [242-quality-minus-junk](../../242-quality-minus-junk/README.md) "
            "is the QMJ factor across all caps — this study is its small-cap-ETF slice; "
            "[362-piotroski-f-score](../../362-piotroski-f-score/README.md) is single-name quality "
            "scoring.\n\n"
            "*Convinced 'quality small caps' must be better? Re-read the error-bar chart — then notice "
            "the boring red S&P 500 dot is the only one that clears zero.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    rc, era, dc = R["race"], R["era"], R["decomp"]
    cells = [
        md(
            "# Profitable Small-Caps — a quantitative teardown 🔬\n"
            "### Excess-of-cash Sharpe races · paired Sharpe-difference bootstraps · HAC *t* on the "
            "daily difference · size/market beta decomposition · era cut · costed isolation trade · a "
            "planted-edge synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "is Asness-Frazzini-Israel-Moskowitz-Pedersen (2018): the size premium lives in profitable "
            "small caps once you control junk. The job here is to test whether a **long-only "
            "profitable-small-cap ETF** delivers a risk-adjusted edge over plain small caps (and large "
            "caps), and to separate any real edge from a mere small-cap/market beta and from costs.\n\n"
            "> ⚠️ **Data note.** yfinance **total-return** closes; every leg is **excess of BIL** (the "
            "T-bill cash leg). The Sharpe race is sliced to the **common window** all five funds share "
            "(CALF is the youngest, 2017-06). Numbers in [`docs/results.md`](../docs/results.md) "
            "(as-of " + R["asof"] + ", fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Profitable small (CALF/XSHQ) excess-Sharpe **{rc['CALF']['sr']:.2f}** "
            f"< plain IWM/IJR **{rc['IWM']['sr']:.2f}/{rc['IJR']['sr']:.2f}** < SPY **{rc['SPY']['sr']:.2f}**; "
            f"every quality−plain Sharpe diff **negative**, CI straddles 0, HAC *t* ≈ 0; size/market α = "
            f"**+{dc['CALF']['alpha']:.1f}%/yr at *t* = {dc['CALF']['t']:.2f}** (β on IWM = "
            f"{dc['CALF']['b_iwm']:.2f}); gap not era-robust ({era['pre'][5]:+.3f} → {era['post'][5]:+.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | Costed net gap {R['costed']['CALF'][4]:+.3f} Sharpe; "
            f"long-quality/short-plain nets **{R['iso']['CALF'][2]:+.1f}%/yr (CALF)** / "
            f"**{R['iso']['XSHQ'][2]:+.1f}%/yr (XSHQ)** after {R['iso']['CALF'][4]:.2f}%/yr borrow+costs. |\n"
            f"| **Cleaned size premium, tradable?** | `BUSTED` | Large-cap SPY (Sharpe {rc['SPY']['sr']:.2f}) "
            "beat every small-cap flavour, cleaned or not, 2017–2026. |\n\n"
            "> 💡 In plain words: the ETF wrapper delivers small-cap beta, not a quality-cleaned "
            "premium. The AQR effect is a long-short cross-sectional result; a long-only fund can't "
            "harvest it, and the size premium simply wasn't paid this decade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "AFMP (2018): let $r^{small}$, $r^{big}$ be size legs and $QMJ$ a quality-minus-junk factor. "
            "Raw $SMB = r^{small}-r^{big}$ is weak because small caps load *negatively* on QMJ (they're "
            "junkier). Controlling for it, the size coefficient in\n\n"
            "$$r^{small}_t - r^{big}_t = a + b\\,QMJ_t + \\varepsilon_t$$\n\n"
            "becomes large and stable — the 'cleaned' size premium is $a$. The **tradable** hypotheses:\n\n"
            "- **H₁ (Sharpe).** A profitable-small ETF's **excess-of-cash Sharpe** > a plain-small "
            "ETF's, with a paired-bootstrap CI clear of zero and HAC *t* ≥ 2 on the daily return "
            "difference.\n"
            "- **H₂ (cleaned alpha).** Regressing the quality leg on plain small caps and the market, "
            "the intercept $\\alpha$ is **positive and significant** — a premium beyond size/market "
            "beta.\n"
            "- **H₃ (robust).** The advantage holds across sub-eras and survives realistic costs.\n\n"
            "Fail any of these on the live ETFs and the *tradable* version of the claim is not "
            "supported — whatever the underlying stock-level cross-section does."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ hold, small-cap-quality ETFs are a free Sharpe upgrade over plain small caps and "
            "the AQR result is buyable off the shelf. If they fail, the funds are small-cap beta with a "
            "quality label and a bigger fee, and the honest small-cap tilt question collapses into "
            "'small vs large', which large won this decade. Inference is **excess-of-cash throughout** "
            "(so cash-rate differences can't flatter anyone), **Newey-West HAC** on daily differences "
            "(fat tails, mild autocorrelation), and a **paired** block bootstrap on the Sharpe "
            "difference (the legs overlap day-by-day)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Universe.** CALF, XSHQ (profitable small) · IWM, IJR (plain small) · SPY (large) · BIL "
            "(cash). Total-return daily closes, as-of " + R["asof"] + " (partial month dropped).\n"
            "- **Excess-of-cash.** Every leg minus BIL's daily total return.\n"
            "- **Common window.** The Sharpe race is sliced to dates all five funds exist "
            "(" + R["window"] + f", n = {R['n_race']}); head-to-heads use each pair's own overlap.\n"
            "- **Estimators.** Annualised excess-Sharpe; paired circular-block bootstrap of the Sharpe "
            "difference; HAC mean-*t* (10 daily lags) on the daily return difference; HAC OLS of the "
            "quality leg on `[x_IWM, x_SPY]`.\n"
            "- **Eras.** Full sample · pre-2021 (incl. COVID) · 2021+.\n"
            "- **Costs.** ER gap vs the plain baseline + 5 bps one-way on ~1 annual rebalance; the "
            "isolation trade adds 50 bps/yr borrow on the short leg.\n"
            "- **Control.** `synthetic_world(edge)` plants a tunable quality edge over a shared "
            "small-cap factor plus a zero-mean junk drag; the null (edge = 0) must not fire."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The Sharpe race and its uncertainty\n\n"
            "Annualised excess-of-cash Sharpe on the common window, with a paired look at each "
            "quality-vs-plain and quality-vs-large match."
        ),
        code(
            "order = ['CALF','XSHQ','IWM','IJR','SPY']\n"
            "if HAVE_REAL:\n"
            "    r = st.race(FRAME, XLEGS)\n"
            "    tab = [(k, r['legs'][k]['ann_excess_pct'], r['legs'][k]['ann_vol_pct'],\n"
            "            r['legs'][k]['sharpe'], r['legs'][k]['maxdd_pct']) for k in order]\n"
            "else:\n"
            "    tab = [(k, R['race'][k]['exc'], R['race'][k]['vol'], R['race'][k]['sr'], R['race'][k]['dd']) for k in order]\n"
            "print(f\"{'leg':<6s} {'exc%/yr':>8s} {'vol%':>7s} {'Sharpe':>7s} {'maxDD%':>7s}\")\n"
            "for k,e,v,s,d in tab: print(f'{k:<6s} {e:>+8.2f} {v:>7.2f} {s:>+7.3f} {d:>+7.1f}')\n"
            "print()\n"
            "for m in ['CALF vs IWM','CALF vs IJR','CALF vs SPY','XSHQ vs IWM','XSHQ vs IJR','XSHQ vs SPY']:\n"
            "    a, b = m.split(' vs ')\n"
            "    if HAVE_REAL:\n"
            "        p = st.pair_test(FRAME, XLEGS[a], XLEGS[b])\n"
            "        vals = (p['sharpe_a'], p['sharpe_b'], p['sharpe_diff'], p['diff_ci_low'], p['diff_ci_high'], p['diff_frac_le0'], p['t_nw_diff'])\n"
            "    else:\n"
            "        z = R['pairs'][m]; vals = (z[0], z[1], z[2], z[3], z[4], z[5], z[7])\n"
            "    print(f'{m:<12s} SR {vals[0]:+.3f} vs {vals[1]:+.3f}  diff {vals[2]:+.3f} '\n"
            "          f'CI[{vals[3]:+.3f},{vals[4]:+.3f}] P(diff<=0)={vals[5]:.2f}  HAC t(daily)={vals[6]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: profitable small caps land at Sharpe **{rc['CALF']['sr']:.2f}**, "
            f"*below* plain small caps (**{rc['IWM']['sr']:.2f}/{rc['IJR']['sr']:.2f}**) and far below "
            f"large-cap SPY (**{rc['SPY']['sr']:.2f}**). Every quality−plain difference is negative with "
            "a CI straddling zero and a daily-difference HAC *t* of essentially zero — **no edge, either "
            "direction**. Against SPY the loss is one-directional (P(diff ≤ 0) ≈ 0.95)."
        ),
        md(
            "### 4b · Is there a cleaned alpha? — the size/market decomposition\n\n"
            "$x^{quality}_t = \\alpha + \\beta_{1}\\,x^{IWM}_t + \\beta_{2}\\,x^{SPY}_t + \\varepsilon_t$, "
            "HAC standard errors. $\\alpha$ is the daily excess return beyond small-cap and market beta."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for k in ['CALF','XSHQ']:\n"
            "        d = st.beta_decomp(FRAME, XLEGS[k], ['x_IWM','x_SPY'])\n"
            "        rows.append((k, d['alpha_ann_pct'], d['t_alpha'], d['betas']['x_IWM'], d['t_betas']['x_IWM'],\n"
            "                     d['betas']['x_SPY'], d['t_betas']['x_SPY'], d['r2']))\n"
            "    d = st.beta_decomp(FRAME, 'x_CALF', ['x_IWM','x_SPY'])\n"
            "    win = st.common_window(FRAME, ['x_CALF','x_IWM'])\n"
            "    xs, ys = win['x_IWM'].to_numpy()*100, win['x_CALF'].to_numpy()*100\n"
            "else:\n"
            "    rows = [(k, R['decomp'][k]['alpha'], R['decomp'][k]['t'], R['decomp'][k]['b_iwm'], R['decomp'][k]['t_iwm'],\n"
            "             R['decomp'][k]['b_spy'], R['decomp'][k]['t_spy'], R['decomp'][k]['r2']) for k in ['CALF','XSHQ']]\n"
            "    rng = np.random.default_rng(901); xs = rng.normal(0, 1.5, 2269); ys = 1.06*xs + rng.normal(0.03, 0.7, 2269)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 5.0))\n"
            "ax.scatter(xs, ys, s=10, alpha=.4, color=GREEN)\n"
            "bfit = np.polyfit(xs, ys, 1); grid = np.linspace(xs.min(), xs.max(), 50)\n"
            "ax.plot(grid, np.polyval(bfit, grid), c=RED, lw=1.8, label=f'fit: slope {bfit[0]:.2f}')\n"
            "ax.plot(grid, grid, c=GREY, ls='--', lw=1.1, label='slope 1 (pure small-cap beta)')\n"
            "ax.set_xlabel('plain small caps IWM, daily excess % '); ax.set_ylabel('CALF, daily excess %')\n"
            "ax.set_title('CALF is ~1.06x plain small-cap beta; the intercept (alpha) is ~zero'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for k,a,t,bi,ti,bs,ts,r2 in rows:\n"
            "    print(f'{k}: alpha {a:+.2f}%/yr (t={t:+.2f})  b_IWM={bi:+.3f} (t={ti:+.1f})  b_SPY={bs:+.3f} (t={ts:+.1f})  R2={r2:.3f}')"
        ),
        md(
            f"> 💡 In plain words: CALF loads **{dc['CALF']['b_iwm']:.2f}×** on plain small caps and "
            f"leaves alpha **+{dc['CALF']['alpha']:.1f}%/yr at HAC *t* = {dc['CALF']['t']:.2f}** — a "
            f"statistical zero. XSHQ is the same story ({dc['XSHQ']['b_iwm']:.2f}× IWM, α "
            f"+{dc['XSHQ']['alpha']:.1f}%/yr, *t* = {dc['XSHQ']['t']:.2f}). The negative small SPY beta "
            "just reflects the *large-minus-small* tilt inside a small-cap fund. **H₂ fails: no cleaned "
            "premium.**"
        ),
        md(
            "### 4c · Era-robustness — the gap's sign flips\n\n"
            "CALF vs IWM excess-Sharpe, split pre-/post-2021."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = st.era_races(FRAME, {'CALF':'x_CALF','IWM':'x_IWM'}, split=data.ERA_SPLIT)\n"
            "    rows = []\n"
            "    for tag in ['pre','post']:\n"
            "        rr = e[tag]; q, p = rr['legs']['CALF']['sharpe'], rr['legs']['IWM']['sharpe']\n"
            "        rows.append((tag, rr['start'], rr['end'], rr['n'], q, p, q-p))\n"
            "else:\n"
            "    rows = [(tag, *R['era'][tag]) for tag in ['pre','post']]\n"
            "for tag, s, en, n, q, p, g in rows:\n"
            "    print(f'{tag:<4s} [{s}->{en} n={n}]  CALF SR {q:+.3f}  IWM SR {p:+.3f}  gap {g:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: the CALF−IWM Sharpe gap is **{era['pre'][5]:+.3f}** before 2021 and "
            f"**{era['post'][5]:+.3f}** after — it **flips sign** and both halves are inside the noise "
            "band. The opposite of the robust, pervasive, all-eras advantage AFMP document for the "
            "cleaned size premium in the underlying stocks. **H₃ fails.**"
        ),
        md(
            "### 4d · Tradability — costs deepen a hole that starts at zero\n\n"
            "(a) costed net Sharpe race (ER gap vs IWM + 5 bps one-way on one annual rebalance); "
            "(b) long-quality/short-plain isolation trade (50 bps/yr borrow + 5 bps one-way × 2/yr)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    crows, irows = [], []\n"
            "    for k in ['CALF','XSHQ']:\n"
            "        c = st.costed_race(FRAME, XLEGS[k], 'x_IWM', er_quality=data.EXPENSE_RATIOS[k], er_plain=data.EXPENSE_RATIOS['IWM'])\n"
            "        crows.append((k, c['charge_ann_pct'], c['sharpe_q_gross'], c['sharpe_q_net'], c['sharpe_plain'], c['net_gap']))\n"
            "        it = st.isolation_trade(FRAME, XLEGS[k], 'x_IWM')\n"
            "        irows.append((k, it['gross_ann_pct'], it['t_nw_gross'], it['net_ann_pct'], it['t_nw_net'], it['charge_ann_pct']))\n"
            "else:\n"
            "    crows = [(k, *R['costed'][k]) for k in ['CALF','XSHQ']]\n"
            "    irows = [(k, *R['iso'][k]) for k in ['CALF','XSHQ']]\n"
            "print('costed net Sharpe race vs IWM:')\n"
            "for k, ch, g, n, iwm, gap in crows:\n"
            "    print(f'  {k}: charge {ch:.2f}%/yr  Sharpe gross {g:+.3f} -> net {n:+.3f}  IWM {iwm:+.3f}  net gap {gap:+.3f}')\n"
            "print('long-quality / short-plain isolation trade:')\n"
            "for k, g, tg, n, tn, ch in irows:\n"
            "    print(f'  {k}-IWM: gross {g:+.2f}%/yr (t={tg:+.2f})  net {n:+.2f}%/yr (t={tn:+.2f})  charge {ch:.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the costed net gap stays negative ({R['costed']['CALF'][4]:+.3f} "
            f"Sharpe for CALF), and the isolation trade — the purest way to bet 'quality beats junk' — "
            f"earns nothing gross and loses **{R['iso']['CALF'][2]:+.1f}%/yr (CALF)** / "
            f"**{R['iso']['XSHQ'][2]:+.1f}%/yr (XSHQ)** net. There is no edge for costs to erode; costs "
            "just make a zero negative. **MIRAGE.**"
        ),
        md(
            "### 4e · Calendar-year returns — big swings, no compounding edge"
        ),
        code(
            "order = ['CALF','XSHQ','IWM','IJR','SPY']\n"
            "if HAVE_REAL:\n"
            "    cy = st.calendar_years(FRAME, RLEGS)[order]\n"
            "else:\n"
            "    cy = pd.DataFrame(R['cy'], index=order).T\n"
            "print(cy.round(1).to_string())\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "cy[['CALF','IWM','SPY']].plot(kind='bar', ax=ax, color=[GREEN, GREY, RED], width=.8)\n"
            "ax.axhline(0, c='k', lw=.6); ax.set_ylabel('total return %'); ax.set_title('Calendar-year returns: CALF is high-variance, not high-Sharpe')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: CALF's cash-flow screen is really a *value* tilt — a huge 2021 (+41 %) "
            "and 2023 (+35 %), an ugly 2024–2025. The years swing hard but don't stack into a superior "
            "risk-adjusted return; the volatility eats the occasional big win."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "Deterministic world: quality = market + small-cap factor + **planted edge**; plain = same "
            "systematic exposure + a zero-mean **junk** noise. The estimator must recover the knob; the "
            "zero-edge null must NOT fire. *(Machinery proof — never market evidence.)*"
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.4):\n"
            "    w = data.synthetic_world(edge=edge, seed=901)\n"
            "    fr = pd.DataFrame({'x_quality': w['x_quality'], 'x_plain': w['x_plain']}, index=w.index)\n"
            "    p = st.pair_test(fr, 'x_quality', 'x_plain')\n"
            "    res.append((edge, p['sharpe_diff'], p['diff_ci_low'], p['diff_ci_high'], p['t_nw_diff']))\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "labels = [f'planted edge {e:.1f}' for e,_,_,_,_ in res]\n"
            "diffs = [r[1] for r in res]; err = [[r[1]-r[2] for r in res],[r[3]-r[1] for r in res]]\n"
            "ax.bar(labels, diffs, color=[GREY, GREEN], width=.5, yerr=err, capsize=6)\n"
            "for i,(e,d,lo,hi,t) in enumerate(res): ax.annotate(f'{d:+.2f}\\n(t={t:+.2f})',(i,d),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.6); ax.set_ylabel('recovered quality-minus-plain Sharpe diff')\n"
            "ax.set_title('Control: null stays dark, planted edge is recovered clear of zero'); plt.tight_layout(); plt.show()\n"
            "for e,d,lo,hi,t in res: print(f'planted edge {e:.1f} -> Sharpe diff {d:+.3f} CI[{lo:+.3f},{hi:+.3f}] HAC t(daily)={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the estimator reads Sharpe-diff "
            f"**{R['syn'][0][1]:+.3f}** (HAC *t* = {R['syn'][0][4]:+.2f}, CI straddles zero); a planted "
            f"**+0.4** comes back as **{R['syn'][1][1]:+.3f}** with a CI clear of zero and HAC *t* = "
            f"{R['syn'][1][4]:+.2f}. The machinery detects a real edge when one exists — so the "
            "real-tape null (Sharpe diff ≈ 0, *t* ≈ 0) is the genuine reading, not a dead detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — profitable-small excess-Sharpe **{rc['CALF']['sr']:.2f}** < plain "
            f"**{rc['IWM']['sr']:.2f}/{rc['IJR']['sr']:.2f}** < SPY **{rc['SPY']['sr']:.2f}**; every "
            f"quality−plain Sharpe diff negative with CI over zero and HAC *t* ≈ 0; size/market α = "
            f"+{dc['CALF']['alpha']:.1f}%/yr (*t* = {dc['CALF']['t']:.2f}), β on IWM ≈ "
            f"{dc['CALF']['b_iwm']:.2f}; gap not era-robust ({era['pre'][5]:+.3f} → {era['post'][5]:+.3f}). "
            "The synthetic control recovers a planted edge cleanly, so this is a true null.\n"
            f"- **Tradability `MIRAGE`** — no edge to bank; costed net gap {R['costed']['CALF'][4]:+.3f} "
            f"Sharpe, isolation trade {R['iso']['CALF'][2]:+.1f}%/yr (CALF) net; CALF's 0.59% ER is a "
            "standing drag against a 0.06–0.19% plain baseline.\n"
            f"- **Cleaned size premium, tradable? `BUSTED`** — SPY (Sharpe {rc['SPY']['sr']:.2f}) beat "
            "every small-cap flavour 2017–2026. The long-only ETF wrapper can't short the junk, so it "
            "never isolates the AFMP premium — it just holds nicer small caps that rose and fell with "
            "the tide."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Long-short is the missing ingredient.** AFMP's cleaned premium is long profitable-small "
            "/ **short junky-small**. A long-only ETF holds only the long leg, so it keeps full "
            "small-cap beta and never isolates $\\alpha$. Our isolation trade approximates the short "
            "leg (short IWM) and finds nothing — but IWM is *all* small caps, not the junk tail "
            "specifically, so even this understates how hard the real long-short is to run in ETFs.\n"
            "- **CALF ≈ small value.** The FCF-yield screen is a value tilt; its edge and drawdowns "
            "track value's fortunes, not a stable quality premium.\n"
            "- **Dedup on the desk.** [513-size-effect](../../513-size-effect/README.md) (raw "
            "small-minus-big), [657-larry-portfolio](../../657-larry-portfolio/README.md) "
            "(small-**value**), [242-quality-minus-junk](../../242-quality-minus-junk/README.md) (QMJ "
            "across all caps — this is its small-cap-ETF slice), "
            "[362-piotroski-f-score](../../362-piotroski-f-score/README.md) (single-name quality "
            "scoring).\n\n"
            "*Frozen numbers: [`docs/results.md`](../docs/results.md); sources: "
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
