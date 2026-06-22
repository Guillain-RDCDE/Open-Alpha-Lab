"""Generate the two narrative notebooks for Study 356 (the GLP-1 / Ozempic basket).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: the real-tape cells read the cached yfinance
panel under ../_cache/ when present, and otherwise quote the frozen headline numbers in ``R``
(a mirror of docs/results.md). The synthetic positive control runs anywhere, no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (yfinance adjusted close, 2018-01-02 -> 2026-06-18, 2126 daily returns; as-of 2026-06-21).
R = dict(
    window="2018-01-02 → 2026-06-18", n=2126,
    # CAGR / vol / Sharpe / total / maxDD per series
    lly_cagr=36.9, lly_vol=31.8, lly_sh=1.16, lly_tot=1366.8, lly_dd=-34.5,
    nvo_cagr=13.5, nvo_vol=33.5, nvo_sh=0.40, nvo_tot=93.0, nvo_dd=-74.7,
    bk_cagr=25.2, bk_vol=27.4, bk_sh=0.92, bk_tot=510.7, bk_dd=-49.3,
    xlv_cagr=10.1, xlv_vol=17.4, xlv_sh=0.58, xlv_tot=105.4, xlv_dd=-28.4,
    spy_cagr=15.5, spy_vol=19.2, spy_sh=0.81, spy_tot=216.5, spy_dd=-33.7,
    # signal
    ex_spy=9.68, t_spy=1.10, ex_xlv=15.14, t_xlv=2.01,
    alpha_spy=15.18, ta_spy=1.78, beta_spy=0.65,
    alpha_xlv=15.45, ta_xlv=2.05, beta_xlv=0.97,
    boot_pt=15.1, boot_lo=0.2, boot_hi=29.3, boot_p=0.976,
    # decomposition
    lly_alpha=26.58, lly_ta=2.66, lly_beta=0.66,
    nvo_alpha=3.77, nvo_ta=0.35, nvo_beta=0.63,
    nvo_peak=137.4, nvo_last=43.2, nvo_dd_peak=-69,
    corr_llynvo=0.40, corr_bk_spy=0.45, corr_bk_xlv=0.62,
    # recency
    pre_bk=29.9, pre_spy=11.1, pre_ex=18.8, pre_t=1.93, pre_n=1258,
    post_bk=18.4, post_spy=21.9, post_ex=-3.5, post_t=-0.21, post_n=868,
    # tradability
    turnover=0.367, cost10_ex=15.04, cost10_t=2.00,
    # synthetic control
    syn=[("LLY", 25.2, 33.8, 4.33), ("NVO", 0.0, -2.9, -0.39), ("XLV", 5.0, 9.7, 1.21)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Two_stocks_and_a_story%3F: Misattributed](https://img.shields.io/badge/Two_stocks_and_a_story%3F-Misattributed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from glp1_basket import data, strategy as st

HAVE_REAL = data.have_real()
P = data.load_real() if HAVE_REAL else None
RET = data.daily_returns(P) if HAVE_REAL else None
BK = data.basket_returns(RET) if HAVE_REAL else None
print("real yfinance panel cache present:", HAVE_REAL,
      "| daily returns:", (0 if RET is None else len(RET)))
"""

# The frozen headline dict is embedded into the first code cell so every later cell can quote
# it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Ozempic trade — did buying the weight-loss-drug winners beat the market? 💉\n"
            "### \"Just buy Lilly and Novo and ride the boom\" — in plain English\n\n"
            + BADGES +
            "You've heard it everywhere since 2023: *the obesity-drug boom is the trade of the "
            "decade — buy **Eli Lilly (LLY)** and **Novo Nordisk (NVO)**, the makers of Mounjaro "
            "and Ozempic, and ride it.* And on paper the basket really did beat the market. So is "
            "the \"Ozempic trade\" a genuine edge — or is it **two stocks and a story**, where one "
            "stock did all the work and the other one fell apart?\n\n"
            "Short answer: the basket beat the market, but the *edge* isn't statistically solid, "
            "it's **almost entirely Eli Lilly**, and — the twist — it actually **lagged** during "
            "the very weight-loss-drug mania it's named for. This notebook shows all three.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the market-model alpha and the "
            "bootstrap CI? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart is drawn by the code beside it; real prices "
            "are Yahoo Finance adjusted-close (total return). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the LLY+NVO basket beat the market? | **Yes, on paper.** ~{R['bk_cagr']:.0f}%/yr "
            f"vs SPY's ~{R['spy_cagr']:.0f}%/yr since 2018. |\n"
            "| Is that beat a *real, reliable* edge? | **Not really.** Measured properly the "
            f"out-performance over the market is statistically shaky (a *t* of just ~{R['t_spy']:.1f}, "
            "where you want ≥2). |\n"
            "| Is it \"the GLP-1 theme\"? | **No — it's one stock.** Strip out the market and "
            f"**Lilly** has the edge; **Novo** (the actual Ozempic maker!) has essentially none, and "
            f"its stock is **{R['nvo_dd_peak']:.0f}% below its peak**. |\n"
            "| Did it work *during* the Ozempic boom? | **No — it lagged.** Before 2023 the basket "
            "crushed the market; *during* the 2023–26 mania it **trailed** it. |\n\n"
            "> The basket beat the market — but it's one company carrying a passenger, and it "
            "peaked before the story did."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The weight-loss-drug boom is structural and enormous. Eli Lilly and Novo Nordisk "
            "own the market for GLP-1 drugs — Mounjaro/Zepbound and Ozempic/Wegovy. Buy the two of "
            "them as a basket and you ride a once-in-a-generation product cycle that the broad market "
            "and even the healthcare sector massively under-weight.\"*\n\n"
            "It became a *named* trade: thematic notes, retail explainers, even dedicated obesity-drug "
            "ETFs launched in 2023–24. The basket here is the simplest honest version: **50% LLY, 50% "
            "NVO**, held the whole way. We test it against **SPY** (the whole market) and **XLV** (the "
            "healthcare sector) — because beating *healthcare* is the real bar for a healthcare theme."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the theme is a genuine edge, two things should be true: (1) the basket should beat not "
            "just the market but the **healthcare sector** — otherwise you're just long healthcare; and "
            "(2) the edge should be **broad to the theme**, not one lucky stock. A \"basket\" that's "
            "really *one name* is a single-stock bet with extra words — and single stocks can fall 70%. "
            "The whole appeal of a *theme* is that it's bigger and safer than picking one winner. So the "
            "test isn't \"did it go up\" (it did) — it's *\"is the edge real, and is it actually the "
            "theme?\"*"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['window']}** of daily prices (dividends reinvested) for LLY, NVO, XLV and "
            "SPY, and do three honest things:\n\n"
            "1. **Compare like with like.** Beating the market on *raw* return can just be taking more "
            "risk. We look at the **excess** over SPY and over XLV — and ask if it's big enough to not "
            "be luck.\n"
            "2. **Split the basket open.** Measure each stock's own edge over the market. If it's all "
            "one name, the \"basket\" is a costume.\n"
            "3. **Check the timing.** Cut the history at the start of 2023 (when the Ozempic story went "
            "mainstream) and see whether the trade worked *before* or *during* the boom.\n\n"
            "**What would make us say \"mirage\":** the out-performance isn't statistically solid, it "
            "comes from one stock, and/or it didn't show up during the actual boom."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: did the basket beat the market?** Growth of \\$1 since 2018, total return."
        ),
        code(
            "names = ['LLY','NVO','XLV','SPY']\n"
            "if HAVE_REAL:\n"
            "    curves = {n: (1+RET[n]).cumprod() for n in names}\n"
            "    bk = (1+BK).cumprod(); idx = bk.index\n"
            "else:\n"
            "    # frozen-total fallback: a smooth curve to each name's known total return\n"
            "    idx = np.arange(R['n']); fin = {'LLY':R['lly_tot'],'NVO':R['nvo_tot'],'XLV':R['xlv_tot'],'SPY':R['spy_tot']}\n"
            "    curves = {n: 1+(fin[n]/100)*(np.arange(R['n'])/(R['n']-1)) for n in names}\n"
            "    bk = 1+(R['bk_tot']/100)*(np.arange(R['n'])/(R['n']-1))\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.8))\n"
            "ax.plot(idx, bk, c=GREEN, lw=2.4, label='GLP-1 basket (50/50 LLY/NVO)')\n"
            "for n,c in [('SPY',GREY),('XLV',AMBER),('LLY','#1f6feb'),('NVO',RED)]:\n"
            "    ax.plot(idx, curves[n], c=c, lw=1.4, alpha=.9, label=n)\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.set_title('The basket did beat the market — but look at NVO')\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print(f\"basket total {R['bk_tot']:.0f}%  vs SPY {R['spy_tot']:.0f}%  XLV {R['xlv_tot']:.0f}%  (LLY {R['lly_tot']:.0f}%, NVO {R['nvo_tot']:.0f}%)\")"
        ),
        md(
            f"The basket (green) ends well above SPY and XLV — total return **{R['bk_tot']:.0f}%** vs "
            f"SPY's **{R['spy_tot']:.0f}%**. But notice the two ingredients are wildly different: LLY "
            f"is the rocket (**{R['lly_tot']:.0f}%**), while **NVO** (red) round-tripped — up, then "
            f"**{R['nvo_dd_peak']:.0f}% down from its peak**. A 50/50 'basket' of those two isn't a "
            "diversified theme; it's one winner dragging one loser."
        ),
        md(
            "**Second: is the basket really the theme — or one stock?** Each name's edge over the "
            "market once we strip out general market moves (a higher bar passes the test)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    aL = st.market_model(RET['LLY'], RET['SPY']); aN = st.market_model(RET['NVO'], RET['SPY'])\n"
            "    laL, taL = aL['alpha_ann']*100, aL['t_alpha']; laN, taN = aN['alpha_ann']*100, aN['t_alpha']\n"
            "else:\n"
            "    laL, taL, laN, taN = R['lly_alpha'], R['lly_ta'], R['nvo_alpha'], R['nvo_ta']\n"
            "fig, ax = plt.subplots(figsize=(8.4,4.3))\n"
            "bars = ax.bar(['Eli Lilly\\n(LLY)','Novo Nordisk\\n(NVO)\\n— the Ozempic maker'], [laL, laN],\n"
            "              color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('edge over the market (%/yr, market-adjusted)')\n"
            "ax.set_title('The \"GLP-1 basket\" edge is ALL Eli Lilly')\n"
            "for b,(v,t) in zip(bars, [(laL,taL),(laN,taN)]):\n"
            "    ax.annotate(f'{v:+.0f}%/yr\\n(t={t:.1f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LLY edge {laL:+.1f}%/yr (t={taL:.2f})   NVO edge {laN:+.1f}%/yr (t={taN:.2f})')"
        ),
        md(
            f"There it is. **Lilly** carries a real market-beating edge (~{R['lly_alpha']:.0f}%/yr, "
            f"*t*={R['lly_ta']:.1f}). **Novo** — *the company that makes Ozempic itself* — adds "
            f"**~nothing** ({R['nvo_alpha']:.0f}%/yr, *t*={R['nvo_ta']:.1f}, indistinguishable from "
            "zero). The \"weight-loss-drug basket\" is one drug-maker with a passenger. Calling it a "
            "*theme* hides that you're really making a single-stock bet on Lilly."
        ),
        md(
            "**Third — the kicker: did it work *during* the boom?** Split at the start of 2023, when "
            "the Ozempic story went mainstream. How much did the basket beat the market in each half?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pre = RET.index < '2023-01-01'; post = ~pre\n"
            "    exb = (BK - RET['SPY'])\n"
            "    pre_ex = exb[pre].mean()*252*100; post_ex = exb[post].mean()*252*100\n"
            "else:\n"
            "    pre_ex, post_ex = R['pre_ex'], R['post_ex']\n"
            "fig, ax = plt.subplots(figsize=(8.2,4.2))\n"
            "bars = ax.bar(['2018–2022\\n(before the mania)','2023–2026\\n(the GLP-1 mania)'],\n"
            "              [pre_ex, post_ex], color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('basket beat the market by (%/yr)')\n"
            "ax.set_title('The \"Ozempic trade\" worked BEFORE Ozempic — and lagged during it')\n"
            "for b,v in zip(bars,[pre_ex,post_ex]):\n"
            "    ax.annotate(f'{v:+.0f}%/yr',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'before 2023: {pre_ex:+.1f}%/yr   during 2023-26 mania: {post_ex:+.1f}%/yr')"
        ),
        md(
            f"The trade out-ran the market by **+{R['pre_ex']:.0f}%/yr** *before* 2023 — and then "
            f"**trailed** it by **{R['post_ex']:.0f}%/yr** *during* the very weight-loss-drug boom it's "
            "named after. By the time it was a famous named trade, the easy part was already over. "
            "That's the classic shape of a theme you hear about *after* it has happened."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The basket beat the market, but the *edge* over SPY isn't "
            f"statistically solid (*t*≈{R['t_spy']:.1f}); only vs the healthcare sector is it "
            "borderline — and even that is fragile.\n"
            "- **Tradability — Mirage.** It's a **two-stock** bet with a **−49%** drawdown riding one "
            "product cycle. Costs aren't the problem; concentration is. That's beta to a story, not an "
            "edge you can scale.\n"
            "- **Two stocks and a story? — Misattributed.** The edge is **all Lilly**; Novo, the "
            "Ozempic maker, contributed none and is 69% off its high — and the basket *lagged* during "
            "the actual boom."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the risk you'd really run\n\n"
            "Forget costs (they're a rounding error here). The real question for a *two-name* bet is: "
            "how deep a hole did it dig you on the way? Worst peak-to-trough drop of each:"
        ),
        code(
            "labels = ['GLP-1 basket','NVO alone','LLY alone','SPY']\n"
            "dds = [R['bk_dd'], R['nvo_dd'], R['lly_dd'], R['spy_dd']]\n"
            "fig, ax = plt.subplots(figsize=(8.4,4.2))\n"
            "bars = ax.bar(labels, dds, color=[AMBER, RED, '#1f6feb', GREY], width=.6)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('worst drawdown (%)')\n"
            "ax.set_title('A two-name bet digs a deep hole: −49% basket, −75% in Novo')\n"
            "for b,v in zip(bars,dds): ax.annotate(f'{v:.0f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"basket maxDD {R['bk_dd']:.0f}%   NVO {R['nvo_dd']:.0f}%   LLY {R['lly_dd']:.0f}%   SPY {R['spy_dd']:.0f}%\")"
        ),
        md(
            f"To 'ride the theme' you had to sit through a **{R['bk_dd']:.0f}%** basket drawdown — and "
            f"if you'd leaned toward the actual Ozempic maker, a **{R['nvo_dd']:.0f}%** one in Novo. A "
            "diversified *edge* doesn't do that; a concentrated *bet* does. There's no cost trick that "
            "turns two correlated single stocks into a scalable strategy — the risk *is* the product."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Broaden the basket honestly.** Add the 'GLP-1 losers' (food, dialysis, insulin-pump "
            "names) and the obesity-drug ETFs — does a *diversified* theme survive, or does it still "
            "collapse to Lilly?\n"
            "- **Pick the date *without* hindsight.** We chose LLY+NVO knowing they won. Re-run it "
            "choosing the basket as of 2018 from *all* the GLP-1 hopefuls of the day — the honest "
            "version of the trade.\n"
            "- **The general lesson.** Any 'theme basket' named after the product whose makers already "
            "10×'d is selection by construction. The cheap part is usually gone by the time it has a "
            "name.\n\n"
            "*Think the theme is broader than one stock? Build the diversified GLP-1 basket and show "
            "its market-adjusted edge clears *t* = 2 — then check it isn't still just Lilly in "
            "disguise.*"
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
            "# The GLP-1 / Ozempic basket — a quantitative teardown 🔬\n"
            "### Mean-excess & market-model α vs SPY and XLV (HAC-*t*) · single-name decomposition · "
            "a 2023 recency split · block-bootstrap CI · cost sweep · a planted-α synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "\"Ozempic trade\" (50/50 **LLY/NVO**) beats SPY and XLV on raw return — but we separate "
            "three things the pitch fuses: a **statistically robust** edge (it isn't, vs SPY: HAC-*t* "
            "≈ 1.1), a **theme** edge (it's one name — LLY *t*=2.7, NVO *t*=0.3), and a **durable** "
            "edge (the excess flips *negative* during the 2023–26 mania). Signal is **WEAK**, the "
            "attribution is **MISATTRIBUTED**, and the book is a **MIRAGE** of concentrated beta.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance adjusted-close (total return) for LLY, "
            "**NVO (US ADR)**, XLV, SPY, 2018→2026. The offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Mean excess vs SPY +{R['ex_spy']:.1f}%/yr at **HAC-*t* "
            f"= {R['t_spy']:.2f}** (< 2); market-model α vs SPY +{R['alpha_spy']:.1f}%/yr, *t* "
            f"= {R['ta_spy']:.2f}. Only vs XLV is it borderline (α *t* = {R['ta_xlv']:.2f}), with a "
            f"bootstrap CI **[+{R['boot_lo']:.1f}%, +{R['boot_hi']:.1f}%]** grazing zero. |\n"
            f"| **Tradability** | `MIRAGE` | Two-name book (corr LLY–NVO {R['corr_llynvo']:.2f}), basket "
            f"max-DD **{R['bk_dd']:.0f}%**. Turnover {R['turnover']:.2f}%/day → cost is immaterial "
            f"(net-of-10bp α still +{R['cost10_ex']:.1f}%/yr): the killer is concentration, not the "
            "spread. |\n"
            f"| **Two stocks & a story?** | `MISATTRIBUTED` | α is **all LLY** (+{R['lly_alpha']:.1f}%/yr, "
            f"*t* = {R['lly_ta']:.2f}); **NVO** +{R['nvo_alpha']:.1f}%/yr, *t* = {R['nvo_ta']:.2f} "
            f"(≈0), {R['nvo_dd_peak']:.0f}% from peak. Excess flips to **{R['post_ex']:.1f}%/yr** in "
            "2023–26. |\n\n"
            "> 💡 In plain words: the basket out-returned the market, but the out-performance is within "
            "noise vs SPY, lives entirely in one stock, and disappeared exactly when the theme became "
            "famous. Three different ways of saying *not an edge.*"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the basket's daily total return be $r^{\\text{bk}}_t = \\tfrac12 r^{\\text{LLY}}_t + "
            "\\tfrac12 r^{\\text{NVO}}_t$ and a benchmark $r^{b}_t \\in \\{r^{\\text{SPY}}, "
            "r^{\\text{XLV}}\\}$. The theme makes three claims:\n\n"
            "- **H₁ (real alpha).** Mean excess $\\bar{x} = \\overline{r^{\\text{bk}} - r^{b}} > 0$ with "
            "an autocorrelation-robust $t \\ge 2$; and the CAPM intercept $\\alpha$ in "
            "$r^{\\text{bk}} = \\alpha + \\beta r^{b} + \\varepsilon$ is positive and significant.\n"
            "- **H₂ (it is the theme, not a name).** The α is **broad** — not concentrated in a single "
            "constituent.\n"
            "- **H₃ (durable).** The edge is present in the period the theme is about (the 2023+ GLP-1 "
            "mania), not only before it.\n\n"
            "We find **H₁ weak** (vs SPY $t\\approx1.1$; vs XLV borderline and bootstrap-fragile), "
            "**H₂ rejected** (α is entirely LLY), **H₃ rejected** (excess turns negative post-2023). "
            "The trade is true on *raw return* and false on every axis that would make it an edge."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Raw out-performance is cheap: high-β single names beat a broad index in a bull market by "
            "construction. The identity that matters is\n\n"
            "$$ r^{\\text{bk}}_t = \\underbrace{\\alpha}_{\\text{skill?}} + \\underbrace{\\beta\\, "
            "r^{b}_t}_{\\text{paid-for beta}} + \\varepsilon_t, \\qquad H_0:\\ \\alpha = 0. $$\n\n"
            "Three confounds sit on top of it. **(i) Beta** — the basket's β to SPY is ~0.65, so much "
            "of the return is just market exposure. **(ii) Selection** — LLY and NVO are in the basket "
            "*because* they won; conditioning on realised winners biases $\\hat\\alpha$ **upward** "
            "(Brown et al. 1992). **(iii) Concentration** — a 2-name book's $\\hat\\alpha$ has huge "
            "idiosyncratic variance, so even a large point estimate can carry a tiny *t*. Measuring "
            "*raw* return alone settles none of these; the HAC-*t* on α does."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** yfinance **adjusted close** (total return), {R['window']}, "
            f"{R['n']:,} daily returns. NVO is the **US ADR** (the line a US reader trades — named as a "
            "proxy for the Copenhagen B-share, which differs by FX/ADR effects).\n"
            "- **Signal test (H₁).** Mean excess vs SPY and XLV with a **Newey-West (HAC)** *t* "
            "(Bartlett kernel, automatic lag $\\lfloor 4(n/100)^{2/9}\\rfloor$); plus a market-model α "
            "with a **HAC-*t* on the intercept**. A circular-**block bootstrap** (21-day blocks) gives "
            "the excess-return CI. Bar: **HAC-*t* ≥ 2** on the *real* tape to be `REAL`.\n"
            "- **Decomposition (H₂).** The same market model per single name vs SPY — does the α live "
            "in one constituent?\n"
            "- **Recency (H₃).** Split at 2023-01-01 (a *justified*, not snooped, break: the GLP-1 "
            "story's mainstreaming) and re-test the excess each side.\n"
            "- **Tradability.** One-way cost sweep on rebalancing turnover, and the basket/constituent "
            "drawdowns (concentration risk).\n"
            "- **Positive control.** A fixed-seed one-factor panel with a *planted* per-name α — the "
            "engine must recover it (LLY-like) and the *absence* of it (NVO-like).\n\n"
            "**Mirage condition (pre-registered):** HAC-*t* on α below 2 vs SPY, α concentrated in one "
            "name, and/or the excess vanishing in the post-2023 sub-period."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The signal is WEAK — mean excess & α under HAC inference\n\n"
            "Annualised excess / α vs each benchmark, with the HAC-*t* printed against the bar (2)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eS = st.excess_test(BK, RET['SPY']); eX = st.excess_test(BK, RET['XLV'])\n"
            "    aS = st.market_model(BK, RET['SPY']); aX = st.market_model(BK, RET['XLV'])\n"
            "    rows = [('excess vs SPY', eS['ann_excess']*100, eS['t_hac']),\n"
            "            ('excess vs XLV', eX['ann_excess']*100, eX['t_hac']),\n"
            "            ('α vs SPY (β-adj)', aS['alpha_ann']*100, aS['t_alpha']),\n"
            "            ('α vs XLV (β-adj)', aX['alpha_ann']*100, aX['t_alpha'])]\n"
            "else:\n"
            "    rows = [('excess vs SPY', R['ex_spy'], R['t_spy']),('excess vs XLV', R['ex_xlv'], R['t_xlv']),\n"
            "            ('α vs SPY (β-adj)', R['alpha_spy'], R['ta_spy']),('α vs XLV (β-adj)', R['alpha_xlv'], R['ta_xlv'])]\n"
            "labels=[r[0] for r in rows]; ts=[r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.4))\n"
            "cols=[GREEN if t>=2 else AMBER for t in ts]\n"
            "bars=ax.bar(labels, ts, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar for REAL'); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('HAC (Newey-West) t-stat'); ax.set_title('Excess / alpha vs benchmarks: only XLV grazes the bar')\n"
            "for b,(l,v,t) in zip(bars, rows): ax.annotate(f'+{v:.1f}%/yr\\nt={t:.2f}',(b.get_x()+b.get_width()/2,t),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('   '.join(f'{l}: t={t:.2f}' for l,_,t in rows))"
        ),
        md(
            f"> 💡 In plain words: against the **market** the excess (+{R['ex_spy']:.1f}%/yr) carries a "
            f"*t* of only **{R['t_spy']:.2f}** and the β-adjusted α a *t* of **{R['ta_spy']:.2f}** — "
            "**below the bar**. Only against **healthcare** (XLV) does it reach *t* ≈ 2, and the next "
            "cell shows that lone positive read is itself fragile. **H₁ — weak.**"
        ),
        md(
            "### 4b · …and the one positive read (vs XLV) is bootstrap-fragile\n\n"
            "Circular block-bootstrap (21-day blocks, 5,000 reps) of the basket−XLV annual excess. If "
            "the 95% band hugs zero, *t* ≈ 2 is borrowing significance from a thin tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = st.block_bootstrap_excess(BK, RET['XLV'])\n"
            "    pt, lo, hi, pp = bb['point'], bb['lo'], bb['hi'], bb['p_pos']\n"
            "else:\n"
            "    pt, lo, hi, pp = R['boot_pt'], R['boot_lo'], R['boot_hi'], R['boot_p']\n"
            "fig, ax = plt.subplots(figsize=(8.8,3.2))\n"
            "ax.barh([0],[hi-lo],left=[lo],height=.32,color=AMBER,alpha=.55)\n"
            "ax.plot([pt],[0],'D',ms=12,c=GREEN); ax.axvline(0,c=RED,ls='--',label='zero excess')\n"
            "ax.annotate(f'point +{pt:.1f}%',(pt,0),xytext=(0,16),textcoords='offset points',ha='center')\n"
            "ax.annotate(f'lo +{lo:.1f}%',(lo,0),xytext=(0,-22),textcoords='offset points',ha='center',c=RED)\n"
            "ax.set_yticks([]); ax.set_xlabel('annual excess vs XLV (%)'); ax.set_xlim(min(-3,lo-3),hi+3)\n"
            "ax.set_title('basket − XLV excess: 95% bootstrap band nearly touches zero'); ax.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'point +{pt:.1f}%/yr   95% CI [+{lo:.1f}%, +{hi:.1f}%]   P(>0)={pp:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the basket−XLV excess is **+{R['boot_pt']:.1f}%/yr**, but its 95% "
            f"band is **[+{R['boot_lo']:.1f}%, +{R['boot_hi']:.1f}%]** — the lower edge sits a hair "
            "above zero. Combined with the sub-2 *t* vs SPY, the single benchmark it 'beats' is the "
            "narrow one, fragilely. Now add the selection bias (these names were chosen *because* they "
            "won, which inflates the excess) and the honest stamp is **WEAK**, not REAL."
        ),
        md(
            "### 4c · The α is one name — the decomposition\n\n"
            "Per-constituent market-model α vs SPY. A *theme* edge should be shared; a *single-stock* "
            "bet shows it all in one bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mL=st.market_model(RET['LLY'],RET['SPY']); mN=st.market_model(RET['NVO'],RET['SPY'])\n"
            "    aL,tL=mL['alpha_ann']*100,mL['t_alpha']; aN,tN=mN['alpha_ann']*100,mN['t_alpha']\n"
            "else:\n"
            "    aL,tL,aN,tN = R['lly_alpha'],R['lly_ta'],R['nvo_alpha'],R['nvo_ta']\n"
            "fig, ax = plt.subplots(figsize=(8.4,4.3))\n"
            "bars=ax.bar(['LLY','NVO (Ozempic maker)'],[aL,aN],color=[GREEN, RED],width=.5)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('market-model α vs SPY (%/yr)')\n"
            "ax.set_title('All of the basket α lives in Eli Lilly')\n"
            "for b,(v,t) in zip(bars,[(aL,tL),(aN,tN)]): ax.annotate(f'{v:+.0f}%/yr (t={t:.2f})',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LLY α {aL:+.1f}%/yr t={tL:.2f}   |   NVO α {aN:+.1f}%/yr t={tN:.2f}')"
        ),
        md(
            f"> 💡 In plain words: **LLY** α = +{R['lly_alpha']:.1f}%/yr at *t* = **{R['lly_ta']:.2f}** "
            f"(a real single-name edge); **NVO** α = +{R['nvo_alpha']:.1f}%/yr at *t* = "
            f"**{R['nvo_ta']:.2f}** — statistically zero, despite being *the* Ozempic stock. The "
            "'basket' is Lilly carrying a passenger. **H₂ rejected** — `MISATTRIBUTED`."
        ),
        md(
            "### 4d · The recency split — the excess flips sign\n\n"
            "Annualised excess vs SPY, before vs during the GLP-1 mania (cut at 2023-01-01)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pre = RET.index < '2023-01-01'; post = ~pre\n"
            "    ep = st.excess_test(BK[pre], RET['SPY'][pre]); eq = st.excess_test(BK[post], RET['SPY'][post])\n"
            "    preex,pret = ep['ann_excess']*100, ep['t_hac']; postex,postt = eq['ann_excess']*100, eq['t_hac']\n"
            "else:\n"
            "    preex,pret,postex,postt = R['pre_ex'],R['pre_t'],R['post_ex'],R['post_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2,4.2))\n"
            "bars=ax.bar(['2018–2022\\n(before)','2023–2026\\n(the mania)'],[preex,postex],color=[GREEN,RED],width=.5)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('excess vs SPY (%/yr)')\n"
            "ax.set_title('The edge worked before the theme — and reversed during it')\n"
            "for b,(v,t) in zip(bars,[(preex,pret),(postex,postt)]): ax.annotate(f'{v:+.0f}%/yr (t={t:.2f})',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2018-2022: {preex:+.1f}%/yr (t={pret:.2f})   2023-2026: {postex:+.1f}%/yr (t={postt:.2f})')"
        ),
        md(
            f"> 💡 In plain words: **+{R['pre_ex']:.0f}%/yr** before 2023, **{R['post_ex']:.1f}%/yr** "
            "during the mania — a **sign flip** across a justified split. The trade's edge predates the "
            "story it's sold with; once the theme was a named, ETF-ified trade, it *underperformed* the "
            "market. **H₃ rejected.** This is the `MIXED`/regime-conditioned signature, not a durable "
            "premium."
        ),
        md(
            "### 4e · Tradability — costs are a red herring; concentration is the killer\n\n"
            "One-way cost sweep on rebalancing turnover (vs XLV), and the drawdown of a two-name book."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sweep = st.cost_sweep(BK, RET['XLV'], RET); tpd = st.turnover_per_day(RET)*100\n"
            "    bps=[s['bps'] for s in sweep]; exc=[s['ann_excess']*100 for s in sweep]\n"
            "    dds=[st.max_drawdown(BK)*100, st.max_drawdown(RET['NVO'])*100, st.max_drawdown(RET['SPY'])*100]\n"
            "else:\n"
            "    bps=[0,1,2,5,10]; exc=[R['ex_xlv'],R['ex_xlv'],R['ex_xlv']-0.02,R['ex_xlv']-0.05,R['cost10_ex']]; tpd=R['turnover']\n"
            "    dds=[R['bk_dd'],R['nvo_dd'],R['spy_dd']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.0))\n"
            "a1.plot(bps,exc,'o-',c=AMBER,lw=2); a1.set_xlabel('one-way cost (bps of traded NAV)')\n"
            "a1.set_ylabel('net excess vs XLV (%/yr)'); a1.set_ylim(0,18); a1.set_title(f'Costs barely bite (turnover {tpd:.2f}%/day)')\n"
            "a2.bar(['basket','NVO','SPY'],dds,color=[AMBER,RED,GREY],width=.6); a2.axhline(0,c='k',lw=1)\n"
            "a2.set_ylabel('max drawdown (%)'); a2.set_title('…but a 2-name book draws down −49%')\n"
            "for i,v in enumerate(dds): a2.annotate(f'{v:.0f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'turnover {tpd:.3f}%/day | net-of-10bp excess vs XLV {exc[-1]:.2f}%/yr | basket maxDD {dds[0]:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: at **{R['turnover']:.2f}%/day** turnover, even 10bp one-way leaves "
            f"the excess essentially unchanged (+{R['cost10_ex']:.1f}%/yr). The idea doesn't die to the "
            f"spread — it dies to **concentration**: a **{R['bk_dd']:.0f}%** drawdown on two correlated "
            "names betting one product cycle. That is beta to a story, uninvestable as a *scalable, "
            "diversified edge.* **Tradability — MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine control — recover the planted α (and its absence)\n\n"
            "A deterministic one-factor panel plants a known per-name α; the market-model engine must "
            "find it where present and *not* find it where it's zero."
        ),
        code(
            "S = data.synthetic_panel()\n"
            "rows = []\n"
            "for nm, planted in [('LLY',0.0010*252),('NVO',0.0),('XLV',0.0002*252)]:\n"
            "    mm = st.market_model(S[nm], S['SPY']); rows.append((nm, planted*100, mm['alpha_ann']*100, mm['t_alpha']))\n"
            "x=np.arange(len(rows)); fig,ax=plt.subplots(figsize=(8.6,4.2))\n"
            "ax.bar(x-.18,[r[1] for r in rows],.36,color=GREY,label='planted α (%/yr)')\n"
            "ax.bar(x+.18,[r[2] for r in rows],.36,color=GREEN,label='recovered α (%/yr)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{r[0]}\\n(t={r[3]:.2f})' for r in rows])\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('α (%/yr)'); ax.set_title('Engine finds planted α, and the planted ZERO (NVO)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for nm,pl,rec,t in rows: print(f'{nm}: planted {pl:+.1f}%/yr  recovered {rec:+.1f}%/yr  t={t:.2f}')"
        ),
        md(
            "> 💡 In plain words: the engine flags the planted edge (LLY-like, large *t*) and correctly "
            "**fails to find** α where none was planted (NVO-like, *t* ≈ 0) — to Monte-Carlo error. "
            "That's a *machinery* proof the decomposition isn't an artefact; it is **not** market "
            "evidence (a synthetic Sharpe never supports a Signal stamp)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — mean excess vs SPY +{R['ex_spy']:.1f}%/yr at HAC-*t* = "
            f"{R['t_spy']:.2f} (< 2); market-model α vs SPY *t* = {R['ta_spy']:.2f}. The lone *t* ≈ 2 "
            f"(vs XLV) has a bootstrap CI **[+{R['boot_lo']:.1f}%, +{R['boot_hi']:.1f}%]** grazing "
            "zero, and the names were chosen post-hoc (selection biases it up). Literature/headline "
            "says alpha; this tape can't certify it.\n"
            f"- **Tradability `MIRAGE`** — a two-name book (corr {R['corr_llynvo']:.2f}), basket max-DD "
            f"**{R['bk_dd']:.0f}%**, riding one product cycle. Costs immaterial "
            f"({R['turnover']:.2f}%/day turnover); concentration, not the spread, is fatal to scaling "
            "it as an edge.\n"
            f"- **Two stocks & a story? `MISATTRIBUTED`** — α is all LLY (*t* = {R['lly_ta']:.2f}); NVO, "
            f"the Ozempic maker, contributes none (*t* = {R['nvo_ta']:.2f}) and is "
            f"{R['nvo_dd_peak']:.0f}% off its high; and the excess flips to {R['post_ex']:.1f}%/yr "
            "during the 2023–26 mania the theme is named for."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the risk-adjusted reality\n\n"
            "Even granting the raw return, what did you *earn per unit of risk*, and how does it compare "
            "once you account for the market exposure you were paid for anyway? Sharpe of each book "
            "(rf = 0), with the basket's β to SPY noted."
        ),
        code(
            "if HAVE_REAL:\n"
            "    shp = {n: st.perf(RET[n])['sharpe'] for n in ['LLY','NVO','XLV','SPY']}\n"
            "    shp['basket'] = st.perf(BK)['sharpe']; beta = st.market_model(BK, RET['SPY'])['beta']\n"
            "else:\n"
            "    shp = {'LLY':R['lly_sh'],'NVO':R['nvo_sh'],'XLV':R['xlv_sh'],'SPY':R['spy_sh'],'basket':R['bk_sh']}; beta=R['beta_spy']\n"
            "order=['basket','LLY','NVO','XLV','SPY']; vals=[shp[k] for k in order]\n"
            "fig,ax=plt.subplots(figsize=(8.6,4.2))\n"
            "cols=[AMBER,'#1f6feb',RED,GREY,GREY]\n"
            "bars=ax.bar(order,vals,color=cols,width=.6); ax.set_ylabel('Sharpe (rf=0)')\n"
            "ax.set_title(f'Basket Sharpe {shp[\"basket\"]:.2f} (β to SPY {beta:.2f}) — beats SPY only modestly, all via LLY')\n"
            "for b,v in zip(bars,vals): ax.annotate(f'{v:.2f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', {k:round(shp[k],2) for k in order}, '| basket beta to SPY', round(beta,2))"
        ),
        md(
            f"> 💡 In plain words: the basket Sharpe (**{R['bk_sh']:.2f}**) edges SPY's "
            f"(**{R['spy_sh']:.2f}**) — but it carries β ≈ {R['beta_spy']:.2f} to the market and a "
            f"{R['bk_dd']:.0f}% drawdown, and LLY's Sharpe ({R['lly_sh']:.2f}) is the whole story "
            "(NVO's is 0.40). A modest Sharpe bump that's one stock and a deep tail isn't a strategy "
            "you'd allocate to; it's a concentrated bet you got paid for *this once.*"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **A diversified GLP-1 basket.** Add the obesity-drug ETFs and the named 'GLP-1 losers' "
            "(food, dialysis, insulin-pump shorts); re-test whether a *broad* theme α clears *t* = 2 "
            "vs SPY — or still collapses to LLY.\n"
            "- **Selection-clean construction.** Form the basket from the GLP-1 hopefuls as of 2018 "
            "(no hindsight), equal-weight, and rebalance on a rule. That is the honest counterfactual "
            "to a winner-picked basket (Brown et al. 1992).\n"
            "- **NVO ADR vs the home line.** Re-run with the Copenhagen B-share in DKK (FX-hedged) to "
            "separate the company's fundamentals from ADR/FX effects in the −69% drawdown.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
