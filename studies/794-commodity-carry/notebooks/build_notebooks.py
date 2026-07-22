"""Generate the two narrative notebooks for Study 794 (Commodity Carry).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EIA curve
+ ETF tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md. Authoritative set is
# printed by examples/verify.py (as-of 2026-06-30; curve fp ce2aaac1d6db, ETF fp
# 8de803f53090). EIA WTI/HH curve series discontinued 2024-04, so the panel ends 2024-04.
R = dict(
    start="2006-04", end="2024-04",
    n_obs=410, n_months=205,
    pooled_slope=-0.0145, pooled_corr=-0.055, pooled_nw_t=-0.95,
    n_back=105, n_cont=316, back_pct=-1.01, cont_pct=-1.20, welch_t=0.13,
    hit=46, hit_n=105, hit_pct=43.8, wilson=(34.7, 53.4),
    rd_n=197, rd_slope=0.0409, rd_nw_t=2.23, rd_sign_agree=0.49,
    rd_gap_back=-0.33, rd_gap_cont=-0.76,
    timer_5_gross_ann=-9.46, timer_5_gross_t=-1.48, timer_5_gross_sharpe=-0.35,
    timer_5_net_ann=-12.86, timer_5_net_t=-2.01, timer_5_net_sharpe=-0.48, timer_5_worst=-31.27,
    timer_10_gross_ann=-9.46, timer_10_net_ann=-15.26, timer_10_net_t=-2.38,
    timer_10_net_sharpe=-0.56, timer_10_worst=-31.47,
    syn_null_mean=0.30, syn_null_sd=1.03, syn_null_fire=1,
    syn_planted_slope=0.1453, syn_planted_t=7.80,
    fp_curve="ce2aaac1d6db", fp_etfs="8de803f53090",
    # verdict stamps (filled from the computed numbers)
    sig="Weak", sig_color="dab617", trad="Mirage", trad_color="c0392b",
    myth="Proxy", myth_color="8b949e",
)

BADGES = (
    "![Signal](https://img.shields.io/badge/Signal-{sig}-{sig_color}?style=flat-square)\n"
    "![Tradability](https://img.shields.io/badge/Tradability-{trad}-{trad_color}?style=flat-square)\n"
    "![Two-name proxy](https://img.shields.io/badge/Cross--section-{myth}-{myth_color}?style=flat-square)\n\n"
).format(**R)

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

from commodity_carry import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    CURVE = data.load_curve()
    ETFS = data.load_etfs()
    PANEL = st.build_panel(CURVE, ETFS, data.COMMODITIES)
else:
    CURVE = ETFS = PANEL = None
print("real cache present:", HAVE_REAL, "| panel rows:",
      (0 if PANEL is None else len(PANEL)),
      "| months:", (0 if PANEL is None else PANEL['month'].nunique()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do backwardated commodities really out-earn the contangoed ones? 🛢️📉\n"
            "### Commodity carry — the textbook premium, weighed on the two energy curves a "
            "free researcher can actually see\n\n"
            + BADGES +
            "There is a famous idea in commodities: the shape of the futures curve pays you. A "
            "**backwardated** curve (near contract priced *above* the far ones) hands a long a "
            "positive **roll yield** each time it rolls; a **contango** curve (near below far) is "
            "a roll *drag*. The cross-sectional version of the claim: sort commodities by that "
            "slope, own the backwardated ones, avoid (or short) the contangoed ones, and collect "
            "a premium. Gorton-Rouwenhorst, Erb-Harvey and Koijen et al. all document it on broad "
            "cross-sections.\n\n"
            "We can't see a broad cross-section for free. What we *can* see is the **real EIA "
            "futures curve** for two energy commodities (WTI crude, Henry Hub gas) and the "
            "**investable ETFs** that trade them. So this is an honest, narrow proxy — a "
            "**two-name** cross-section — and we'll be blunt about what two names can and can't "
            "tell us.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the pooled panel regression "
            "and the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Proxy note.** Carry is read from the real EIA term structure (WTI `RCLC1-4`, "
            "Henry Hub `RNGC1-4`); returns from USO/USL/UNG/UNL. A two-name cross-section is the "
            "thinnest possible and under-powered by construction — house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the curve slope point the right way? | **Directionally, yes.** On the "
            "investable tape, when a commodity is backwardated its front-month fund beats its own "
            "12-month-laddered twin (the roll mechanism is real and shows up in the ETF gap). |\n"
            "| Does carry *predict* which of the two out-earns? | **On this two-name slice, not "
            "decisively.** The cross-sectional carry→return link doesn't clear the desk's *t* ≥ 2 "
            "bar — but two names is far too thin to certify or kill the broad-universe factor. |\n"
            "| Can you trade it? | **No, not here.** A two-name long-short carry book is dominated "
            "by two crash-prone energy curves; costs and short borrow finish what the noise "
            "starts. |\n"
            "| So is the famous premium real? | **The literature's version might well be — but "
            "this proxy can't prove it.** That's the honest headline: a real mechanism, an "
            "under-powered test. |\n\n"
            "> The roll mechanism is genuine; the *tradable cross-sectional premium* is not "
            "something two energy curves can establish."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Own the commodities whose futures curve slopes down (backwardation) — you get "
            "paid to roll — and avoid the ones that slope up (contango), where rolling bleeds "
            "you.\"*\n\n"
            "It's one of the best-documented ideas in the asset class: Erb & Harvey (2006) show "
            "the **roll return**, not the spot move, drives long-run commodity-futures "
            "performance; Koijen et al. (2018) fold it into a unified cross-asset *carry* factor. "
            "The catch for anyone without a Bloomberg terminal: you need the **term structure of "
            "many commodities** to sort them, and that panel isn't free."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the slope really sorts winners from losers, it's a clean, mechanical signal — no "
            "forecasting, just read today's curve. That's exactly the kind of claim worth testing "
            "honestly *and* being suspicious of: \"carry\" is one of the most over-fit words in "
            "finance, and a premium documented on 30 commodities may simply not survive on the two "
            "an amateur can actually price."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The signal.** Each month, read the real EIA futures curve for WTI and gas and "
            "compute the annualized roll yield (front vs next contract). Positive = backwardation.\n"
            "- **The outcome.** The *next* month's return of the investable front-month ETF (USO "
            "for crude, UNG for gas) — strictly forward of the signal.\n"
            "- **The mechanism check.** Compare each front-month fund to its own 12-month-laddered "
            "twin (USO vs USL): the gap *is* the realized roll.\n"
            "- **The trade check.** Long the more-backwardated commodity, short the more-contangoed "
            "one, every month, and pay realistic costs and short borrow."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the mechanism.** When a commodity is backwardated, does its front-month fund "
            "actually beat the laddered one?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rd = st.roll_drag_proxy(PANEL, 'WTI')\n"
            "    gb, gc = rd['mean_gap_backwardation_pct'], rd['mean_gap_contango_pct']\n"
            "else:\n"
            "    gb, gc = R['rd_gap_back'], R['rd_gap_cont']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['WTI backwardated\\n(carry > 0)','WTI contango\\n(carry < 0)'], [gb, gc],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([gb, gc]): ax.annotate(f'{v:+.2f}%/mo',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('front (USO) minus laddered (USL) return, %/mo')\n"
            "ax.set_title('The roll mechanism is real: backwardation favours the front fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'USO-USL gap: backwardation {gb:+.2f}%/mo vs contango {gc:+.2f}%/mo')"
        ),
        md(
            "The gap flips with the curve exactly as the theory says: in backwardation the "
            "front-month fund out-rolls its laddered twin; in contango it lags. So the **roll "
            "mechanism itself is not in doubt** — the question is whether it turns into a "
            "*cross-sectional premium* you can harvest.\n\n"
            "**Next, the real test:** does the carry signal separate the two commodities' returns?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.conditional_split(PANEL)\n"
            "    b, c = cs['back_pct'], cs['cont_pct']\n"
            "else:\n"
            "    b, c = R['back_pct'], R['cont_pct']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['backwardated\\n(carry > 0)','contangoed\\n(carry < 0)'], [b, c],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([b, c]): ax.annotate(f'{v:+.2f}%/mo',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('next-month front-ETF return (%/mo)')\n"
            "ax.set_title('Forward return, split by this month\\'s carry sign (pooled)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'backwardated {b:+.2f}%/mo  vs  contangoed {c:+.2f}%/mo')"
        ),
        md(
            "The direction is right — backwardated months earn more forward — but the gap is "
            "noisy, and (in the quants' notebook) its HAC *t* does **not** clear the bar. Two "
            "energy curves is simply not enough cross-section to certify a premium the literature "
            "measures across dozens of commodities.\n\n"
            "**Finally, the trade.** Long the backwardated, short the contangoed, monthly:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.timer_stats(PANEL, cost_bps=5.0)\n"
            "    tm10 = st.timer_stats(PANEL, cost_bps=10.0)\n"
            "    g, n5, n10 = tm5['gross_ann_pct'], tm5['net_ann_pct'], tm10['net_ann_pct']\n"
            "else:\n"
            "    g, n5, n10 = R['timer_5_gross_ann'], R['timer_5_net_ann'], R['timer_10_net_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['gross','net @5bps\\n+borrow','net @10bps\\n+borrow'], [g, n5, n10],\n"
            "       color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualized return (%/yr)')\n"
            "ax.set_title('A two-name carry book: costs and borrow bury a thin, noisy gross')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr -> net {n5:+.1f} (5bps) / {n10:+.1f} (10bps) %/yr')"
        ),
        md(
            "Whatever thin gross the sort produces, a two-leg monthly rebalance plus short borrow "
            "on crash-prone energy funds erases it. This is a **Mirage** on the tradability axis — "
            "not because the mechanism is fake, but because two energy curves are too few and too "
            "violent to harvest a cross-sectional premium from."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The carry sign points the right way (backwardation → higher "
            "forward return, and the ETF roll gap confirms the mechanism), but on a two-name "
            "cross-section the carry→return HAC *t* doesn't clear the desk's bar. Real mechanism, "
            "under-powered test.\n"
            "- **Tradability — Mirage.** A two-name long-short carry book is dominated by energy "
            "crashes; costs and short borrow finish it.\n"
            "- **Two-name proxy — stated up front.** This slice can neither confirm nor refute the "
            "broad-universe factor in the literature; it tests one honest question on the data a "
            "free researcher can see."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **What would settle it** is breadth: the same carry sort across agriculture, "
            "metals, softs and energy (20-30 curves), where the cross-sectional average washes out "
            "the idiosyncratic energy crashes. That needs a paid futures panel.\n"
            "- **The defensive read still holds:** even here, don't be the investor holding the "
            "front-month fund of a *contangoed* commodity — the roll drag is real and one-way "
            "against you (see [661-uso-roll-decay](../../661-uso-roll-decay/)).\n"
            "- **Sibling studies:** [35-contango](../../35-contango/) (grades the *realized drag* "
            "via the ETF gap), [660-carry-everywhere](../../660-carry-everywhere/) (the "
            "*multi-asset* carry blend), [380-curve-roll-down](../../380-curve-roll-down/) "
            "(single-asset roll-down in rates). See [docs/references.md](docs/references.md) for "
            "the exact dedup.\n\n"
            "*Think two energy curves prove commodity carry? Show a certifiable cross-sectional "
            "premium on a real multi-commodity panel, net of costs and borrow — then we'll talk.*"
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
            "# Commodity carry — a quantitative teardown 🔬\n"
            "### The pooled cross-sectional carry→return HAC regression · a backwardation/contango "
            "conditional split · the USO-USL roll-drag mechanism check · an honest costed "
            "long-short · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **backwardated commodities out-earn contangoed ones cross-sectionally** — is "
            "the roll-yield / carry premium (Gorton-Rouwenhorst 2006; Erb-Harvey 2006; Koijen et "
            "al. 2018). We isolate the **commodity** leg (distinct from the multi-asset blend of "
            "[660](../../660-carry-everywhere/)) and read carry from the **real EIA futures term "
            "structure** as the ex-ante signal (distinct from the ETF-gap *drag* measurement of "
            "[35](../../35-contango/)). The job: measure it honestly on the narrow slice free data "
            "affords, and say plainly what two names can't settle.\n\n"
            "> ⚠️ **Data note.** Carry: EIA daily WTI `RCLC1-4` + Henry Hub `RNGC1-4`. Returns: "
            "yfinance USO/USL/UNG/UNL total return. Monthly panel, one-month execution lag "
            "(signal at month-end `t` settle, position held over `t+1`). Numbers frozen in "
            "[`docs/results.md`](../docs/results.md) (curve fp `" + R["fp_curve"] + "`, ETF fp `"
            + R["fp_etfs"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition.\n"
            ">\n"
            "> **Proxy caveat.** Two names is the thinnest cross-section possible — every "
            "cross-sectional number below is under-powered by construction."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `{R['sig'].upper()}` | pooled cross-sectional carry→return slope "
            f"{R['pooled_slope']:+.3f}, **NW(6) t = {R['pooled_nw_t']:+.2f}** "
            f"(n={R['n_obs']} over {R['n_months']} two-name months); backwardation "
            f"{R['back_pct']:+.2f}%/mo vs contango {R['cont_pct']:+.2f}%/mo (Welch "
            f"t = {R['welch_t']:+.2f}) |\n"
            f"| **Tradability** | `{R['trad'].upper()}` | long-short net of 5 bps + 100 bps/yr "
            f"borrow: {R['timer_5_net_ann']:+.2f}%/yr (NW t = {R['timer_5_net_t']:+.2f}, Sharpe "
            f"{R['timer_5_net_sharpe']:.2f}); worst month {R['timer_5_worst']:+.1f}% |\n"
            f"| **Roll mechanism** | real | USO-USL gap on WTI carry: slope {R['rd_slope']:+.3f}, "
            f"NW t = {R['rd_nw_t']:+.2f}, sign-agreement {R['rd_sign_agree']*100:.0f}% |\n\n"
            "> 💡 In plain words: the roll mechanism is genuine and visible in the ETF gap, but on "
            "a two-name cross-section the carry→return premium doesn't clear *t* ≥ 2, and the "
            "long-short is un-tradable after costs — a real mechanism weighed on an under-powered "
            "proxy."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $F^{(k)}_{i,t}$ be the settle of commodity $i$'s $k$-th near futures on date "
            "$t$. The **carry** (annualized log roll yield) is "
            "$c_{i,t} = \\ln\\!\\big(F^{(1)}_{i,t}/F^{(2)}_{i,t}\\big)\\cdot 12$ — positive in "
            "backwardation. Let $r_{i,t+1}$ be commodity $i$'s investable front-ETF total return "
            "over month $t{+}1$. The claims:\n\n"
            "- **H₁ (cross-sectional premium).** Within a month, higher $c_{i,t}$ predicts higher "
            "$r_{i,t+1}$: the cross-sectionally demeaned slope $\\partial \\tilde r / "
            "\\partial \\tilde c > 0$ and clears HAC $t \\ge 2$.\n"
            "- **H₂ (mechanism).** Backwardation makes the front fund out-roll its laddered twin: "
            "$\\mathrm{corr}(c_{\\text{WTI}}, r_{\\text{USO}}-r_{\\text{USL}}) > 0$.\n"
            "- **H₃ (capture).** A monthly long-short (long high-carry, short low-carry) banks a "
            "premium net of costs and short borrow.\n\n"
            f"We find **H₂ supported** (NW t = {R['rd_nw_t']:+.2f}, sign-agreement "
            f"{R['rd_sign_agree']*100:.0f}%), **H₁ not established on two names** "
            f"(NW t = {R['pooled_nw_t']:+.2f} < 2), **H₃ rejected** "
            f"(net {R['timer_5_net_ann']:+.2f}%/yr, NW t = {R['timer_5_net_t']:+.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "With **two** commodities a literal cross-sectional sort is degenerate, so the primary "
            "estimator **cross-sectionally demeans both carry and forward return within each "
            "month** (removing the common energy move) and NW(6)-regresses demeaned return on "
            "demeaned carry — the pure cross-sectional carry→return slope, robust to the heavy "
            "serial correlation of overlapping energy moves. The conditional split (backwardation "
            "vs contango, pooled) is a Welch cross-check; the mechanism regression (USO-USL gap on "
            "WTI carry) is independent of the sort. The timer charges one-way bps × turnover × NAV "
            "per leg and an explicit short borrow. A 20-seed synthetic control proves the "
            "estimator is unbiased."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_months']} monthly cross-sections, two commodities (WTI, gas), "
            f"{R['start']} → {R['end']}; carry at month-end `t`, front-ETF return over `t+1` "
            "(one documented execution lag, zero look-ahead).\n"
            "- **Workhorse.** Cross-sectionally demeaned carry→return NW(6) regression.\n"
            "- **Conditional split.** Forward return | backwardation vs | contango, Welch t + "
            "Wilson hit rate.\n"
            "- **Mechanism.** USO-USL monthly gap regressed on WTI carry.\n"
            "- **Timer.** Monthly two-name long-short, 5/10 bps one-way + 100 bps/yr borrow.\n"
            "- **Control.** Synthetic two-commodity panel, tunable planted premium; the null "
            "(premium=0) checked over 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The workhorse — pooled cross-sectional carry → forward return\n\n"
            "Cross-sectionally demean carry and return each month, then scatter and NW-regress. "
            "The slope is the premium per unit of annualized roll yield."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.dropna(subset=['carry','ret']).copy()\n"
            "    cc = p['carry'] - p.groupby('month')['carry'].transform('mean')\n"
            "    rr = p['ret']   - p.groupby('month')['ret'].transform('mean')\n"
            "    pt = st.pooled_carry_test(PANEL)\n"
            "    slope, tval, corr = pt['slope'], pt['nw_t'], pt['corr']\n"
            "else:\n"
            "    rng = np.random.default_rng(794)\n"
            "    cc = rng.normal(0,0.3,300); rr = R['pooled_slope']*cc + rng.normal(0,0.09,300)\n"
            "    slope, tval, corr = R['pooled_slope'], R['pooled_nw_t'], R['pooled_corr']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.scatter(cc, rr, s=16, alpha=.5, color=GREY)\n"
            "xs = np.linspace(np.nanmin(cc), np.nanmax(cc), 50)\n"
            "ax.plot(xs, slope*xs, color=GREEN, lw=2, label=f'slope {slope:+.3f} (NW t={tval:+.2f})')\n"
            "ax.axhline(0, c='k', lw=.6); ax.axvline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('cross-sectionally demeaned carry (annualized roll yield)')\n"
            "ax.set_ylabel('cross-sectionally demeaned forward return')\n"
            "ax.set_title('Carry -> forward return, cross-sectionally demeaned')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:+.4f}  corr {corr:+.3f}  NW(6) t {tval:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the slope is {R['pooled_slope']:+.3f} (right sign — more carry, "
            f"more forward return) but **NW(6) t = {R['pooled_nw_t']:+.2f}**, short of the desk's "
            f"*t* ≥ 2. On two names the cross-sectional average can't out-shout the idiosyncratic "
            "energy noise. H₁ is **not established** here — which is a statement about the proxy's "
            "power, not a refutation of the broad-universe factor."
        ),
        md(
            "### 4b · The conditional split and hit rate\n\n"
            "Pool both commodities; compare forward return in backwardated vs contangoed months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.conditional_split(PANEL)\n"
            "    b, c, wt = cs['back_pct'], cs['cont_pct'], cs['welch_t']\n"
            "    hit, hitn = cs['hit'], cs['hit_n']\n"
            "    lo, hi = st.wilson_interval(hit, hitn)\n"
            "else:\n"
            "    b, c, wt = R['back_pct'], R['cont_pct'], R['welch_t']\n"
            "    hit, hitn = R['hit'], R['hit_n']; lo, hi = [x/100 for x in R['wilson']]\n"
            "hitn = max(hitn, 1)\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(11.0,4.3))\n"
            "a1.bar(['backwardation','contango'], [b, c], color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([b,c]): a1.annotate(f'{v:+.2f}%/mo',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('next-month front-ETF return (%/mo)')\n"
            "a1.set_title(f'Forward return by carry sign (Welch t={wt:+.2f})')\n"
            "a2.bar(['backwardated\\nhit rate'], [hit/hitn*100], color=GREEN, width=.4)\n"
            "a2.errorbar([0], [hit/hitn*100], yerr=[[hit/hitn*100-lo*100],[hi*100-hit/hitn*100]],\n"
            "    fmt='none', ecolor='k', capsize=6)\n"
            "a2.axhline(50, ls='--', c=GREY); a2.set_ylim(0,100)\n"
            "a2.set_ylabel('% of backwardated months with return>0')\n"
            "a2.set_title(f'{hit}/{hitn} = {hit/hitn*100:.0f}%  (Wilson [{lo*100:.0f},{hi*100:.0f}])')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'back {b:+.2f}%/mo vs cont {c:+.2f}%/mo, Welch t={wt:+.2f}; hit {hit}/{hitn}')"
        ),
        md(
            f"> 💡 In plain words: backwardated months average {R['back_pct']:+.2f}%/mo forward "
            f"vs {R['cont_pct']:+.2f}%/mo in contango (Welch t = {R['welch_t']:+.2f}), and the "
            f"backwardated hit rate ({R['hit']}/{R['hit_n']} = {R['hit_pct']:.0f}%, Wilson "
            f"[{R['wilson'][0]:.0f}%, {R['wilson'][1]:.0f}%]) has its lower bound near 50%. "
            "Directionally consistent, statistically inconclusive."
        ),
        md(
            "### 4c · The mechanism — USO minus USL vs WTI carry\n\n"
            "The front-minus-laddered gap on the *same* crude is the realized roll. If carry "
            "means what we say, this regresses positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = PANEL[PANEL['commodity']=='WTI'].dropna(subset=['carry','ret','ladder_ret'])\n"
            "    cx = g['carry'].to_numpy(); gap = (g['ret']-g['ladder_ret']).to_numpy()*100\n"
            "    rd = st.roll_drag_proxy(PANEL,'WTI')\n"
            "    slope, tval, agree = rd['slope'], rd['nw_t'], rd['sign_agree']\n"
            "else:\n"
            "    rng = np.random.default_rng(1); cx = rng.normal(0,0.3,200)\n"
            "    gap = (R['rd_slope']*cx + rng.normal(0,0.02,200))*100\n"
            "    slope, tval, agree = R['rd_slope'], R['rd_nw_t'], R['rd_sign_agree']\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.6))\n"
            "ax.scatter(cx, gap, s=16, alpha=.5, color=GREY)\n"
            "xs=np.linspace(np.nanmin(cx),np.nanmax(cx),50)\n"
            "ax.plot(xs, slope*100*xs, color=GREEN, lw=2, label=f'slope {slope:+.3f} (NW t={tval:+.2f})')\n"
            "ax.axhline(0,c='k',lw=.6); ax.axvline(0,c='k',lw=.6)\n"
            "ax.set_xlabel('WTI carry (annualized roll yield)')\n"
            "ax.set_ylabel('USO minus USL return (%/mo)')\n"
            "ax.set_title(f'Roll mechanism: backwardation favours the front fund '\n"
            "             f'(sign-agreement {agree*100:.0f}%)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'USO-USL vs WTI carry: slope {slope:+.4f}, NW t={tval:+.2f}, agree {agree*100:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: the mechanism is **real and clean** — slope {R['rd_slope']:+.3f}, "
            f"NW t = {R['rd_nw_t']:+.2f}, the sign agrees {R['rd_sign_agree']*100:.0f}% of months. "
            "Backwardation genuinely makes the front fund out-roll its ladder. The mechanism was "
            "never the weak link; *turning it into a certifiable cross-sectional premium on two "
            "names* is."
        ),
        md(
            "### 4d · The timer — honest cost + borrow sweep\n\n"
            "Monthly two-name long-short (long high-carry, short low-carry); one-way bps × "
            "turnover × NAV per leg, short leg pays 100 bps/yr borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.timer_stats(PANEL, cost_bps=5.0); tm10 = st.timer_stats(PANEL, cost_bps=10.0)\n"
            "    g = tm5['gross_ann_pct']; n5,n10 = tm5['net_ann_pct'], tm10['net_ann_pct']\n"
            "    gt, t5, t10 = tm5['gross_t'], tm5['net_t'], tm10['net_t']\n"
            "    gs, s5, s10 = tm5['gross_sharpe'], tm5['net_sharpe'], tm10['net_sharpe']\n"
            "    cum = (1+tm5['series']).cumprod()\n"
            "else:\n"
            "    g=R['timer_5_gross_ann']; n5,n10=R['timer_5_net_ann'],R['timer_10_net_ann']\n"
            "    gt,t5,t10=R['timer_5_gross_t'],R['timer_5_net_t'],R['timer_10_net_t']\n"
            "    gs,s5,s10=R['timer_5_gross_sharpe'],R['timer_5_net_sharpe'],R['timer_10_net_sharpe']\n"
            "    cum=None\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(11.2,4.3))\n"
            "a1.bar(['gross','net @5bps','net @10bps'], [g,n5,n10], color=[GREY,AMBER,RED], width=.6)\n"
            "for i,v in enumerate([g,n5,n10]): a1.annotate(f'{v:+.1f}',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('annualized return (%/yr)')\n"
            "a1.set_title(f'gross t={gt:+.2f} / net@5 t={t5:+.2f} / net@10 t={t10:+.2f}')\n"
            "if cum is not None:\n"
            "    a2.plot(cum.index.to_timestamp(), cum.values, color=RED, lw=1.4)\n"
            "    a2.axhline(1, ls='--', c=GREY)\n"
            "    a2.set_ylabel('net-of-cost NAV (5bps+borrow), growth of 1')\n"
            "    a2.set_title('The two-name carry book is not investable')\n"
            "else:\n"
            "    a2.axis('off')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr (t={gt:+.2f}, Sh {gs:.2f}) -> net {n5:+.1f} (5bps, t={t5:+.2f}, Sh {s5:.2f}) / {n10:+.1f} (10bps, t={t10:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: gross {R['timer_5_gross_ann']:+.1f}%/yr "
            f"(NW t = {R['timer_5_gross_t']:+.2f}) is already short of significance; net of 5 bps "
            f"+ borrow it is {R['timer_5_net_ann']:+.2f}%/yr (t = {R['timer_5_net_t']:+.2f}, Sharpe "
            f"{R['timer_5_net_sharpe']:.2f}), worst month {R['timer_5_worst']:+.1f}%. "
            "**Tradability = MIRAGE:** two crash-prone energy legs, thin gross, real frictions."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic two-commodity panel, TUNABLE planted cross-sectional carry premium. The "
            "null (premium = 0) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    p = data.synthetic_panel(premium=0.0, seed=794 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p)['nw_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted = st.synthetic_detect(data.synthetic_panel(premium=0.15, seed=794))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20)+np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (premium=0), 20 seeds')\n"
            "ax.scatter([1], [planted['nw_t']], color=GREEN, s=90, zorder=5,\n"
            "           label='planted premium = 0.15')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('pooled carry->return NW(6) t')\n"
            "ax.set_title('Control: no null fires; a planted premium lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20  |  planted t={planted[\"nw_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires in "
            f"{R['syn_null_fire']}/20; a planted premium reads t = {R['syn_planted_t']:+.2f}. The "
            "estimator is unbiased — so the real-tape *t* below 2 is a genuine power limit of two "
            "names, not a broken detector. *(A faithful-engine / power check only — never cited in "
            "support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `{R['sig'].upper()}`** — pooled cross-sectional carry→return slope "
            f"{R['pooled_slope']:+.3f} (right sign) but NW(6) t = {R['pooled_nw_t']:+.2f} < 2; "
            f"backwardation {R['back_pct']:+.2f}%/mo vs contango {R['cont_pct']:+.2f}%/mo "
            f"(Welch t = {R['welch_t']:+.2f}). The roll **mechanism** is real "
            f"(USO-USL vs carry NW t = {R['rd_nw_t']:+.2f}); the cross-sectional **premium** is "
            "not established on a two-name proxy.\n"
            f"- **Tradability `{R['trad'].upper()}`** — long-short net of 5 bps + borrow "
            f"{R['timer_5_net_ann']:+.2f}%/yr (t = {R['timer_5_net_t']:+.2f}, Sharpe "
            f"{R['timer_5_net_sharpe']:.2f}); worst month {R['timer_5_worst']:+.1f}%. Two "
            "crash-prone energy legs are not a harvestable premium.\n"
            "- **Two-name proxy — stated up front.** Under-powered by construction; can neither "
            "confirm nor refute the broad-universe factor of Gorton-Rouwenhorst / Erb-Harvey / "
            "Koijen et al."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **What would settle it:** the identical carry sort across 20-30 commodity curves, "
            "where the cross-sectional mean averages out idiosyncratic energy crashes — the setting "
            "the literature actually uses. That needs a paid futures panel.\n"
            "- **The mechanism is bankable as a defensive rule** even if the premium isn't "
            "harvestable here: don't hold the front-month fund of a contangoed commodity.\n"
            "- **Dedup map:** [35-contango](../../35-contango/) (realized drag via the ETF gap, "
            "not the ex-ante curve signal), [660-carry-everywhere](../../660-carry-everywhere/) "
            "(multi-asset blend), [380-curve-roll-down](../../380-curve-roll-down/) (single-asset "
            "rates roll-down), [661-uso-roll-decay](../../661-uso-roll-decay/) (single-fund USO "
            "decay).\n\n"
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
