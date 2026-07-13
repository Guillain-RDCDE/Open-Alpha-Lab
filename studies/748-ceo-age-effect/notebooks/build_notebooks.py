"""Generate the two narrative notebooks for Study 748 (CEO-Age-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached prices under ../_cache/
if present and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so
the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; scored
# 2024-12-31 age split, monthly total-return tape 2018-02 -> 2026-06; 40 curated CEOs;
# price fp 4c394346b808, L/S panel fp bcef5ab08c9a).
R = dict(
    fp_px="4c394346b808", fp_panel="bcef5ab08c9a",
    n_months=101, start="2018-02", end="2026-06",
    n_young=14, n_old=26, n_total=40, score="2024-12-31", young_max=55,
    young_ann=29.24, young_vol=32.44, young_sh=0.90,
    old_ann=21.27, old_vol=19.76, old_sh=1.08,
    mkt_ann=14.46, mkt_sh=0.87,
    ls_ann=7.97, ls_vol=20.88, ls_sh=0.38, ls_mean_m=0.664, ls_t=0.92, hac_lags=4,
    alpha_m=0.248, alpha_ann=2.98, alpha_t=0.36, beta=0.35, beta_t=2.7, r2=0.07,
    net_ann=7.19, gross_sh=0.38, net_sh=0.34, cost_bps=5.0, borrow_bps=75.0, turnover=0.30,
    placebo_p=0.523, placebo_obs_t=0.92,
    # cutoff sweep: (label, ls_ann%, ls_t, alpha_t, beta)
    cutoff=[("age<50", 1.1, 0.105, -0.657, 0.496),
            ("age<55", 8.0, 0.924, 0.361, 0.345),
            ("age<60", 11.0, 1.901, 1.461, 0.238)],
    # subperiod sweep: (label, ls_ann%, ls_t, alpha_t, beta)
    sub=[("2018-2020", 26.6, 1.961, 2.029, 0.124),
         ("2021-2022", -34.5, -2.222, -2.353, 0.176),
         ("2023-2026", 16.8, 1.643, -0.035, 0.806)],
    # synthetic control: (age_alpha, mean CAPM alpha t over 25 seeds, mean raw HAC t)
    ctrl=[(0.000, 0.01, -0.00), (0.004, 2.12, 1.51), (0.008, 4.23, 3.03), (0.012, 6.35, 4.54)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#3b6fb0"

from ceo_age_effect import data, strategy as st

def load_real():
    \"\"\"Cache-first real monthly L/S frame (empty offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    if px.empty:
        return pd.DataFrame(), pd.DataFrame()
    return px, data.build_returns(px)

PX, RET = load_real()
HAVE_REAL = not RET.empty
print("real CEO-age L/S panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(RET)} months, {RET.index.min().date()} -> {RET.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Young Boss, Hot Stock? — do firms with young CEOs beat firms with old ones? 👴\n"
            "### Sorting big companies by the age of the person in the corner office, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Young--CEO_edge%3F: Misattributed](https://img.shields.io/badge/Young--CEO_edge%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "There's a tidy business-school story, and it even has real academic papers behind it: "
            "**young** CEOs are aggressive empire-builders who take big swings and run their companies "
            "hot; **old** CEOs are cautious cash-harvesters who dial the risk down. If that flows "
            "through to the share price, you could just sort the market by the boss's *birthday* and "
            "buy the young ones.\n\n"
            "So we build that exact trade — long the young-CEO firms, short the old-CEO firms — and "
            "watch what actually happens.\n\n"
            "> 📓 **This is the plain-language layer.** Want the HAC *t*-stats, the CAPM alpha and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do young-CEO firms earn *more*? | **Yes, on paper** — the long/short earns "
            f"**+{R['ls_ann']:.0f}%/yr** — but it's a coin-flip statistically (*t* = {R['ls_t']:.1f}). |\n"
            f"| Is it the CEO's *age*? | **No.** The young basket is stuffed with young-founder "
            "**growth-tech** (NVDA-adjacent, COIN, HOOD, SHOP…). The gap is a tech/size bet, not a "
            "birthday. |\n"
            f"| Do young-CEO firms even win *risk-adjusted*? | **No** — the young basket's Sharpe is "
            f"**{R['young_sh']:.2f}** vs the old basket's **{R['old_sh']:.2f}**. More return, *much* "
            "more stomach-churn — the old bosses actually win per unit of risk. |\n"
            "| Could you trade it? | **No.** Strip out the market and the 'edge' is ~zero; it flips "
            "sign whenever growth stocks fall out of favour. |\n\n"
            "> The 'aggressive young CEO' story is *half* true — their stocks really are wilder. But "
            "wild isn't the same as good, and 'young' here is just a disguise for 'growth tech.'"
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Young CEOs take more risk and chase growth; old CEOs play it safe. So young-CEO "
            "firms should behave — and pay — differently.\"*\n\n"
            "This one isn't pure folklore — it has a serious research pedigree. **Serfling (2014)** "
            "found older CEOs run *less* risky corporate policies (lower R&D, less leverage, smoother "
            "earnings); **Yim (2013)** found *younger* CEOs do far more acquisitions ('the "
            "acquisitiveness of youth'). The leap the *trade* makes is the shaky part: that riskier "
            "corporate behaviour turns into a **predictable stock-return edge** you can harvest.\n\n"
            "We hand-build a transparent table of ~40 big-name CEOs, look up each one's **birth "
            "year** (public record), split them into **young** (under 55) and **old**, and test the "
            "long-young / short-old book."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If a CEO's age really predicted stock returns, you'd have a screen that needs no "
            "accounting and no charts — just a Wikipedia birthday. It would also be a tidy proof that "
            "'management style' shows up cleanly in prices. The catch we'll keep hammering: the young-"
            "CEO bucket is *also* the recent-IPO, high-growth, high-beta bucket. Any 'age effect' has "
            "to fight its way past the fact that **young founders run tech companies** — and tech is "
            "what actually moved."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the table.** For each big-name CEO, look up the birth year and compute their "
            "age at a fixed scoring date.\n"
            "2. **Split young vs old.** Equal-weight the young-CEO stocks into one basket, the old-CEO "
            "stocks into another.\n"
            "3. **Long young, short old.** Track that market-neutral spread month by month for eight "
            "years.\n"
            "4. **Ask two questions.** Is the spread reliably positive (not just lucky)? And — the "
            "decisive one — **does anything survive once you subtract the market?**\n\n"
            "*The mirage tell to watch for:* an edge that only exists because one basket is riskier "
            "than the other. That's not skill from youth; that's just more beta.\n\n"
            "*Timing:* a CEO's age is public years in advance, so there's nothing to peek at — the "
            "book is built from information everyone already has."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Do young-CEO firms earn more — and is it worth the ride?** Here are the two baskets "
            "and the market, by annual return *and* by Sharpe ratio (return per unit of risk)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bs = st.bucket_stats(RET)\n"
            "    lab = ['young\\nCEOs','old\\nCEOs','market\\n(SPY)']\n"
            "    rets = [bs['young']['ann_ret']*100, bs['old']['ann_ret']*100, bs['mkt']['ann_ret']*100]\n"
            "    shs  = [bs['young']['sharpe'], bs['old']['sharpe'], bs['mkt']['sharpe']]\n"
            "else:\n"
            f"    lab = ['young\\nCEOs','old\\nCEOs','market\\n(SPY)']\n"
            f"    rets = [{R['young_ann']}, {R['old_ann']}, {R['mkt_ann']}]\n"
            f"    shs  = [{R['young_sh']}, {R['old_sh']}, {R['mkt_sh']}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.bar(lab, rets, color=[GREEN, GREY, BLUE], width=.6)\n"
            "a1.set_ylabel('annualised return %'); a1.set_title('Young-CEO firms earned more...')\n"
            "a2.bar(lab, shs, color=[GREEN, GREY, BLUE], width=.6)\n"
            "a2.set_ylabel('Sharpe (return per unit risk)'); a2.set_title('...but LOST on risk-adjusted return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'young {rets[0]:+.1f}%/yr (Sharpe {shs[0]:.2f}) | old {rets[1]:+.1f}%/yr (Sharpe {shs[1]:.2f})')"
        ),
        md(
            f"There's the first crack. Young-CEO firms did earn more (+{R['young_ann']:.0f}% vs "
            f"+{R['old_ann']:.0f}% a year) — but look at the right panel: their **Sharpe ratio is "
            f"lower** ({R['young_sh']:.2f} vs {R['old_sh']:.2f}). They pay more return because they "
            f"take *much* more risk ({R['young_vol']:.0f}% volatility vs {R['old_vol']:.0f}%). Per "
            "unit of white-knuckle, the **old** bosses actually delivered more. 'Aggressive youth' is "
            "real — but aggression isn't free money."
        ),
        md(
            "**Fine — but the long/short spread is still positive. Is *that* real?** Let's follow the "
            "long-young / short-old book through time and split it by era."
        ),
        code(
            "sub = " + repr(R["sub"]) + "\n"
            "if HAVE_REAL:\n"
            "    splits = [('2018-2020','2018-01-01','2020-12-31'),\n"
            "              ('2021-2022','2021-01-01','2022-12-31'),\n"
            "              ('2023-2026','2023-01-01','2026-06-30')]\n"
            "    sp = st.subperiod_sweep(RET, splits)\n"
            "    labs = list(sp.index); vals = (sp['ls_ann']*100).tolist()\n"
            "else:\n"
            "    labs = [s[0] for s in sub]; vals = [s[1] for s in sub]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(labs, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long-young / short-old, %/yr')\n"
            "ax.set_title('The \"young premium\" flips sign by era — that is a growth bet, not an age effect')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('by era (%/yr):', dict(zip(labs, [round(v,1) for v in vals])))"
        ),
        md(
            "That's the tell. In 2018-2020 young-CEO growth ran hot (**+27%/yr**); in 2021-2022, when "
            "rates spiked and growth stocks were taken to the woodshed, the same book lost **−34%/yr**; "
            "in the 2023-2026 AI melt-up it bounced back. A real *age* effect wouldn't care what "
            "interest rates were doing. This one is a **growth-vs-value bet** wearing a birthday hat."
        ),
        md(
            "**Last check: does the answer depend on where we draw the 'young' line?** It shouldn't, "
            "if it's real. Watch the *t*-stat (how many standard errors from zero — you want it past "
            "~2) as we move the cutoff from under-50 to under-60."
        ),
        code(
            "cut = " + repr(R["cutoff"]) + "\n"
            "if HAVE_REAL:\n"
            "    cs = st.cutoff_sweep(data, PX, [50, 55, 60])\n"
            "    labs = list(cs.index); ts = cs['ls_t'].tolist()\n"
            "else:\n"
            "    labs = [c[0] for c in cut]; ts = [c[2] for c in cut]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar(labs, ts, color=GREY, width=.5)\n"
            "ax.axhline(2, c=RED, ls='--', lw=1.2, label='t = 2 (significance bar)')\n"
            "ax.axhline(-2, c=RED, ls='--', lw=1.2)\n"
            "ax.set_ylabel('long/short t-stat'); ax.legend()\n"
            "ax.set_title('Move the age line, the \"signal\" wanders — and never clears the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t by cutoff:', dict(zip(labs, [round(t,2) for t in ts])))"
        ),
        md(
            "The *t*-stat crawls from ~0.1 to ~1.9 as we slide the cutoff — never once clearing the "
            "2.0 line, and its value depends entirely on an **arbitrary choice** we made. A real "
            "effect doesn't evaporate when you nudge the definition by five years."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The long/short is +{R['ls_ann']:.0f}%/yr but *t* = {R['ls_t']:.1f} "
            f"(a coin flip); a random re-labelling of the CEOs beats it {R['placebo_p']*100:.0f}% of "
            "the time.\n"
            "- **Tradability — Mirage.** Subtract the market and the edge is ~zero — it's growth-tech "
            "beta you could rent from an ETF for less.\n"
            "- **Young-CEO edge? — Misattributed.** The gap is sector/size/vintage (young founders run "
            "tech), not the boss's age. Risk-adjusted, the old CEOs win."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            f"You *could* run long-young / short-old — turnover is tiny (a CEO's age barely changes), "
            f"so costs are a rounding error: **+{R['ls_ann']:.0f}%/yr gross → +{R['net_ann']:.0f}%/yr "
            "net**. But that misses the point. Every dollar of that spread is **market/growth beta**: "
            "you'd be paying a short-borrow to run a leveraged bet on tech beating value, dressed up "
            "as a demographic insight. When growth cracks (see 2022), the whole thing inverts. There "
            "is no age *alpha* to harvest — just a factor tilt you could get cheaper, and cleaner, "
            "elsewhere."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The confound is the lesson.** 'Young CEO' and 'growth-tech founder' are nearly the "
            "same set of companies over this era. To isolate *age* you'd need young CEOs of *boring* "
            "firms and old CEOs of *exciting* ones — a matched sample the curated table can't provide.\n"
            "- **The corporate-behaviour claim may still be true.** Serfling and Yim measured "
            "*policies* (R&D, M&A, leverage), not stock returns. Riskier firms ≠ higher returns — "
            "often the opposite. This study only busts the *trade*, not their finding.\n\n"
            "*Think age carries alpha? Fork this, match each young CEO to an old CEO in the same "
            "sector and size, and re-run the spread. Spoiler: the tech tilt is doing all the work.*"
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
            "# The CEO-Age Effect — a quantitative teardown 🔬\n"
            "### Curated CEO→age table · equal-weight long-young/short-old · Newey-West HAC *t* · CAPM alpha-vs-beta · label-shuffle placebo · cutoff & regime sweeps · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Young--CEO_edge%3F: Misattributed](https://img.shields.io/badge/Young--CEO_edge%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its null.* We map large-cap CEOs to their age from "
            "public birth years, build the equal-weight long-young / short-old book, and test whether "
            "**any return survives once the market is removed** — with the autocorrelation-robust bar "
            "the desk requires.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance monthly total-return series for a "
            f"hand-curated {R['n_total']}-CEO table (as-of 2026-06-30, panel fp `{R['fp_panel']}`); the "
            "offline core and tests run on a deterministic synthetic world. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Long-young/short-old **+{R['ls_ann']:.1f}%/yr**, but Newey-West "
            f"HAC **t = {R['ls_t']:.2f}** ({R['hac_lags']} lags, n = {R['n_months']}); label-shuffle "
            f"placebo **p = {R['placebo_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | CAPM **alpha = {R['alpha_ann']:+.1f}%/yr** at HAC "
            f"**t = {R['alpha_t']:.2f}** (market beta **{R['beta']:+.2f}**, t {R['beta_t']:.1f}). "
            f"Net +{R['net_ann']:.1f}%/yr is pure growth beta. |\n"
            f"| **Young-CEO edge?** | `MISATTRIBUTED` | Young Sharpe **{R['young_sh']:.2f}** < old "
            f"**{R['old_sh']:.2f}**; the spread flips sign by regime (+27→−34→+17 %/yr) and depends on "
            "the arbitrary age cutoff. |\n\n"
            "> 💡 In plain words: the raw spread is real-ish but it is *beta*, not *alpha* — young "
            "founders run high-beta growth-tech, and this window rewarded that factor, not the birthday."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_i \\in \\{\\text{young}, \\text{old}\\}$ be firm $i$'s CEO-age bucket and $r_{it}$ "
            "its month-$t$ total return. The equal-weight long/short is "
            "$LS_t = \\bar r^{\\,\\text{young}}_t - \\bar r^{\\,\\text{old}}_t$.\n\n"
            "- **H₁ (premium).** $\\mathbb{E}[LS] > 0$ at Newey-West HAC $|t| \\ge 2$.\n"
            "- **H₂ (alpha, not beta).** In $LS_t = \\alpha + \\beta\\,\\text{MKT}_t + \\varepsilon_t$, "
            "the intercept $\\alpha > 0$ at HAC $|t| \\ge 2$ — the edge is *not* just market exposure.\n"
            "- **H₃ (robustness).** The sign/magnitude is stable across the age cutoff and across "
            "sub-periods.\n\n"
            f"We find **H₁ not rejected** ($t$ = {R['ls_t']:.2f}, placebo $p$ = {R['placebo_p']:.2f}), "
            f"**H₂ decisively not rejected** ($\\alpha$-$t$ = {R['alpha_t']:.2f}; $\\beta$ = "
            f"{R['beta']:.2f} at $t$ = {R['beta_t']:.1f}), and **H₃ rejected** (sign flips by regime, "
            "$t$ wanders with the cutoff). The Signal axis is additionally capped below `REAL` **by "
            "construction**: a curated ~40-name table whose young bucket is a growth-tech / recent-IPO "
            "cohort is confounded and not survivorship-free."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the confound that eats the claim\n\n"
            "This is the textbook **omitted-variable** trap. `young CEO` is nearly collinear with "
            "`founder-led growth tech that IPO'd recently` (TSLA, META, COIN, HOOD, DASH, SHOP, "
            "SNAP…). Over 2018-2026 the growth/tech factor had a huge, regime-dependent run. So a raw "
            "long-young/short-old spread is *predominantly a bet on that factor*. The honest test is "
            "not 'is the spread positive?' (a factor tilt in a factor-friendly window will be) but "
            "'**is there an intercept once you regress the market out?**' — the CAPM alpha in Beat 4b."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Table.** Curated CEO birth years → age at the scoring date (`data.curated_ages`), "
            "bucketed young (< 55) / old, joined to real yfinance monthly total returns.\n"
            "- **Book.** Equal-weight young minus equal-weight old, `data.build_returns` (dollar-"
            "neutral; a name enters only in months it has a price).\n"
            "- **H₁.** Newey-West HAC *t* on the mean monthly L/S (`strategy.hac_mean_t`, automatic "
            "Bartlett lag).\n"
            "- **H₂.** CAPM regression with a HAC covariance (`strategy.capm_alpha`): alpha, its *t*, "
            "the market beta.\n"
            "- **Null.** Label-shuffle placebo — reshuffle young/old across names, recompute the HAC "
            "|t| (`strategy.placebo_pvalue`).\n"
            "- **Robustness.** Cutoff sweep (age < 50 / 55 / 60) and a three-regime sub-period sweep.\n"
            "- **Frictions.** One-way cost × turnover on both legs + an annual borrow on the short "
            "(old) leg; gross and net.\n"
            "- **Positive control.** A deterministic synthetic world where young firms carry BOTH a "
            "higher beta and a plantable `age_alpha`; the engine must light up the *alpha* t only when "
            "a real premium is planted — averaged over 25 seeds (house rule).\n\n"
            "Timing: CEO age is public years in advance (no look-ahead); membership is calendar-known "
            "so the contemporaneous monthly return is tradable — a conservative one-month formation "
            "lag moves the *t* from "
            f"{R['ls_t']:.2f} to 0.83, i.e. nothing."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The raw premium — H₁\n\n"
            "The long/short's annualised return and its Newey-West HAC *t*, with the label-shuffle "
            "placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bs = st.bucket_stats(RET); h = bs['ls']['hac']\n"
            "    pl = st.placebo_pvalue(PX, data, n_perm=2000)\n"
            "    ls_ann = bs['ls']['ann_ret']*100; t = h['t']; p = pl['p']; lags = h['lags']\n"
            "else:\n"
            f"    ls_ann, t, p, lags = {R['ls_ann']}, {R['ls_t']}, {R['placebo_p']}, {R['hac_lags']}\n"
            "if HAVE_REAL:\n"
            "    cum = (1+RET['ls']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "    ax.plot(cum.index, cum.values, c=GREEN, lw=1.6)\n"
            "    ax.axhline(1, c=GREY, ls='--', lw=1)\n"
            "    ax.set_ylabel('growth of $1 (long-young/short-old)')\n"
            "    ax.set_title(f'Long-young/short-old: +{ls_ann:.0f}%/yr but HAC t = {t:.2f} (placebo p {p:.2f})')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f'L/S {ls_ann:+.1f}%/yr | Newey-West HAC t {t:+.2f} ({lags} lags) | label-shuffle placebo p {p:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **not rejected.** +{R['ls_ann']:.0f}%/yr looks like a lot, but "
            f"the HAC *t* is only {R['ls_t']:.2f} — well short of 2 — and a *random* young/old "
            f"relabelling of the same CEOs produces a |t| this big **{R['placebo_p']*100:.0f}% of the "
            "time**. No reliable premium on the tape."
        ),
        md(
            "### 4b · Alpha vs beta — H₂ (the decisive control)\n\n"
            "Regress the L/S on the market. If the 'young premium' is just a high-beta growth tilt, "
            "the **intercept (alpha)** collapses even while the raw spread looks big."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ca = st.capm_alpha(RET['ls'].to_numpy(), RET['mkt'].to_numpy())\n"
            "    a_ann = ca['alpha']*12*100; a_t = ca['alpha_t']; beta = ca['beta']; b_t = ca['beta_t']; r2 = ca['r2']\n"
            "    x = RET['mkt'].to_numpy()*100; y = RET['ls'].to_numpy()*100\n"
            "else:\n"
            f"    a_ann, a_t, beta, b_t, r2 = {R['alpha_ann']}, {R['alpha_t']}, {R['beta']}, {R['beta_t']}, {R['r2']}\n"
            "    rng = np.random.default_rng(0); x = rng.normal(0,4,101); y = beta*x + rng.normal(0,5,101)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=18, c=GREY, alpha=.7)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, a_ann/12 + beta*xs, c=RED, lw=1.8, label=f'beta {beta:+.2f} (t {b_t:.1f})')\n"
            "ax.axhline(0, c='k', lw=.6); ax.axvline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('market monthly return %'); ax.set_ylabel('long/short monthly return %')\n"
            "ax.legend(); ax.set_title(f'The L/S IS market beta: alpha {a_ann:+.1f}%/yr at t {a_t:.2f} (not significant)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CAPM: alpha {a_ann:+.2f}%/yr (t {a_t:+.2f}) | market beta {beta:+.2f} (t {b_t:+.1f}) | R2 {r2:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **decisively not rejected.** The book carries a real market beta "
            f"(**{R['beta']:+.2f}**, *t* = {R['beta_t']:.1f}) — it goes up when the market does — but "
            f"the **alpha is {R['alpha_ann']:+.1f}%/yr at *t* = {R['alpha_t']:.2f}**, indistinguishable "
            "from zero. The entire 'young-CEO premium' is the market exposure. There is no age alpha."
        ),
        md(
            "### 4c · Robustness — H₃ (cutoff & regime)\n\n"
            "A real effect is stable. This one's *t* wanders with the arbitrary age line, and its "
            "*sign* flips with the macro regime."
        ),
        code(
            "cut = " + repr(R["cutoff"]) + "\n"
            "sub = " + repr(R["sub"]) + "\n"
            "if HAVE_REAL:\n"
            "    cs = st.cutoff_sweep(data, PX, [50,55,60])\n"
            "    clabs = list(cs.index); cts = cs['ls_t'].tolist()\n"
            "    splits=[('2018-2020','2018-01-01','2020-12-31'),('2021-2022','2021-01-01','2022-12-31'),('2023-2026','2023-01-01','2026-06-30')]\n"
            "    sp = st.subperiod_sweep(RET, splits); slabs=list(sp.index); sats=sp['alpha_t'].tolist()\n"
            "else:\n"
            "    clabs=[c[0] for c in cut]; cts=[c[2] for c in cut]\n"
            "    slabs=[s[0] for s in sub]; sats=[s[3] for s in sub]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))\n"
            "a1.bar(clabs, cts, color=GREY, width=.5); a1.axhline(2,c=RED,ls='--',lw=1); a1.axhline(-2,c=RED,ls='--',lw=1)\n"
            "a1.set_ylabel('long/short HAC t'); a1.set_title('t wanders with the cutoff, never clears 2')\n"
            "cols=[GREEN if v>0 else RED for v in sats]\n"
            "a2.bar(slabs, sats, color=cols, width=.5); a2.axhline(2,c=GREY,ls='--',lw=1); a2.axhline(-2,c=GREY,ls='--',lw=1)\n"
            "a2.set_ylabel('CAPM alpha t'); a2.set_title('alpha t flips sign by regime (+2.0 / -2.4 / 0.0)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cutoff t:', dict(zip(clabs,[round(t,2) for t in cts])))\n"
            "print('regime alpha t:', dict(zip(slabs,[round(t,2) for t in sats])))"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** Slide the 'young' line and the *t* runs 0.1 → 0.9 → "
            "1.9 (never significant). Split by era and the alpha *t* is +2.0 (2018-20 growth boom), "
            "−2.4 (2021-22 rate shock), ≈0 (2023-26). A single sign that reverses with interest rates "
            "is the signature of a **factor exposure**, not a CEO-age premium."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ not rejected (HAC *t* {R['ls_t']:.2f}, placebo *p* "
            f"{R['placebo_p']:.2f}); capped below `REAL` by construction (curated, confounded, not "
            "survivorship-free).\n"
            f"- **Tradability `MIRAGE`** — CAPM alpha {R['alpha_ann']:+.1f}%/yr at *t* {R['alpha_t']:.2f}; "
            f"the +{R['net_ann']:.0f}%/yr net spread is market/growth beta you can rent cheaper.\n"
            f"- **Young-CEO edge? `MISATTRIBUTED`** — young Sharpe {R['young_sh']:.2f} < old "
            f"{R['old_sh']:.2f}; the gap is sector/size/vintage, and its sign flips by regime."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Turnover is negligible (age is nearly static), so frictions barely dent it — which only "
            "underlines that there is nothing real to charge them against."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nc = st.net_of_costs(RET, cost_bps=5.0, borrow_ann_bps=75.0, annual_turnover=0.30)\n"
            "    gross, net = nc['gross_ann']*100, nc['net_ann']*100\n"
            "    gsh, nsh = nc['gross_sharpe'], nc['net_sharpe']\n"
            "else:\n"
            f"    gross, net, gsh, nsh = {R['ls_ann']}, {R['net_ann']}, {R['gross_sh']}, {R['net_sh']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREY, GREY], width=.5)\n"
            "ax.set_ylabel('long/short return %/yr')\n"
            "ax.set_title(f'Gross +{gross:.0f}%/yr -> net +{net:.0f}%/yr (Sharpe {gsh:.2f}->{nsh:.2f}) — but it is all beta')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{gross:.1f}%/yr (Sharpe {gsh:.2f}) -> net +{net:.1f}%/yr (Sharpe {nsh:.2f}); 5bps/leg x 0.30 turnover + 75bps borrow')"
        ),
        md(
            "> 💡 In plain words: net still ~7%/yr, but its Sharpe (~0.34) is *entirely* the market "
            "beta — a growth ETF gives you the same exposure without a short-borrow. There is no "
            "age-specific residual to harvest. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or would it print alpha-*t* ≈ 0 no matter what? Build "
            "a synthetic world where young firms carry BOTH a higher beta (like the real one) AND a "
            "genuine planted `age_alpha`. The CAPM alpha *t* must stay flat at the null (beta tilt "
            "only) and rise sharply once a real premium is planted — averaged over 25 seeds so no "
            "single lucky seed fakes it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "alphas = [0.0, 0.002, 0.004, 0.006, 0.008, 0.012]\n"
            "res = [st.synthetic_mean_alpha_t(data, age_alpha=a, n_seeds=25) for a in alphas]\n"
            "at = [r['mean_alpha_t'] for r in res]; rt = [r['mean_raw_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.plot([a*12*100 for a in alphas], at, 'o-', c=GREEN, lw=2, label='CAPM alpha t (the honest test)')\n"
            "ax.plot([a*12*100 for a in alphas], rt, 's--', c=GREY, lw=1.6, label='raw spread HAC t (beta-contaminated)')\n"
            "ax.axhline(2, c=RED, ls=':', lw=1.2, label='t = 2')\n"
            "ax.set_xlabel('planted young-minus-old alpha (%/yr)'); ax.set_ylabel('mean t (25 seeds)')\n"
            "ax.legend(); ax.set_title('The engine catches a REAL age premium — the real tape shows none')\n"
            "plt.tight_layout(); plt.show()\n"
            "for a, r in zip(alphas, res): print(f'age_alpha {a*12*100:+5.1f}%/yr -> mean alpha t {r[\"mean_alpha_t\"]:+.2f}, mean raw t {r[\"mean_raw_t\"]:+.2f}')"
        ),
        md(
            "At the null (`age_alpha = 0`) the CAPM alpha *t* sits at ~0 — no false signal, even though "
            "young firms carry a higher beta (the raw *t* is noisy but the alpha is clean). Plant a "
            "real premium and the alpha *t* sails past 2. So the engine is a faithful detector, and "
            f"the flat real-tape alpha (*t* = {R['alpha_t']:.2f}) is a statement about **the world**: a "
            "CEO's age carries no return alpha on this tape, and a curated ~40-name growth-tilted table "
            "could never certify one if it did. For the sibling C-suite folklore, see "
            "[543 Western-Zodiac-CEO](../../543-western-zodiac-ceo/) and [391 CEO-Turnover](../../391-ceo-turnover/)."
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
