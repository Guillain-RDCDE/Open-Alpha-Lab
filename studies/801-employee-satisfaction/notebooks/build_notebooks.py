"""Generate the two narrative notebooks for Study 801 (Employee Satisfaction).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached monthly
tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total return,
# 17 perennial "100 Best Companies to Work For" members vs SPY, 2016-01 -> 2026-06, 126 months).
R = dict(
    start="2016-01-31", end="2026-06-30", n_months=126, n_names=17,
    basket_ann=20.16, market_ann=15.02, basket_vol=19.4, market_vol=15.2,
    alpha_monthly_bps=23.2, alpha_ann=2.82, beta=1.15,
    t_alpha_nw=1.10, t_alpha_ols=1.06,
    excess_mean_bps=42.8, excess_t_nw=1.90, excess_t_ols=1.96, ir=0.60,
    hit=74, hit_pct=58.7, wilson=(50.0, 66.9),
    placebo_mean=0.20, placebo_sd=1.18, placebo_n=5000, placebo_p=0.013, placebo_k=17,
    early_alpha=8.12, early_t=2.77, early_n=63,
    late_alpha=-2.25, late_t=-0.59, late_n=63,
    to_pct=2.4, drag5=0.01, drag10=0.03,
    net5_alpha=2.81, net5_t=1.10, net10_alpha=2.79, net10_t=1.09,
    syn_null_mean=-0.07, syn_null_sd=0.96, syn_null_fire=1, syn_planted_t=2.99, syn_planted_alpha=3.98,
    fp="4116fe2691f7",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Persist%3F: Busted](https://img.shields.io/badge/Persist%3F-Busted-8b949e?style=flat-square)\n\n"
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

from employee_satisfaction import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    MONTHLY, BASKET_PRESENT, CONTROL_PRESENT = data.load_real()
    BASKET = st.equal_weight_basket(MONTHLY, BASKET_PRESENT)
    MARKET = MONTHLY[data.MARKET]
else:
    MONTHLY = BASKET_PRESENT = CONTROL_PRESENT = BASKET = MARKET = None
print("real cache present:", HAVE_REAL, "| basket names:",
      (0 if BASKET_PRESENT is None else len(BASKET_PRESENT)),
      "| months:", (0 if MONTHLY is None else len(MONTHLY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the best companies to work for beat the market? 😊\n"
            "### Employee satisfaction as a hidden asset — a famous finding, honestly re-run on a "
            "hand-picked basket of survivors\n\n"
            + BADGES +
            "It's one of the most quoted results in behavioural finance: a portfolio of Fortune's "
            "**\"100 Best Companies to Work For\"** beats the market (Alex Edmans, 2011). The story "
            "is lovely — happy employees quietly compound into earnings the market forgets to "
            "price. So we did the obvious thing: hand-picked a basket of companies that have been "
            "*perennial* members of that list and still trade today, and raced it against the "
            "S&P 500.\n\n"
            "It wins. And that's exactly where the honesty has to start.\n\n"
            "> 📓 **Plain-language layer.** Want the alpha *t*-stats, the survivor placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 17 listed perennial members, equal-weight, monthly total "
            "return, 2016→2026 (126 months). We **hand-picked known survivors** — that stacks the "
            "deck, and we say so throughout. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the basket beat the market? | **Yes, on paper.** +{R['basket_ann']:.1f}%/yr "
            f"vs +{R['market_ann']:.1f}%/yr for the S&P 500 — about **5 points a year**. |\n"
            f"| Is that skill, or just more risk? | **Mostly more risk.** The basket carries "
            f"**β = {R['beta']:.2f}** — it's a leveraged bet on the market. Strip that out and the "
            f"'skill' left over is only **+{R['alpha_ann']:.1f}%/yr**, and it's not statistically "
            f"reliable (*t* = {R['t_alpha_nw']:.2f}, the bar is 2). |\n"
            f"| Did we cheat by picking winners? | **Partly, unavoidably.** We chose companies "
            f"that are *still around and still great to work for in 2026* — a survivor's list. "
            f"But when we pit it against random baskets of other large survivors, ours still comes "
            f"out ahead ({R['placebo_p']*100:.1f}% of them match it), so it isn't *only* that. |\n"
            f"| Does the edge last? | **No.** It was strong in 2016-2021 and turned **negative** "
            f"in 2021-2026. One good stretch, then gone. |\n\n"
            "> The basket beats the market — but almost all of that is just riding the market "
            "harder, and the leftover 'satisfaction' edge is small, shaky, and already faded."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Happy employees stay longer, work harder and serve customers better. That shows "
            "up in profits the market is too short-sighted to price — so a list of the best "
            "employers is a list of quietly-underpriced stocks.\"*\n\n"
            "This isn't bar-stool folklore — it's a careful, published finding (Edmans 2011, "
            "*Journal of Financial Economics*): the \"100 Best Companies to Work For\" earned a "
            "risk-adjusted alpha of about 3.5%/yr from 1984-2009. Employee satisfaction as an "
            "**intangible asset** the market underprices. We're taking the strong version "
            "seriously — and then asking what's left of it on a modern tape, once we're honest "
            "about how the basket was built."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a rare thing: a signal you can read off a **magazine list**, no "
            "data feed required, tapping a genuine market blind spot rather than a fleeting "
            "mispricing. It would also say something deep — that the market systematically "
            "under-values human capital. That's worth taking seriously. It's *also* exactly the "
            "kind of feel-good story that survives because nobody checks whether the "
            "'outperformance' is just owning a basket of big tech-y winners during a decade that "
            "loved big tech-y winners."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The race.** Equal-weight the **{R['n_names']}** listed perennial members, "
            "rebalance monthly, and compare total return to the S&P 500 over 126 months.\n"
            "- **The risk check.** Beating the market isn't skill if you just took more market "
            "risk. We measure **alpha** — the return left *after* subtracting what the basket's "
            "market exposure (its beta) already explains.\n"
            "- **The survivor check.** We hand-picked known survivors. So we build thousands of "
            "**random** baskets of *other* large-cap survivors and ask: does a random survivor "
            "basket earn alpha too? If yes, our 'edge' is just survivorship.\n"
            "- **The persistence check.** Split the decade in half. A real intangible premium "
            "should show up in both halves, not just one."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race.** Basket vs market, annualized."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(BASKET, MARKET)\n"
            "    ba, ma, al, bt = h['basket_ann_pct'], h['market_ann_pct'], h['alpha_ann_pct'], h['beta']\n"
            "else:\n"
            "    ba, ma, al, bt = R['basket_ann'], R['market_ann'], R['alpha_ann'], R['beta']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['Best-to-Work-For\\nbasket','S&P 500\\n(market)','leftover ALPHA\\n(risk-adjusted)'],\n"
            "       [ba, ma, al], color=[GREEN, GREY, AMBER], width=.6)\n"
            "for i,v in enumerate([ba, ma, al]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualized return (%)')\n"
            "ax.set_title(f'Beats the market by ~5pts/yr - but carries beta = {bt:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'basket {ba:+.2f}%/yr | market {ma:+.2f}%/yr | risk-adjusted alpha {al:+.2f}%/yr (beta {bt:.2f})')"
        ),
        md(
            f"The basket wins the raw race by ~5 points a year. But look at the third bar: once "
            f"you subtract the return the basket's **market exposure** (β = {R['beta']:.2f}) "
            f"already earns, the *skill* left over is just **+{R['alpha_ann']:.1f}%/yr** — and "
            f"that number is not statistically reliable (*t* = {R['t_alpha_nw']:.2f}; you want ≥ 2 "
            "before you believe it). Most of the 'outperformance' is simply owning a basket that "
            "bounces around **more** than the market.\n\n"
            "**Next, the survivor check.** We picked winners on purpose. Is that the whole story?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(BASKET, MARKET)\n"
            "    pl = st.survivor_placebo(MONTHLY, CONTROL_PRESENT, MARKET,\n"
            "                             basket_size=len(BASKET_PRESENT), n_draws=2000)\n"
            "    alphas = pl['alphas_ann_pct']; obs = h['alpha_ann_pct']\n"
            "else:\n"
            "    rng = np.random.default_rng(801)\n"
            "    alphas = rng.normal(R['placebo_mean'], R['placebo_sd'], 2000); obs = R['alpha_ann']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(alphas, bins=40, color=GREY, alpha=.85, label='random baskets of OTHER large-cap survivors')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'our satisfaction basket {obs:+.2f}%/yr')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('risk-adjusted alpha of the basket (%/yr)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Random survivor baskets earn ~0 alpha; ours beats {100-R['placebo_p']*100:.0f}% of them\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"random survivor-basket alpha ~ {R['placebo_mean']:+.2f}%/yr; our basket beats \"\n"
            "      f\"{100-R['placebo_p']*100:.1f}% of them (placebo p = {R['placebo_p']:.3f})\")"
        ),
        md(
            f"This is the one genuinely *favourable* result. A **random** basket of other big "
            f"survivors earns basically **zero** alpha (~{R['placebo_mean']:+.1f}%/yr), and our "
            f"satisfaction basket beats **{100-R['placebo_p']*100:.0f}%** of them. So the edge "
            "isn't *purely* 'we cheated by picking survivors' — there's something about *these* "
            "names. The catch: these names are also tech-and-quality-heavy over a decade that "
            "showered exactly those with returns, and that tilt (not employee happiness) is the "
            "more likely explanation — which the next check makes vivid.\n\n"
            "**Finally, does it last?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_split(BASKET, MARKET)\n"
            "    e, l = sp['early']['alpha_ann_pct'], sp['late']['alpha_ann_pct']\n"
            "    et, lt = sp['early']['t_alpha_nw'], sp['late']['t_alpha_nw']\n"
            "else:\n"
            "    e, l, et, lt = R['early_alpha'], R['late_alpha'], R['early_t'], R['late_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['2016-2021\\n(first half)','2021-2026\\n(second half)'], [e, l],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    ax.annotate(f'{v:+.1f}%/yr\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('risk-adjusted alpha (%/yr)')\n"
            "ax.set_title('The edge lived in the first half - and reversed in the second')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'first half {e:+.2f}%/yr (t={et:+.2f}) | second half {l:+.2f}%/yr (t={lt:+.2f})')"
        ),
        md(
            f"There's the tell. The alpha was a real **+{R['early_alpha']:.1f}%/yr** in 2016-2021 "
            f"(*t* = {R['early_t']:.2f} — it clears the bar on its own) and turned "
            f"**{R['late_alpha']:.1f}%/yr** in 2021-2026. A durable intangible premium shouldn't "
            "evaporate and flip sign halfway through. This looks like one lucky stretch (or an "
            "anomaly that faded once Edmans' paper made it famous), not a permanent feature of "
            "how markets price happy workplaces."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The basket beats the market and even beats random survivor "
            f"baskets, but the *risk-adjusted* alpha Edmans claimed is only +{R['alpha_ann']:.1f}"
            f"%/yr at *t* = {R['t_alpha_nw']:.2f} — under the bar — and we picked survivors and "
            "controlled only for the market. Literature-supported, tape-unproven.\n"
            f"- **Tradability — Mirage.** Trading costs are trivial here; the problem is there's "
            f"no reliable alpha to collect. The ~5pts/yr is **β = {R['beta']:.2f}** — leverage you "
            "can buy on an index fund — not skill.\n"
            "- **\"Does it persist?\" — Busted.** Strong in the first five years, negative in the "
            "next. One window, then gone."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The honest missing control.** Edmans used a *four-factor* alpha; we used a "
            "*one-factor* (market) alpha because point-in-time factor data is hard to get offline. "
            "A four-factor model would strip out the size/value/**momentum**/quality tilts this "
            "basket carries — and would almost certainly shrink the +2.8%/yr further. Our number "
            "is an **upper bound**.\n"
            "- **The survivorship you can't fully undo.** We named the members who left the tape "
            "(Whole Foods, Ultimate Software, Nordstrom — all acquired or taken private) and "
            "excluded them; a truly clean test needs the *point-in-time* list each year, which "
            "Fortune doesn't give away.\n"
            "- **Sibling studies:** [392-glassdoor-sentiment](../../392-glassdoor-sentiment/) "
            "(the crowd-rating version), [526-intangible-value](../../526-intangible-value/) "
            "(intangibles broadly), [751-fortune-500-inclusion](../../751-fortune-500-inclusion/) "
            "(a different Fortune list, tested as an event study) — see "
            "[docs/references.md](../docs/references.md).\n\n"
            "*Think the satisfaction premium is real and durable? Show a **four-factor** alpha, "
            "point-in-time membership, and persistence in the 2020s — then we'll talk.*"
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
            "# Employee satisfaction alpha — a quantitative teardown 🔬\n"
            "### The CAPM-alpha Newey-West/OLS split · a random survivor-basket placebo · a "
            "first/second-half persistence cut · a costed long-only timer · a 20-seed synthetic "
            "power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **Edmans (2011): a \"100 Best Companies to Work For\" portfolio earns "
            "positive risk-adjusted alpha** — is tested here as a held equal-weight basket vs the "
            "market, with two caps stated up front: the basket is **hand-picked from known "
            "survivors** (survivorship on the Signal axis) and we run a **CAPM (market-only) "
            "alpha, not a 4-factor alpha**, so any positive alpha is an *upper bound*.\n\n"
            "> ⚠️ **Data note.** yfinance daily total-return closes → monthly, "
            f"{R['start']} → {R['end']} ({R['n_months']} months), {R['n_names']} listed perennial "
            "members + SPY + a 34-name survivor placebo pool, cached. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | CAPM α **+{R['alpha_ann']:.2f}%/yr** at NW "
            f"**t = {R['t_alpha_nw']:.2f}** (OLS {R['t_alpha_ols']:.2f}), β = {R['beta']:.2f}; "
            f"excess-over-market NW t = {R['excess_t_nw']:.2f}; beats random survivor baskets "
            f"(placebo p = {R['placebo_p']:.3f}) but survivor-picked + market-only control |\n"
            f"| **Tradability** | `MIRAGE` | turnover {R['to_pct']:.1f}%/mo → cost drag "
            f"{R['drag5']:.2f}%/yr; net α {R['net5_alpha']:.2f}%/yr at t = {R['net5_t']:.2f} — no "
            f"certified alpha; the +5pp/yr is β = {R['beta']:.2f} |\n"
            f"| **Persist?** | `BUSTED` | first half α +{R['early_alpha']:.2f}%/yr "
            f"(t = {R['early_t']:.2f}); second half {R['late_alpha']:.2f}%/yr (t = {R['late_t']:.2f}) |\n\n"
            "> 💡 In plain words: the basket out-returns SPY, but at β = 1.15 that is mostly "
            "leverage; the risk-adjusted residual is sub-2, front-loaded into 2016-2021, and "
            "negative since — a weak, non-persistent signal on a survivor-picked basket."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "Let $r^B_t$ be the equal-weight basket's monthly total return and $r^M_t$ the market "
            "(SPY). Edmans' claim is a positive **risk-adjusted** intercept:\n\n"
            "$$ r^B_t = \\alpha + \\beta\\, r^M_t + \\varepsilon_t, \\qquad H_1:\\ \\alpha > 0. $$\n\n"
            "- **H₁ (alpha).** $\\alpha > 0$ with a HAC $t \\ge 2$ — the satisfaction premium the "
            "paper reports.\n"
            "- **H₂ (not just survivorship).** The basket's $\\alpha$ exceeds that of random "
            "baskets drawn from a pool of large-cap survivors *not* chosen for workplace prestige.\n"
            "- **H₃ (persistence).** $\\alpha$ is present in both halves of the sample, not one "
            "lucky window.\n\n"
            f"We find **H₁ not supported** (NW t = {R['t_alpha_nw']:.2f} < 2; β = {R['beta']:.2f} "
            f"absorbs most of the raw +5pp/yr), **H₂ supported** (placebo p = {R['placebo_p']:.3f}), "
            f"**H₃ rejected** (early t = {R['early_t']:.2f}, late t = {R['late_t']:.2f}). The one "
            "check that *helps* the claim (H₂) is exactly the one that can't certify magnitude — a "
            "control sharing the survivorship bias speaks to *existence*, never to *size*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly basket returns are mildly autocorrelated, so the primary is a **Newey-West "
            "(3-lag) HAC t** on the CAPM intercept, with a plain OLS t as a cross-check. We also "
            "read the **excess-over-market** series $d_t = r^B_t - r^M_t$ directly (mean, NW t, "
            "annualized information ratio) — a beta-agnostic 'does it beat the market' view — and "
            "put a **Wilson** interval on the monthly win-rate. The survivorship bias is not just "
            "named; it is **sized** by a 5,000-draw random survivor-basket placebo, and the "
            "persistence question is a **pre-declared** halving of the sample, not a snooped "
            "break-point."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {R['n_names']} listed perennial \"100 Best\" members, equal-weight, "
            f"monthly rebalanced, {R['start']} → {R['end']} ({R['n_months']} months).\n"
            "- **Headline.** CAPM α (NW + OLS t) + excess-over-market (NW t, IR) + Wilson win-rate.\n"
            "- **Survivorship.** 5,000 random equal-weight baskets from a 34-name large-cap "
            "survivor pool not chosen for prestige; right-tail placebo p.\n"
            "- **Persistence.** First-half vs second-half CAPM α, each with its own NW t.\n"
            "- **Execution (timer).** Equal weights set at the month-end close, earn next month "
            "(calendar rebalance, zero look-ahead); cost = one-way × NAV × realised turnover; "
            "long-only, no borrow.\n"
            "- **Control.** Synthetic monthly market + β·mkt + idio basket, plantable α knob; the "
            "null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline CAPM alpha and the excess series\n\n"
            "The market model on 126 monthly returns; the intercept is the risk-adjusted alpha, "
            "the excess series is the beta-agnostic cross-check."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(BASKET, MARKET)\n"
            "    print(f\"months {h['n_months']}\")\n"
            "    print(f\"basket {h['basket_ann_pct']:+.2f}%/yr (vol {h['basket_vol_pct']:.1f}%)  vs  \"\n"
            "          f\"market {h['market_ann_pct']:+.2f}%/yr (vol {h['market_vol_pct']:.1f}%)\")\n"
            "    print(f\"CAPM alpha {h['alpha_ann_pct']:+.2f}%/yr (beta {h['beta']:.2f}); \"\n"
            "          f\"NW t = {h['t_alpha_nw']:+.2f}, OLS t = {h['t_alpha_ols']:+.2f}\")\n"
            "    print(f\"excess-over-market {h['excess_mean_bps']:+.1f} bps/mo; NW t = {h['excess_t_nw']:+.2f}, \"\n"
            "          f\"OLS t = {h['excess_t_ols']:+.2f}, IR {h['ir_ann']:.2f}/yr\")\n"
            "    print(f\"win-rate {h['hit']}/{h['n_months']} = {h['hit_rate']*100:.1f}% \"\n"
            "          f\"Wilson [{h['hit_lo']*100:.1f}%, {h['hit_hi']*100:.1f}%]\")\n"
            "    al, tnw, tols, ex = h['alpha_ann_pct'], h['t_alpha_nw'], h['t_alpha_ols'], h['excess_t_nw']\n"
            "else:\n"
            "    al, tnw, tols, ex = R['alpha_ann'], R['t_alpha_nw'], R['t_alpha_ols'], R['excess_t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "bars = ['CAPM alpha\\nNW t','CAPM alpha\\nOLS t','excess-over-mkt\\nNW t']\n"
            "vals = [tnw, tols, ex]\n"
            "ax.bar(bars, vals, color=[AMBER, AMBER, AMBER], width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='|t| = 2 bar')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('t-statistic'); ax.set_ylim(0, 2.4)\n"
            "ax.set_title(f'Every t sits under the bar (alpha {al:+.2f}%/yr)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the risk-adjusted alpha is **+{R['alpha_ann']:.2f}%/yr** but "
            f"its HAC t is only **{R['t_alpha_nw']:.2f}** — the market beta (β = {R['beta']:.2f}) "
            f"explains most of the raw +5pp/yr gap. The beta-agnostic excess series is a little "
            f"stronger (NW t = {R['excess_t_nw']:.2f}) but still under 2, and the win-rate's "
            f"Wilson floor sits on {R['wilson'][0]:.1f}%. On the claim Edmans actually made — a "
            "*risk-adjusted* premium — this tape can't certify it."
        ),
        md(
            "### 4b · The survivorship placebo — random baskets of other survivors\n\n"
            "We picked known survivors. The honest yardstick: does a *random* basket of large-cap "
            "survivors *not* chosen for prestige earn alpha too?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(BASKET, MARKET)\n"
            "    pl = st.survivor_placebo(MONTHLY, CONTROL_PRESENT, MARKET,\n"
            "                             basket_size=len(BASKET_PRESENT), n_draws=3000)\n"
            "    alphas = pl['alphas_ann_pct']; obs = h['alpha_ann_pct']\n"
            "    p = st.placebo_pvalue(obs, alphas)\n"
            "else:\n"
            "    rng = np.random.default_rng(801)\n"
            "    alphas = rng.normal(R['placebo_mean'], R['placebo_sd'], 3000)\n"
            "    obs, p = R['alpha_ann'], R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(alphas, bins=45, color=GREY, alpha=.85, label='random survivor baskets (not prestige-picked)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'satisfaction basket {obs:+.2f}%/yr')\n"
            "ax.axvline(np.mean(alphas), c=RED, lw=1.5, ls='--', label=f'random mean {np.mean(alphas):+.2f}%/yr')\n"
            "ax.set_xlabel('CAPM alpha (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Not pure survivorship: placebo p = {p:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"random survivor-basket alpha mean {np.mean(alphas):+.2f}%/yr (sd {np.std(alphas,ddof=1):.2f}); \"\n"
            "      f\"basket {obs:+.2f}%/yr; p = {p:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: random survivor baskets cluster near **zero** alpha "
            f"(~{R['placebo_mean']:+.2f}%/yr), and the satisfaction basket sits in the right tail "
            f"(**p = {R['placebo_p']:.3f}**). So the outperformance is **not** explained by "
            "survivorship-plus-equal-weighting alone. Read this narrowly: it rules out the "
            "*laziest* confound, but the placebo does **not** control for the basket's factor tilt "
            "(tech/quality over a tech/quality decade), and a control that shares the survivorship "
            "bias can speak to *existence*, never to *magnitude*. It doesn't rescue a sub-2 alpha."
        ),
        md(
            "### 4c · Persistence — the decisive cut\n\n"
            "Halve the 126 months. A durable intangible premium should appear in both halves."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_split(BASKET, MARKET)\n"
            "    e, l = sp['early']['alpha_ann_pct'], sp['late']['alpha_ann_pct']\n"
            "    et, lt = sp['early']['t_alpha_nw'], sp['late']['t_alpha_nw']\n"
            "else:\n"
            "    e, l, et, lt = R['early_alpha'], R['late_alpha'], R['early_t'], R['late_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['2016-2021\\n(n={})'.format(R['early_n']), '2021-2026\\n(n={})'.format(R['late_n'])],\n"
            "       [e, l], color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    ax.annotate(f'{v:+.1f}%/yr\\nt={t_:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('CAPM alpha (%/yr)')\n"
            "ax.set_title('Alpha lived in the first half, reversed in the second')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.2f}%/yr (t={et:+.2f}) | late {l:+.2f}%/yr (t={lt:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the full-sample alpha is insignificant because it is "
            f"**entirely front-loaded** — +{R['early_alpha']:.2f}%/yr at t = {R['early_t']:.2f} in "
            f"2016-2021 (clears the bar on its own), then {R['late_alpha']:.2f}%/yr at "
            f"t = {R['late_t']:.2f} in 2021-2026. A premium that flips sign halfway through is the "
            "signature of a lucky window or a post-publication-decayed anomaly — not a permanent "
            "mispricing of human capital. **H₃ rejected.**"
        ),
        md(
            "### 4d · The timer — costs are not the constraint\n\n"
            "Hold the equal-weight basket, rebalance monthly; cost = one-way × NAV × realised "
            "turnover; long-only, no borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.timer_stats(BASKET, MARKET, MONTHLY, BASKET_PRESENT, cost_bps=5.0)\n"
            "    tm10 = st.timer_stats(BASKET, MARKET, MONTHLY, BASKET_PRESENT, cost_bps=10.0)\n"
            "    g, n5, n10 = tm5['gross_alpha_ann_pct'], tm5['net_alpha_ann_pct'], tm10['net_alpha_ann_pct']\n"
            "    tg, t5, t10 = tm5['gross_t_nw'], tm5['net_t_nw'], tm10['net_t_nw']\n"
            "    to = tm5['turnover_1way_pct']\n"
            "else:\n"
            "    g, n5, n10 = R['alpha_ann'], R['net5_alpha'], R['net10_alpha']\n"
            "    tg, t5, t10, to = R['t_alpha_nw'], R['net5_t'], R['net10_t'], R['to_pct']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,(v,t_) in enumerate([(g,tg),(n5,t5),(n10,t10)]):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\nt={t_:+.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('alpha (%/yr)')\n"
            "ax.set_title(f'Turnover only {to:.1f}%/mo - cost barely moves it; alpha was never certified')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'turnover {to:.1f}%/mo | gross {g:+.2f}%/yr (t={tg:+.2f}) -> '\n"
            "      f'net@5 {n5:+.2f} (t={t5:+.2f}) / net@10 {n10:+.2f} (t={t10:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: a buy-and-hold equal-weight basket barely trades "
            f"({R['to_pct']:.1f}%/mo one-way), so cost is **not** the binding constraint "
            f"(drag {R['drag5']:.2f}-{R['drag10']:.2f}%/yr). The binding constraint is that there "
            f"is **no certified alpha to harvest**: gross and net are both ~+{R['alpha_ann']:.1f}"
            f"%/yr at t ≈ {R['t_alpha_nw']:.1f}, and the +5pp/yr of raw outperformance is "
            f"β = {R['beta']:.2f} — leverage you can buy on SPY directly. **Tradability = MIRAGE**: "
            "beta wearing an intangible's clothes."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic monthly market + equal-weight basket of β·mkt + idio names, TUNABLE planted "
            "monthly alpha. The null (alpha = 0) is checked over **20 seeds**, never one stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    w = data.synthetic_world(alpha_bps=0.0, seed=801 + s_)\n"
            "    null_ts.append(st.synthetic_detect(w)['t_alpha_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "w = data.synthetic_world(alpha_bps=40.0, seed=801)\n"
            "planted_t = st.synthetic_detect(w)['t_alpha_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (alpha=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5, label='planted alpha = +40 bps/mo')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('CAPM alpha NW t')\n"
            "ax.set_title('Control: null stays near 0; a planted alpha lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses the bar in "
            f"{R['syn_null_fire']}/20 seeds (≈ the 5% a 2σ bar expects); a planted +40 bp/mo alpha "
            f"reads t = {R['syn_planted_t']:.2f}. The machinery is unbiased — the sub-2 real-tape "
            "t is a genuine null, not a broken detector. *(A faithful-engine / power check only — "
            "never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — CAPM α = +{R['alpha_ann']:.2f}%/yr at NW t = {R['t_alpha_nw']:.2f} "
            f"(OLS {R['t_alpha_ols']:.2f}), β = {R['beta']:.2f}; excess-over-market NW t = "
            f"{R['excess_t_nw']:.2f} (< 2); win-rate {R['hit_pct']:.1f}% (Wilson floor "
            f"{R['wilson'][0]:.1f}%). Beats random survivor baskets (placebo p = {R['placebo_p']:.3f}) "
            "so it isn't pure survivorship — but a hand-picked survivor set + market-only control "
            "caps this at literature-supported, tape-unproven.\n"
            f"- **Tradability `MIRAGE`** — turnover {R['to_pct']:.1f}%/mo (drag {R['drag5']:.2f}%/yr, "
            f"not binding); net α +{R['net5_alpha']:.2f}%/yr at t = {R['net5_t']:.2f}. The +5pp/yr "
            f"of raw outperformance is β = {R['beta']:.2f} — beta you can buy directly, not alpha.\n"
            f"- **Persist? `BUSTED`** — early α +{R['early_alpha']:.2f}%/yr (t = {R['early_t']:.2f}), "
            f"late {R['late_alpha']:.2f}%/yr (t = {R['late_t']:.2f}). One window, then gone."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The four-factor gap.** Edmans' alpha is a *Carhart four-factor* number; ours is "
            "CAPM. The basket loads on size/value/**momentum**/quality that a 4-factor model would "
            "absorb — our +2.8%/yr is an **upper bound**, and a proper factor attribution would "
            "very likely shrink it toward zero. Point-in-time factor returns offline are the "
            "natural next build.\n"
            "- **Point-in-time membership.** The clean test uses each year's *actual* list, not a "
            "2026 recollection of perennial members. Fortune doesn't sell that feed freely; we "
            "named the survivorship (Whole Foods, Ultimate Software, Nordstrom left the tape) and "
            "sized it with the placebo instead.\n"
            "- **Dedup map:** [392-glassdoor-sentiment](../../392-glassdoor-sentiment/) (crowd "
            "ratings, a different construct), [526-intangible-value](../../526-intangible-value/) "
            "(intangibles broadly), [751-fortune-500-inclusion](../../751-fortune-500-inclusion/) "
            "(a size-ranked Fortune list as an event study). This is the **best-employer** list "
            "held as a **basket**.\n\n"
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
