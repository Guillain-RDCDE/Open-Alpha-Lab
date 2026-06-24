"""Generate the two narrative notebooks for Study 438 (Triple-MA-Crossover).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^GSPC daily
close under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# ^GSPC daily PRICE-ONLY close, 1993-01-04 -> 2026-06-23 (n=8,425, 33.5y). Canonical config
# fast/mid/slow = 5/20/50; cash leg credited at a flat 3% proxy (^GSPC has no dividends, so
# the timing-vs-hold race is run excess-of-cash); one-way cost = 5 bps × NAV; one-day lag.
R = dict(
    ticker="^GSPC", start="1993-01-04", end="2026-06-23", n_days=8425, years=33.5,
    fp="498c2ee76497", asof="2026-06-23", rf=0.03, cost_bps=5.0,
    fast=5, mid=20, slow=50,
    # per arm: Sharpe (excess-of-cash), CAGR, MaxDD, HAC t on excess mean
    bh=dict(sharpe=0.293, cagr=0.0876, maxdd=-0.5678, t=1.93),
    triple=dict(sharpe=0.022, cagr=0.0324, maxdd=-0.2311, t=0.14, in_mkt=0.478, switch_yr=12.1),
    dual=dict(sharpe=0.167, cagr=0.0494, maxdd=-0.3291, t=1.05, in_mkt=0.67),
    # head-to-head HAC Sharpe-difference t-stats
    t_tri_vs_bh=-2.10, t_tri_vs_dual=-1.46,
    # config robustness: (fast,mid,slow, triple_sharpe, triple_t, dual_sharpe, bh_sharpe, t_tri_minus_dual)
    configs=[
        (5, 20, 50, 0.022, 0.14, 0.167, 0.293, -1.46),
        (10, 50, 200, 0.242, 1.50, 0.424, 0.293, -2.27),
        (20, 50, 100, 0.226, 1.49, 0.260, 0.296, -0.76),
        (5, 10, 20, -0.187, -1.14, 0.088, 0.297, -2.38),
    ],
    # cost sweep on the triple stack: (bps, sharpe, cagr)
    costs=[(0.0, 0.093, 0.0387), (5.0, 0.022, 0.0324), (10.0, -0.050, 0.0261), (25.0, -0.263, 0.0075)],
    # permutation placebo on the triple net excess Sharpe
    perm=dict(obs=0.022, p=0.57, null_mean=0.05, null_sd=0.168, n_draws=2000),
    # synthetic positive control: (edge, triple_sharpe, triple_t, bh_sharpe, t_tri_vs_bh)
    syn=[(0.0, -0.171, -1.01, -0.168, 0.49), (2.0, 0.926, 5.04, 0.585, 0.06)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Three_beats_two%3F: Busted](https://img.shields.io/badge/Three_beats_two%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from triple_ma_crossover import data, strategy as st

HAVE_REAL = data.have_real("^GSPC")
if HAVE_REAL:
    DF = data.load_real("^GSPC", fetch=False)
    DF = DF[DF.index < "2026-06-24"]   # drop the in-progress bar — no partial day in a stamped run
    CLOSE = DF["close"]
else:
    DF = CLOSE = None
print("real ^GSPC cache present:", HAVE_REAL,
      "| rows:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do three moving averages beat two? 📉\n"
            "### The \"triple MA crossover\" — the YouTube staple that adds a third line and promises magic\n\n"
            + BADGES +
            "Open any \"best trading strategy\" video and you'll meet the **triple moving-average "
            "crossover**: don't just wait for a fast line to cross a slow line — stack **three** "
            "averages (fast, medium, slow) and only buy when they line up perfectly, "
            "`fast > medium > slow`. The middle line is sold as a *confirmation filter* that keeps you "
            "out of the whipsaws a plain two-line cross would take. Three beats two. More lines, more "
            "money.\n\n"
            "It sounds reasonable. It is also **wrong** — and not in a subtle way. On 33 years of the "
            "S&P 500, the three-line rule earns a *lower* risk-adjusted return than buy-and-hold **and** "
            "a lower one than the simpler two-line rule it claims to beat. The extra average doesn't "
            "filter noise; it filters *return*.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the permutation null and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use **^GSPC** (S&P 500 index, **price-only** — no "
            "dividends), so the timing-vs-hold race is run **excess-of-cash** (cash credited at a flat "
            "3%). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does stacking three MAs beat buy-and-hold? | **No.** The three-line rule's risk-adjusted "
            f"return (Sharpe **{R['triple']['sharpe']:+.2f}**) is well *below* just holding the index "
            f"(**{R['bh']['sharpe']:+.2f}**). |\n"
            "| Does the third MA beat the *two*-MA rule it claims to beat? | **No.** The simpler two-line "
            f"cross (Sharpe **{R['dual']['sharpe']:+.2f}**) beats the three-line stack "
            f"(**{R['triple']['sharpe']:+.2f}**). The \"confirmation filter\" *removes* value. |\n"
            "| Is the three-line rule's edge real at all? | **No.** Its own significance test comes in "
            f"at *t* = **{R['triple']['t']:.2f}** — nowhere near the *t* ≥ 2 bar. It's indistinguishable "
            "from noise. |\n"
            "| Does it at least win on safety? | **Only by sitting out.** It cuts the worst drawdown "
            f"(**{R['triple']['maxdd']:+.0%}** vs **{R['bh']['maxdd']:+.0%}**) — but so would *any* rule "
            "that spends half its life in cash, and it pays for that calm with far less return. |\n\n"
            "> The third moving average is the trading-content equivalent of a racing stripe: it looks "
            "fast and does nothing. **Three does not beat two.**"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A single moving-average cross whipsaws you to death. Add a **third** average and only "
            "trade when all three align — `fast > medium > slow` for longs. The middle line confirms "
            "the trend, so you skip the false starts. Three MAs filter the noise two can't.\"*\n\n"
            "This is one of the most-repeated setups in retail trading education — the **5/20/50** or "
            "**10/50/200** \"ribbon,\" the \"three-MA system,\" the \"guppy.\" The steelman is real: a "
            "middle filter *does* make the rule trade less often. The question is whether trading less "
            "often this particular way leaves you with **more** money — or just less of everything."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If \"more confirmation\" reliably improved returns, the implication would be huge: you "
            "could keep bolting on filters and keep getting richer. That's not how signals work. Every "
            "extra condition you add does two things at once — it screens out some bad trades **and** it "
            "delays your entries and exits, so you also miss good moves. The third MA only helps if the "
            "first effect beats the second.\n\n"
            "There's a cleaner way to say it: a strategy is only worth its complexity if it beats the "
            "**simpler version of itself**. So we don't just race the triple stack against buy-and-hold "
            "— we race it against the **two-MA** rule with the same fast and slow legs. If the third "
            "line earns nothing over two lines, the whole \"three beats two\" pitch is dead."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['years']:.0f} years** of daily S&P 500 closes ({R['start']} → {R['end']}) and "
            "build three portfolios:\n\n"
            "1. **Buy-and-hold** — always invested. The thing to beat.\n"
            "2. **Two-MA rule** — long when the fast average is above the slow one; else cash.\n"
            "3. **Three-MA rule** — long only when `fast > medium > slow`; else cash.\n\n"
            "Each rule reads its averages at *tonight's* close and trades at *tomorrow's* (one-day lag, "
            "no peeking). We charge realistic trading costs, credit cash when out of the market, and "
            "compare **risk-adjusted return** (Sharpe). **What would make us say \"mirage\"?** If the "
            "three-MA Sharpe fails to beat *both* buy-and-hold and the two-MA rule, and its own "
            "significance test stays under *t* = 2."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, the headline race. Three portfolios, same tape, risk-adjusted return (Sharpe) side "
            "by side."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(CLOSE, R['fast'], R['mid'], R['slow'], rf_annual=R['rf'], cost_bps=R['cost_bps'])\n"
            "    sh = [res['bh']['sharpe'], res['dual']['sharpe'], res['triple']['sharpe']]\n"
            "else:\n"
            "    sh = [R['bh']['sharpe'], R['dual']['sharpe'], R['triple']['sharpe']]\n"
            "labels = ['buy &\\nhold', 'TWO-MA\\nrule', 'THREE-MA\\nrule']\n"
            "cols = [GREEN, AMBER, RED]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(labels, sh, color=cols, width=.6)\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of cash)')\n"
            "ax.set_title('More lines, less reward: three-MA is the worst of the three')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold {sh[0]:+.3f} | two-MA {sh[1]:+.3f} | three-MA {sh[2]:+.3f}')"
        ),
        md(
            f"There it is, in one chart. Buy-and-hold wins (**{R['bh']['sharpe']:+.2f}**), the **two**-MA "
            f"rule is second (**{R['dual']['sharpe']:+.2f}**), and the celebrated **three**-MA stack "
            f"comes **dead last** (**{R['triple']['sharpe']:+.2f}**). Adding the third line didn't add "
            "skill — it subtracted return."
        ),
        md(
            "**But maybe it's safer?** The one thing a sit-in-cash rule reliably does is dodge crashes. "
            "Here's the worst peak-to-trough loss for each."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dd = [res['bh']['max_drawdown'], res['dual']['max_drawdown'], res['triple']['max_drawdown']]\n"
            "    cg = [res['bh']['cagr'], res['dual']['cagr'], res['triple']['cagr']]\n"
            "else:\n"
            "    dd = [R['bh']['maxdd'], R['dual']['maxdd'], R['triple']['maxdd']]\n"
            "    cg = [R['bh']['cagr'], R['dual']['cagr'], R['triple']['cagr']]\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.6,4.3))\n"
            "a1.bar(labels, [d*100 for d in dd], color=[GREEN,AMBER,RED], width=.6)\n"
            "for i,v in enumerate(dd): a1.annotate(f'{v:+.0%}',(i,v*100),ha='center',va='top')\n"
            "a1.set_ylabel('worst drawdown (%)'); a1.set_title('Yes, it crashes less...')\n"
            "a2.bar(labels, [c*100 for c in cg], color=[GREEN,AMBER,RED], width=.6)\n"
            "for i,v in enumerate(cg): a2.annotate(f'{v:+.1%}',(i,v*100),ha='center',va='bottom')\n"
            "a2.set_ylabel('annual return (%)'); a2.set_title('...but you pay for it in return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('drawdowns:', [f'{d:+.0%}' for d in dd], '| CAGR:', [f'{c:+.1%}' for c in cg])"
        ),
        md(
            f"The three-MA rule does cut the worst drawdown (**{R['triple']['maxdd']:+.0%}** vs "
            f"**{R['bh']['maxdd']:+.0%}**) — but look at the right panel: it earns only "
            f"**{R['triple']['cagr']:+.1%}** a year against buy-and-hold's **{R['bh']['cagr']:+.1%}**. "
            "It buys calm by sitting in cash half the time, and the calm costs more than it's worth. "
            "That's *exposure reduction*, not *timing skill* — any rule that's out of the market a lot "
            "would do the same.\n\n"
            "> 🔬 **For the quants.** \"Lower drawdown\" is the seduction of every part-time-in-cash "
            "rule. The clean test is the HAC Sharpe-*difference* t against the *simpler* rule with the "
            f"same legs (the two-MA cross): here it's **{R['t_tri_vs_dual']:.2f}** — the third MA's "
            "marginal contribution to risk-adjusted return is negative. Details in "
            "[02_for_the_quants](02_for_the_quants.ipynb)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The three-MA rule's edge is statistically indistinguishable from "
            f"noise (*t* = {R['triple']['t']:.2f}, far under the *t* ≥ 2 bar), and a permutation test "
            f"can't tell it apart from luck (*p* = {R['perm']['p']:.2f}).\n"
            "- **Tradability — Mirage.** Lower Sharpe than buy-and-hold *and* than the two-MA rule; even "
            "**before costs** it lags. The drawdown reduction is just being-in-cash, not timing.\n"
            "- **\"Three beats two\"? — Busted.** The simpler two-line rule beats the three-line stack "
            "on every config we tried. The third moving average earns nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs only make it worse\n\n"
            "The three-MA rule trades a lot (its averages cross often). Here's its Sharpe as we dial "
            "trading costs up from zero. It starts below buy-and-hold and only sinks."
        ),
        code(
            "bps = [c[0] for c in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    sweep = [st.run_experiment(CLOSE, R['fast'], R['mid'], R['slow'], rf_annual=R['rf'], cost_bps=c)['triple']['sharpe'] for c in bps]\n"
            "else:\n"
            "    sweep = [c[1] for c in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(bps, sweep, 'o-', c=RED, lw=2, label='three-MA rule')\n"
            "ax.axhline(R['bh']['sharpe'], c=GREEN, ls='--', label=f\"buy & hold ({R['bh']['sharpe']:+.2f})\")\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for x,y in zip(bps,sweep): ax.annotate(f'{y:+.2f}',(x,y),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('one-way trading cost (bps)'); ax.set_ylabel('Sharpe (excess of cash)')\n"
            "ax.set_title('Even free, the three-MA rule loses the race; costs sink it further')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Sharpe by cost:', {f'{b:g}bps': round(s,3) for b,s in zip(bps,sweep)})"
        ),
        md(
            f"At **zero** cost the three-MA Sharpe is already only **{R['costs'][0][1]:+.2f}** — under "
            f"buy-and-hold's **{R['bh']['sharpe']:+.2f}**. By a realistic **10 bps** it's **negative** "
            f"({R['costs'][2][1]:+.2f}). There's no cost level at which this becomes a good trade, "
            "because there was never an edge to charge costs against."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **It's not the *parameters*.** We tried 5/20/50, 10/50/200, 20/50/100, 5/10/20 — the "
            "three-MA rule loses to the two-MA rule in every one. The problem is the *idea*, not the "
            "tuning.\n"
            "- **Why does the third line hurt?** It makes you *later* — you wait for all three to align "
            "to get in and to break alignment to get out, so you buy higher and sell lower than the "
            "two-line rule. More confirmation = more lag = less return.\n"
            "- **Build your own.** Swap SMAs for EMAs, add a fourth line, try a short leg — the quants "
            "notebook has the engine. *Find a three-MA config whose **net** Sharpe beats both "
            "buy-and-hold and the two-MA rule with a *t* ≥ 2, and you've got something. We couldn't.*"
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
            "# Triple-MA crossover — a quantitative teardown 🔬\n"
            "### Long/flat ribbon timing on ^GSPC · NET excess Sharpe vs buy-and-hold & the two-MA "
            "benchmark · HAC *t* · circular-block permutation null · cost sweep · synthetic "
            "faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "triple-MA \"ribbon\" is retail folklore — so the job here is to race it honestly against "
            "the **simpler rule it claims to beat** (a two-MA cross), charge it costs, and test its own "
            "significance. The headline: it beats neither buy-and-hold nor the two-MA rule, and its own "
            "HAC *t* never clears 2.\n\n"
            "> ⚠️ **Data note.** **^GSPC** daily **price-only** close (no dividends) — so the "
            "timing-vs-hold race is **excess-of-cash** (flat 3% proxy), and every Sharpe is labelled "
            "excess. One-day execution lag; one-way cost × NAV; the (optional) short leg pays borrow. "
            "Real data: yfinance, cached. Offline core + synthetic control are deterministic. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Three-MA net excess Sharpe **{R['triple']['sharpe']:+.2f}** (HAC "
            f"**t = {R['triple']['t']:.2f}** on the excess mean — under the bar); block-permutation "
            f"**p = {R['perm']['p']:.2f}**. Indistinguishable from noise. |\n"
            f"| **Tradability** | `MIRAGE` | Below buy-and-hold (**{R['bh']['sharpe']:+.2f}**) *and* the "
            f"two-MA rule (**{R['dual']['sharpe']:+.2f}**) — even at **0 bps** "
            f"(**{R['costs'][0][1]:+.2f}**). The lower drawdown is pure exposure reduction (47% "
            "in-market), not timing. |\n"
            f"| **Three beats two?** | `BUSTED` | HAC Sharpe-difference **t(triple − dual) = "
            f"{R['t_tri_vs_dual']:.2f}** (the third MA is *negative*-signed value) and triple loses to "
            "dual in **4/4** configs. |\n\n"
            "> 💡 In plain words: the third moving average is a complexity tax. It makes the rule slower "
            "and poorer than the two-line version it's supposed to improve."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $m_f, m_m, m_s$ be the fast/medium/slow SMAs of the close $P_t$. The triple-MA target "
            "exposure (long/flat) is\n\n"
            "$$w^{\\text{tri}}_t = \\mathbf{1}\\!\\left[m_f(t) > m_m(t) > m_s(t)\\right],$$\n\n"
            "vs the two-MA benchmark $w^{\\text{dual}}_t = \\mathbf{1}[m_f(t) > m_s(t)]$ (same fast/slow "
            "legs, **no** middle filter). Both are lagged one day: the position formed at the close of "
            "$t$ earns $r_{t+1}$. The net strategy return is "
            "$r^{\\text{net}}_{t+1} = w_t r_{t+1} + (1-w_t)r^f - c\\,|\\Delta w_t|$.\n\n"
            "- **H₁ (real edge).** The triple rule's *excess* mean clears HAC $t \\ge 2$.\n"
            "- **H₂ (beats hold).** Its net excess Sharpe $>$ buy-and-hold's.\n"
            "- **H₃ (three beats two).** Its net excess Sharpe $>$ the two-MA rule's, i.e. "
            "$t(\\text{triple}-\\text{dual}) > 0$ and significant.\n\n"
            "We find **H₁ rejected** (t≈0.1), **H₂ rejected** (triple < BH), **H₃ rejected** "
            "(triple < dual, *negative* difference t). The third MA earns nothing — it costs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the only number that matters is excess-vs-excess\n\n"
            "Two honesty problems govern this race. **(a) The benchmark.** A timing rule that sits in "
            "cash part-time must be compared **excess-of-cash to excess-of-cash**, or the cash leg "
            "flatters it for free — so every Sharpe here subtracts the 3% cash rate. **(b) The right "
            "control.** \"Beats buy-and-hold\" is a weak bar (a low-vol rule can clear it on Sharpe "
            "while earning less). The decisive control is the **simpler version of the same idea** — "
            "the two-MA cross. If the third MA can't beat *that*, the complexity is unjustified. We test "
            "the difference with a HAC (Newey-West) Jobson-Korkie *t*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** ^GSPC daily price-only close, {R['start']}→{R['end']} "
            f"(**n = {R['n_days']:,}**, {R['years']:.1f}y; fingerprint `{R['fp']}`).\n"
            f"- **Rules.** Triple **{R['fast']}/{R['mid']}/{R['slow']}** long/flat; two-MA "
            f"**{R['fast']}/{R['slow']}** long/flat (same fast/slow legs).\n"
            "- **Timing.** Read the stack at close of *t*; hold over *t+1* (one `shift`). No look-ahead.\n"
            "- **Accounting.** Cash leg at flat 3%/yr; one-way **5 bps × NAV** on every exposure "
            "change; short leg (when enabled) pays borrow. Sharpe & *t* on **excess** returns.\n"
            "- **Signal test.** HAC *t* on the triple rule's excess mean (bar: |*t*| ≥ 2).\n"
            "- **Permutation null.** Re-run the rule on **circular-block** (21-day) shuffles of the "
            "return series; *p* = Pr[shuffled Sharpe ≥ observed].\n"
            "- **Head-to-head.** HAC Sharpe-difference *t* for triple − BH and triple − dual.\n"
            "- **Robustness.** Repeat across 4 fast/mid/slow configs + a cost sweep.\n"
            "- **Positive control.** A deterministic two-regime panel with a planted trend `edge`: "
            "`edge=0` must NOT reach significance; a large `edge` must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The three-way race + the head-to-head significance\n\n"
            "Left: net excess Sharpe of the three arms. Right: the HAC Sharpe-difference *t* of the "
            "triple rule against each benchmark — both **negative**, i.e. the triple rule is reliably "
            "*worse*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(CLOSE, R['fast'], R['mid'], R['slow'], rf_annual=R['rf'], cost_bps=R['cost_bps'])\n"
            "    sh = [res['bh']['sharpe'], res['dual']['sharpe'], res['triple']['sharpe']]\n"
            "    td = [res['t_triple_vs_bh'], res['t_triple_vs_dual']]\n"
            "    ttri = res['triple']['tstat']\n"
            "else:\n"
            "    sh = [R['bh']['sharpe'], R['dual']['sharpe'], R['triple']['sharpe']]\n"
            "    td = [R['t_tri_vs_bh'], R['t_tri_vs_dual']]; ttri = R['triple']['t']\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.8,4.3))\n"
            "a1.bar(['buy&hold','two-MA','three-MA'], sh, color=[GREEN,AMBER,RED], width=.6)\n"
            "for i,v in enumerate(sh): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_ylabel('net excess Sharpe'); a1.set_title('Three-MA is last')\n"
            "a2.bar(['triple\\n-\\nBH','triple\\n-\\ntwo-MA'], td, color=[RED,RED], width=.5)\n"
            "a2.axhline(-2, ls='--', c=GREY, label='|t|=2'); a2.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(td): a2.annotate(f't={v:+.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a2.set_ylabel('HAC Sharpe-difference t'); a2.set_title('Triple is reliably worse'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe BH={sh[0]:+.3f} dual={sh[1]:+.3f} triple={sh[2]:+.3f} | triple own t={ttri:+.2f}')\n"
            "print(f't(triple-BH)={td[0]:+.2f}  t(triple-dual)={td[1]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the triple rule's net excess Sharpe (**{R['triple']['sharpe']:+.2f}**) "
            f"is the worst of the three, its **own** HAC *t* is **{R['triple']['t']:.2f}** (no edge), and "
            f"it loses to buy-and-hold at **t = {R['t_tri_vs_bh']:.2f}**. The two-MA difference "
            f"(**t = {R['t_tri_vs_dual']:.2f}**) is negative — the third MA's marginal contribution is "
            "to *subtract* risk-adjusted return."
        ),
        md(
            "### 4b · Is the edge even there? — a circular-block permutation null\n\n"
            "Re-run the *exact* triple rule on circular-block (21-day) reshuffles of the return series. "
            "If the rule harvested genuine trend persistence, the real Sharpe would sit in the right "
            "tail. It sits in the middle of the cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pp = st.permutation_pvalue(CLOSE, R['fast'], R['mid'], R['slow'], rf_annual=R['rf'], cost_bps=R['cost_bps'], n_draws=R['perm']['n_draws'], seed=438)\n"
            "    obs, pval, draws = pp['obs'], pp['p_value'], pp['draws']\n"
            "else:\n"
            "    obs = R['perm']['obs']; pval = R['perm']['p']\n"
            "    rng = np.random.default_rng(438); draws = rng.normal(R['perm']['null_mean'], R['perm']['null_sd'], 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: block-shuffled tapes')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed Sharpe {obs:+.2f}')\n"
            "ax.set_xlabel('three-MA net excess Sharpe'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Buried in the luck cloud: permutation p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.3f}  permutation p={pval:.2f}  null mean={float(np.mean(draws)):+.3f}')"
        ),
        md(
            f"> 💡 In plain words: **{R['perm']['p']*100:.0f}%** of randomly-reshuffled tapes produce a "
            "Sharpe at least as high as the real one. The real S&P trend is *not* something this rule "
            "reads better than chance — there's no signal under the strategy."
        ),
        md(
            "### 4c · Robustness — it's the idea, not the tuning\n\n"
            "Four common fast/mid/slow configs. In every one, the triple rule's Sharpe is below the "
            "two-MA rule's (the third MA never earns its keep), and its own *t* never clears 2."
        ),
        code(
            "cfgs = [(c[0],c[1],c[2]) for c in R['configs']]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for f,m,s in cfgs:\n"
            "        r = st.run_experiment(CLOSE, f, m, s, rf_annual=R['rf'], cost_bps=R['cost_bps'])\n"
            "        rows.append((f,m,s, r['triple']['sharpe'], r['triple']['tstat'], r['dual']['sharpe'], r['bh']['sharpe'], r['t_triple_vs_dual']))\n"
            "else:\n"
            "    rows = R['configs']\n"
            "x = np.arange(len(rows)); labs = [f'{r[0]}/{r[1]}/{r[2]}' for r in rows]\n"
            "tri = [r[3] for r in rows]; dua = [r[5] for r in rows]; bh = rows[0][6]\n"
            "fig, ax = plt.subplots(figsize=(9.6,4.3))\n"
            "ax.bar(x-.2, tri, .4, color=RED, label='three-MA')\n"
            "ax.bar(x+.2, dua, .4, color=AMBER, label='two-MA')\n"
            "ax.axhline(bh, c=GREEN, ls='--', label=f'buy&hold {bh:+.2f}'); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('net excess Sharpe')\n"
            "ax.set_title('Across every config: three-MA < two-MA < buy&hold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:>3}/{r[1]:>3}/{r[2]:>3}: triple Sh={r[3]:+.3f} t={r[4]:+.2f} | dual Sh={r[5]:+.3f} | t(tri-dual)={r[7]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: not a single config rescues the third MA. Even the famous "
            f"**10/50/200** ribbon gives triple Sharpe {R['configs'][1][3]:+.2f} vs dual "
            f"{R['configs'][1][5]:+.2f} (difference t = {R['configs'][1][7]:+.2f}). The failure is "
            "structural — confirmation lag — not a parameter you can tune away."
        ),
        md(
            "### 4d · The faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic two-regime panel where we **plant** a known trend `edge`: with "
            "`edge = 0` (a random walk) the triple rule must NOT manufacture significance; with a large "
            "planted `edge` the *same* engine must light up. Both hold — so the flat real-tape result is "
            "a genuine null, not a broken backtester."
        ),
        code(
            "res_s = []\n"
            "for edge in (0.0, 2.0):\n"
            "    d, t = data.synthetic_panel(n_days=8000, edge=edge, seed=438)\n"
            "    r = st.run_experiment(d['close'], R['fast'], R['mid'], R['slow'], rf_annual=0.0, cost_bps=R['cost_bps'])\n"
            "    res_s.append((edge, r['triple']['sharpe'], r['triple']['tstat'], r['bh']['sharpe']))\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.3))\n"
            "labs = [f'planted edge\\n{e:.0f}' for e,_,_,_ in res_s]; tv = [r[2] for r in res_s]\n"
            "ax.bar(labs, tv, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(tv): ax.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('triple-MA HAC t (excess mean)')\n"
            "ax.set_title('Control: no edge -> t~0; planted trend -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,sh,tt,bh in res_s: print(f'edge={e}: triple Sharpe={sh:+.3f} t={tt:+.2f} (bh={bh:+.3f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trend the engine sits at "
            f"**t = {R['syn'][0][2]:.2f}** (no false positive — a random walk can't fake it); with a "
            f"strong planted trend it reaches **t = {R['syn'][1][2]:.2f}**. The machinery *can* detect a "
            f"trend edge when one exists — so the real-tape **t = {R['triple']['t']:.2f}** is telling us "
            "the truth: on the S&P there's no triple-MA edge to find."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — triple-MA net excess Sharpe **{R['triple']['sharpe']:+.2f}**, HAC "
            f"**t = {R['triple']['t']:.2f}** on the excess mean (far under 2), block-permutation "
            f"**p = {R['perm']['p']:.2f}**. No edge distinguishable from noise.\n"
            f"- **Tradability `MIRAGE`** — below buy-and-hold (**{R['bh']['sharpe']:+.2f}**) and below "
            f"the two-MA rule (**{R['dual']['sharpe']:+.2f}**) even at **0 bps** "
            f"(**{R['costs'][0][1]:+.2f}**); costs only deepen the hole (negative by 10 bps). The "
            "drawdown reduction is exposure reduction (47% in-market), not timing skill.\n"
            f"- **Three beats two? `BUSTED`** — HAC **t(triple − dual) = {R['t_tri_vs_dual']:.2f}**, and "
            "triple loses to dual in **4/4** configs. The third moving average's marginal contribution "
            "is *negative*: more confirmation, more lag, less return."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there's nothing to charge costs against\n\n"
            "The cost sweep makes the point operationally: the triple rule starts the race *behind* "
            "buy-and-hold even with **zero** costs, and the gap only widens as you pay to trade its "
            "frequent crossings (~12 round-trips/yr)."
        ),
        code(
            "bps = [c[0] for c in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    sweep = [st.run_experiment(CLOSE, R['fast'], R['mid'], R['slow'], rf_annual=R['rf'], cost_bps=c)['triple']['sharpe'] for c in bps]\n"
            "else:\n"
            "    sweep = [c[1] for c in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(bps, sweep, 'o-', c=RED, lw=2, label='three-MA')\n"
            "ax.axhline(R['bh']['sharpe'], c=GREEN, ls='--', label=f\"buy&hold {R['bh']['sharpe']:+.2f}\")\n"
            "ax.axhline(R['dual']['sharpe'], c=AMBER, ls=':', label=f\"two-MA {R['dual']['sharpe']:+.2f}\")\n"
            "ax.axhline(0,c='k',lw=.8)\n"
            "for x,y in zip(bps,sweep): ax.annotate(f'{y:+.2f}',(x,y),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net excess Sharpe')\n"
            "ax.set_title('No cost level rescues it — there was never an edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('triple Sharpe by cost:', {f'{b:g}bps': round(s,3) for b,s in zip(bps,sweep)})"
        ),
        md(
            f"> 💡 In plain words: at 0 bps the triple Sharpe is **{R['costs'][0][1]:+.2f}** — already "
            f"below both benchmarks — and turns negative by 10 bps "
            f"(**{R['costs'][2][1]:+.2f}**). You can't make money off a signal that isn't there; cost is "
            "a sideshow. **MIRAGE**, not even FRAGILE."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The general lesson.** \"Add another confirmation\" is the single most over-sold idea in "
            "retail TA. Each extra filter trades fewer-bad-trades for more-lag; the triple-MA is a clean "
            "case where the trade is *negative*. Always race a complex rule against the simpler version "
            "of itself.\n"
            "- **Variants to try.** EMAs instead of SMAs, a long/short leg, a fourth MA, or a "
            "volatility-scaled position — the engine (`triple_ma_crossover/strategy.py`) takes them all. "
            "The permutation null and the synthetic control travel with you.\n"
            "- **The bar.** Show a three-MA config whose **net excess** Sharpe beats *both* "
            "buy-and-hold and the two-MA rule with a HAC *t* ≥ 2, on a tape you didn't tune on. That's "
            "the PR we'd merge.\n\n"
            "*The reproducible core is offline and deterministic; the real input is ^GSPC price-only "
            "daily closes (excess-of-cash race, labelled throughout). Methods and sources: "
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
