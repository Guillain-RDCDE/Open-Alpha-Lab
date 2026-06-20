"""Generate the two narrative notebooks for Study 317 (Fed-Balance-Sheet).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells try the shared
``quantlab`` SPY parquet cache and otherwise fall back to the synthetic tape under a clear
banner. The frozen real-tape headline numbers live in ``R`` (mirroring docs/results.md), so
the prose re-runs identically for any reader regardless of which tape the cell drew.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12,
# SPY split-only 1993-01-29 → 2026-05-29, fingerprint d937d577b4c5).
R = dict(
    asof="2026-06-12", win_start="1993-01-29", win_end="2026-05-29",
    fp="d937d577b4c5", n_days=8390,
    n_qe=1536, n_qt=1462, n_flat=5392,
    qe_mean=8.35, qe_t=3.09, qe_sharpe=1.08,
    qt_mean=5.81, qt_t=2.35, qt_sharpe=0.91,
    flat_mean=2.40, flat_t=1.75, flat_sharpe=0.32,
    qe_qt_diff=2.54, qe_qt_lo=-4.10, qe_qt_hi=9.15, qe_qt_p=0.47,
    qe_flat_diff=5.95, qe_flat_lo=0.36, qe_flat_hi=11.35, qe_flat_p=0.038,
    bh_ann=10.83, bh_sharpe=0.552,
    cash_ann=8.05, cash_sharpe=0.446, cash_t=3.03, cash_gap=-2.78, cash_dsharpe=-0.11,
    short_ann=5.33, short_sharpe=0.279, short_t=1.86, short_gap=-5.50, short_dsharpe=-0.27,
    pct_in_market=82.6,
    # synthetic control row (machinery proof only)
    ctrl_qe_t=2.88, ctrl_qt_t=-2.98, ctrl_diff=23.27, ctrl_lo=12.29, ctrl_hi=34.04,
    null_diff=3.28, null_lo=-7.71, null_hi=14.05,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"
REG_COL = {"QE": GREEN, "QT": RED, "FLAT": GREY}

from fed_balance_sheet import data, strategy as st

# Cache-first real tape; fall back to the synthetic control if offline. The banner
# variable TAPE records which tape every cell below is actually showing.
def load_tape():
    try:
        bars = data.load_real("SPY", fetch=False)
        bars = bars.loc[bars.index < pd.Timestamp("2026-06-01")]  # drop partial month
        return bars, "REAL"
    except Exception as e:
        frame, _ = data.synthetic_daily(qe_edge=0.0, seed=317)   # NULL synthetic fallback
        return frame, "SYNTHETIC-NULL"

bars, TAPE = load_tape()
print("tape in use:", TAPE, "| rows:", len(bars), "| span:",
      bars.index.min().date(), "->", bars.index.max().date())
if TAPE == "REAL":
    print("fingerprint:", data.fingerprint(bars))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Fed-Balance-Sheet — does \"Don't fight the Fed\" actually work? 🏦\n"
            "### When the Fed prints, do stocks rip — and when it tightens, do they sink?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Don't_fight_the_Fed%3F: Busted](https://img.shields.io/badge/Don't_fight_the_Fed%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every macro feed has the chart: the Fed's balance sheet overlaid on the S&P 500, "
            "rising together. The story writes itself — **quantitative easing (QE) pumps stocks; "
            "quantitative tightening (QT) drains them.** *Don't fight the Fed.* This notebook asks, "
            "in plain English, the only question that matters: over 33 years, did stocks really do "
            "**better in QE than in QT** — by enough to act on?\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap CI and the "
            "excess-Sharpe race? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did stocks rise in QE? | **Yes** — about +8.3 bps/day. |\n"
            "| Did stocks rise in QT? | **Also yes** — about +5.8 bps/day. |\n"
            "| Is QE *meaningfully better* than QT? | **No.** The gap is ~+2.5 bps/day, and the "
            "honest confidence interval runs from negative to positive — it's noise. |\n"
            "| Would 'fight the Fed in QT' have paid? | **No** — it underperformed just holding "
            "the index, by a wide margin. |\n\n"
            f"> The headline numbers below are the **real SPY tape** (as-of {R['asof']}, "
            f"split-only, fingerprint `{R['fp']}`). If a cell ran offline it falls back to a "
            "synthetic tape and says so in its banner — but the *prose* numbers are always the "
            "frozen real run from [docs/results.md](../docs/results.md)."
        ),

        md(
            "## 1 · The claim\n\n"
            "\"**Don't fight the Fed.**\" In the QE era the slogan became a balance-sheet rule: the "
            "Fed creates money to buy bonds (QE), that liquidity has to go *somewhere*, and it ends "
            "up in stocks — so you ride QE and step aside (or short) during QT, when the Fed is "
            "pulling that liquidity back out. The overlay chart of 'Fed assets vs the S&P' is the "
            "single most-shared picture in macro commentary."
        ),
        code(
            "# What the Fed was doing, decade by decade — our announced-direction regime table.\n"
            "# (FRED's WALCL dollar series is network-blocked here; we encode the *direction*,\n"
            "#  which is exactly what a 'don't fight the Fed' trader claims to act on.)\n"
            "w = data.regime_windows()\n"
            "for _, row in w.iterrows():\n"
            "    print(f\"{row.start.date()} -> {row.end.date():}  {row.regime}\")\n"
        ),

        md(
            "## 2 · So what?\n\n"
            "If it were true it would be the easiest macro trade alive: read the Fed's press "
            "release, go long when it eases, go to cash when it tightens, and beat the market with "
            "almost no effort. It would also mean the stock market is, at bottom, a liquidity pump "
            "— that *what the Fed does to its balance sheet* matters more than earnings, valuations "
            "or growth. Those are big claims. Let's see if the tape backs them."
        ),

        md(
            "## 3 · How would we even know?\n\n"
            "Dead simple: tag every trading day with the Fed's regime that day (QE, QT, or neither), "
            "then compare the **average daily return** in each bucket. The slogan lives or dies on "
            "one number — **is QE's average clearly bigger than QT's?** If the two are about the "
            "same, the chart that started all this was an illusion. We'd call it a **mirage**."
        ),
        code(
            "# Average daily return by regime, in basis points (1 bp = 0.01%).\n"
            "rm = st.regime_means(bars)\n"
            "print(f\"[tape: {TAPE}]\")\n"
            "ax = rm['mean_bps'].plot(kind='bar', color=[REG_COL[i] for i in rm.index],\n"
            "                         edgecolor='black')\n"
            "ax.axhline(0, color='black', lw=.8)\n"
            "ax.set_ylabel('mean daily return (bps)'); ax.set_xlabel('Fed regime')\n"
            "ax.set_title(f'Daily SPY return by Fed balance-sheet regime  [{TAPE} tape]')\n"
            "plt.xticks(rotation=0); plt.tight_layout(); plt.show()\n"
            "rm.round(2)\n"
        ),

        md(
            "## 4 · The teardown — let's actually look\n\n"
            f"On the real tape the buckets come out: **QE +{R['qe_mean']} bps/day**, "
            f"**QT +{R['qt_mean']} bps/day**, FLAT +{R['flat_mean']} bps/day. Read that twice. "
            "Stocks went up in QE — *and they went up in QT too*. The thing the slogan says should "
            "have been a headwind (the Fed shrinking its balance sheet) coincided with a perfectly "
            "healthy bull market.\n\n"
            f"The gap that the whole idea rests on — **QE minus QT** — is just "
            f"**+{R['qe_qt_diff']} bps/day**, and once you put an honest error bar on it the range "
            f"runs from **{R['qe_qt_lo']}** to **+{R['qe_qt_hi']}** bps. It straddles zero. "
            "Translation: we cannot tell QE and QT apart."
        ),
        code(
            "# The decisive contrast: QE minus QT, with an honest bootstrap error bar.\n"
            "ci = st.diff_block_bootstrap_ci(bars, a='QE', b='QT', n_boot=2000, seed=317)\n"
            "print(f\"[tape: {TAPE}]\")\n"
            "print(f\"QE - QT  = {ci['diff_bps']:+.2f} bps/day\")\n"
            "print(f\"95% CI   = [{ci['ci_lo']:+.2f}, {ci['ci_hi']:+.2f}]   (straddling 0 = noise)\")\n"
            "print(f\"bootstrap p = {ci['p_boot']:.3f}\")\n"
        ),

        md(
            "## 5 · The verdict\n\n"
            "- **Signal — NONE.** The QE-vs-QT gap is indistinguishable from zero. Stocks rose in "
            "both regimes.\n"
            "- **Tradability — MIRAGE.** A rule that goes to cash (or short) in QT *underperforms* "
            "just buying and holding — see below.\n"
            "- **'Don't fight the Fed'? — BUSTED.** Fighting the Fed when it tightened would have "
            "cost you money."
        ),

        md(
            "## 6 · Could you actually trade it?\n\n"
            "Take the slogan literally: long stocks in QE and the quiet periods, **cash in QT** (or "
            "the braver version, **short in QT**). Race it against simply buying and holding SPY the "
            "whole time."
        ),
        code(
            "# The literal 'fight the Fed in QT' rule vs buy-and-hold.\n"
            "import numpy as np\n"
            "rows = []\n"
            "for label, short in [('cash in QT', False), ('short in QT', True)]:\n"
            "    r = st.race(bars, short_qt=short, cost_bps=5.0)\n"
            "    rows.append({'rule': label,\n"
            "                 'ann return %': round(r['timing_net']['ann_ret']*100, 2),\n"
            "                 'Sharpe': round(r['timing_net']['sharpe'], 3)})\n"
            "bh = st.race(bars, short_qt=False, cost_bps=5.0)['buy_hold']\n"
            "rows.append({'rule': 'buy & hold', 'ann return %': round(bh['ann_ret']*100, 2),\n"
            "             'Sharpe': round(bh['sharpe'], 3)})\n"
            "print(f\"[tape: {TAPE}]\")\n"
            "pd.DataFrame(rows).set_index('rule')\n"
        ),
        md(
            f"On the real tape buy-and-hold returns **+{R['bh_ann']}%/yr** at Sharpe "
            f"{R['bh_sharpe']}. Sitting in cash during QT drops you to **+{R['cash_ann']}%/yr**; "
            f"*shorting* during QT — literally fighting the Fed — nearly halves you to "
            f"**+{R['short_ann']}%/yr**. The rule doesn't just fail to add value; it actively "
            "destroys it. (Costs barely matter — the rule trades a fraction of a time per year.)"
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "So why does the overlay chart look so convincing? Because the Fed **eases hardest into "
            "crashes** — QE1 launched in the 2008 wreckage, COVID QE in the March-2020 crater — and "
            "stocks then rebound off the bottom. The eye reads 'QE → stocks up'; the truth is "
            "'crash → Fed eases *and* stocks bounce'. That's the only contrast with any signal here "
            "(QE beats the *quiet* periods, barely), and it's reverse causation, not a tradable "
            "edge.\n\n"
            "Forks worth a PR: swap our announced-direction table for the actual weekly `WALCL` "
            "dollar change once FRED is reachable; test the *net liquidity* version (WALCL − TGA − "
            "reverse repo) that the liquidity crowd actually quotes; or condition on the **rate of "
            "change** of the balance sheet rather than its sign."
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
            "# Fed-Balance-Sheet — a quantitative teardown 🔬\n"
            "### Regime conditioning · HAC inference · block-bootstrap CI · excess-Sharpe race · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Don't_fight_the_Fed%3F: Busted](https://img.shields.io/badge/Don't_fight_the_Fed%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We sort SPY daily "
            "returns by the Fed's announced balance-sheet direction and test the one contrast the "
            "slogan rests on (QE − QT) with an autocorrelation-robust *t* and a circular-block "
            "bootstrap, then race the literal timing rule on an excess-of-cash Sharpe.\n\n"
            "> ⚠️ **Not investment advice.** Real tape: SPY daily, split-only / **price-only** (not "
            "total return), via the shared `quantlab` cache; regimes from a hand-built table "
            "standing in for the network-blocked FRED `WALCL`. Methods & sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition. House "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | The decisive number |\n|---|---|---|\n"
            f"| **Signal** | **NONE** | QE − QT = +{R['qe_qt_diff']} bps/day, 95% CI "
            f"[{R['qe_qt_lo']}, +{R['qe_qt_hi']}], *p* = {R['qe_qt_p']}; QT itself is *up* "
            f"(*t* = +{R['qt_t']}). |\n"
            f"| **Tradability** | **MIRAGE** | Cash-in-QT {R['cash_gap']}%/yr vs buy-and-hold; "
            f"short-in-QT {R['short_gap']}%/yr (Sharpe {R['short_sharpe']} vs {R['bh_sharpe']}). |\n"
            f"| **\"Don't fight the Fed\"?** | **BUSTED** | Fighting the Fed in QT cost money over "
            "33 years; the only live contrast is QE − FLAT (reverse causation: the Fed eases into "
            "crashes that rebound). |\n\n"
            "> 💡 **In plain words.** The chart everyone shares is real — stocks *did* rise during "
            "QE. The trick is that they also rose during QT, so the *difference* (the only thing you "
            "could trade) is noise."
        ),

        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (the slogan):** mean daily SPY return is higher in QE than in QT, by a margin "
            "large enough to clear an autocorrelation-robust *t* on the **difference**.\n"
            "- **H₂ (tradable):** a rule that is long in QE/FLAT and cash-or-short in QT earns a "
            "*higher excess-of-cash Sharpe* than buy-and-hold, net of costs and one execution lag.\n"
            "- **H₃ (causal direction):** any QE-vs-other gap reflects QE *driving* equities, not "
            "the Fed easing endogenously into drawdowns that mechanically rebound (the 'Fed put')."
        ),

        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁ holds, the equity premium is partly a *policy* premium you can time off a public "
            "calendar — an enormous claim about market efficiency. If only H₃'s reverse-causation "
            "story holds, the overlay chart is a survivorship-of-attention artefact and the slogan "
            "is folklore. The desk has already found the *genuine* Fed calendar effects elsewhere "
            "(pre-FOMC drift, Study 67) — this study isolates the **balance-sheet-direction** claim "
            "specifically."
        ),

        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Condition** daily returns on the announced regime (no fitting).\n"
            "2. **Robust inference** — Newey–West HAC *t* on each regime mean and a circular-block "
            "bootstrap CI on the QE − QT contrast (block = 21 ≈ one trading month, preserving vol "
            "clustering).\n"
            "3. **Critique** — check whether any gap is QE − QT (the slogan) or merely QE − FLAT "
            "(reverse causation).\n"
            "4. **Trade it** — long-QE / cash-or-short-QT, **one** execution lag, one-way costs × "
            "NAV, raced on **excess-of-cash** Sharpe (the rule sits in cash part-time).\n"
            "5. **Synthetic control** — confirm the harness banks a *planted* QE>QT edge and finds "
            "nothing on the null.\n\n"
            "> 🔬 **Falsifier, declared up front:** if the QE − QT bootstrap CI straddles zero, we "
            "stamp **NONE** — no goalpost-moving afterward."
        ),
        code(
            "# Regime means with HAC t-stats.\n"
            "rm = st.regime_means(bars)\n"
            "print(f\"[tape: {TAPE}]  H0 for each tstat: mean daily return = 0\")\n"
            "rm.round(3)\n"
        ),
        md(
            "> 💡 **In plain words.** Every regime — including QT — has a positive, often "
            "*significant* up-drift. That alone sinks the slogan: 'fight the Fed in QT' would have "
            "you flat or short during a rising market."
        ),

        md("## 4 · The teardown\n\n### 4.1 The decisive contrast — QE − QT (block bootstrap)"),
        code(
            "ci_qt = st.diff_block_bootstrap_ci(bars, a='QE', b='QT', n_boot=3000, seed=317)\n"
            "ci_fl = st.diff_block_bootstrap_ci(bars, a='QE', b='FLAT', n_boot=3000, seed=317)\n"
            "print(f\"[tape: {TAPE}]\")\n"
            "for name, ci in [('QE - QT  ', ci_qt), ('QE - FLAT', ci_fl)]:\n"
            "    print(f\"{name}: {ci['diff_bps']:+6.2f} bps  CI [{ci['ci_lo']:+.2f}, \"\n"
            "          f\"{ci['ci_hi']:+.2f}]  p={ci['p_boot']:.3f}\")\n"
        ),
        md(
            f"> 💡 **In plain words.** **QE − QT** = +{R['qe_qt_diff']} bps, CI "
            f"[{R['qe_qt_lo']}, +{R['qe_qt_hi']}], *p* = {R['qe_qt_p']} — the slogan's contrast is "
            f"noise. The only flicker is **QE − FLAT** (+{R['qe_flat_diff']} bps, *p* = "
            f"{R['qe_flat_p']}): the Fed eases hardest into crashes, and stocks rebound off the "
            "bottom — H₃'s reverse causation, not a tradable QE/QT edge."
        ),

        md("### 4.2 Could you trade it? — the excess-of-cash race"),
        code(
            "# Both legs netted of the cash rate before Sharpe (the rule is in cash part-time).\n"
            "import numpy as np\n"
            "out = []\n"
            "for label, short in [('cash in QT', False), ('short in QT', True)]:\n"
            "    for c in (0.0, 5.0):\n"
            "        r = st.race(bars, short_qt=short, cost_bps=c)\n"
            "        out.append({'rule': label, 'cost_bps': c,\n"
            "                    'ann %': round(r['timing_net']['ann_ret']*100, 2),\n"
            "                    'Sharpe': round(r['timing_net']['sharpe'], 3),\n"
            "                    'HAC t': round(r['timing_net']['tstat'], 2),\n"
            "                    'd Sharpe vs B&H': round(r['excess_sharpe_diff'], 3),\n"
            "                    '% in mkt': round(r['pct_in_market']*100, 1)})\n"
            "bh = st.race(bars, short_qt=False, cost_bps=5.0)['buy_hold']\n"
            "print(f\"[tape: {TAPE}]  buy&hold: ann {bh['ann_ret']*100:.2f}%  Sharpe {bh['sharpe']:.3f}\")\n"
            "pd.DataFrame(out)\n"
        ),
        md(
            f"> 💡 **In plain words.** The timing book's own return is positive (it is mostly long "
            f"SPY), but it **loses the race**: cash-in-QT {R['cash_gap']}%/yr and ΔSharpe "
            f"{R['cash_dsharpe']}; short-in-QT {R['short_gap']}%/yr and ΔSharpe "
            f"{R['short_dsharpe']}. Turnover is a fraction of a trade per year, so costs are "
            "irrelevant — the rule fails on the simplest test: it underperforms doing nothing."
        ),

        md("### 4.3 Equity curves — the rule never pulls ahead"),
        code(
            "book = st.regime_timing_returns(bars, short_qt=True, cost_bps=5.0)\n"
            "eq_bh = (1 + book['ret']).cumprod()\n"
            "eq_rule = (1 + book['timing_net']).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq_bh.index, eq_bh, color=GREY, label='buy & hold SPY')\n"
            "ax.plot(eq_rule.index, eq_rule, color=RED, label='fight the Fed (short in QT)')\n"
            "# shade QT windows\n"
            "qt = bars['regime'] == 'QT'\n"
            "ax.fill_between(bars.index, 0, eq_bh.max(), where=qt.reindex(bars.index).fillna(False),\n"
            "                color=RED, alpha=.08, label='QT regime')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'Fighting the Fed vs holding the index  [{TAPE} tape]'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
        ),

        md(
            "## 5 · The verdict\n\n"
            "| Axis | Stamp | Decisive statistics |\n|---|---|---|\n"
            f"| Signal | **NONE** | QE − QT +{R['qe_qt_diff']} bps, CI [{R['qe_qt_lo']}, "
            f"+{R['qe_qt_hi']}], *p* = {R['qe_qt_p']}; QT mean *t* = +{R['qt_t']} (up, not down). |\n"
            f"| Tradability | **MIRAGE** | Net ann.: cash {R['cash_ann']}%, short {R['short_ann']}% "
            f"vs B&H {R['bh_ann']}%; ΔSharpe {R['short_dsharpe']}. |\n"
            f"| \"Don't fight the Fed\"? | **BUSTED** | Only QE − FLAT is significant — reverse "
            "causation, not a tradable signal. |\n\n"
            "**REAL on the Signal axis was never on the table:** the contrast the claim depends on "
            "fails the inference bar outright (CI straddles zero). Literature on QE→*yields* is "
            "strong; QE→*equity-regime-drift* is not, and this tape can't certify it."
        ),

        md(
            "## 6 · Could you trade it?\n\n"
            "There is no edge to cost out — break-even cost is undefined when the gross signal is "
            "already a loss against the benchmark. The honest 'capacity' statement is the opposite "
            "of usual: the rule has *plenty* of capacity (it trades a fraction of a time per year "
            "and is mostly in liquid SPY) and that capacity is worthless because the underlying "
            "contrast is noise. The real cost is **opportunity cost** — every day spent in cash or "
            "short during a rising QT market."
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Real `WALCL` once FRED is reachable.** Replace the announced-direction table with "
            "the weekly dollar level and its *rate of change* — test the continuous version, not "
            "just the sign.\n"
            "- **Net liquidity.** The liquidity crowd quotes WALCL − Treasury General Account − "
            "reverse-repo balances, not gross assets; that series moves differently and deserves "
            "its own teardown.\n"
            "- **Endogeneity head-on.** Instrument the balance sheet on pre-scheduled programme "
            "announcements to separate 'QE causes rallies' from 'the Fed eases into rallies'.\n\n"
            "Below: the **synthetic positive control** — proof the harness *can* find a QE/QT edge "
            "when one is planted, so the NONE above is a real absence, not a dead pipeline."
        ),
        code(
            "# Machinery proof (synthetic only — NEVER cited in support of a real stamp).\n"
            "rows = []\n"
            "for edge, name in [(0.0, 'null'), (0.001, 'control (planted QE>QT)')]:\n"
            "    f, _ = data.synthetic_daily(qe_edge=edge, seed=317)\n"
            "    rmn = st.regime_means(f)\n"
            "    ci = st.diff_block_bootstrap_ci(f, n_boot=2000, seed=1)\n"
            "    rows.append({'tape': name, 'QE t': round(rmn.loc['QE','tstat'],2),\n"
            "                 'QT t': round(rmn.loc['QT','tstat'],2),\n"
            "                 'QE-QT bps': round(ci['diff_bps'],2),\n"
            "                 'CI': f\"[{ci['ci_lo']:+.1f}, {ci['ci_hi']:+.1f}]\",\n"
            "                 'finds signal?': 'no' if ci['ci_lo']<0<ci['ci_hi'] else 'YES'})\n"
            "print('SYNTHETIC machinery proof — not market evidence')\n"
            "pd.DataFrame(rows).set_index('tape')\n"
        ),
        md(
            "> 💡 **In plain words.** Plant a real QE>QT drift and the harness flags it (CI excludes "
            "zero); plant nothing and it stays silent. So when the *real* tape comes back NONE, that "
            "is the market telling us there's no edge — not the test failing to look."
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
