"""Generate the two narrative notebooks for Study 936 (Tolerance Bands).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the
fast synthetic control, which is clearly labelled as synthetic and never sits under a
real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. SPY/IEF/GLD/BIL total return,
# 2007-05-30 -> 2026-06-30, one execution lag, 5 bps one-way, excess-of-cash.
R = dict(
    # `fp` is the RETURNS fingerprint (the reproducibility key). `fp_levels` is the
    # adjusted-close fingerprint, which churns on every refetch because auto_adjust
    # rescales the whole level history on each new dividend — see data.fingerprint.
    start="2007-05-30", end="2026-06-30", n_days=4802,
    fp="958f652ce4c9", fp_levels="aa48518ec578",
    # Book A: 60/40 SPY/IEF
    a_drift_s=0.6788, a_ann_s=0.6673, a_qtr_s=0.6570, a_bnd_s=0.6412,
    a_drift_c=7.23, a_ann_c=6.79, a_qtr_c=6.84, a_bnd_c=6.88,
    a_drift_v=11.22, a_ann_v=10.71, a_qtr_v=11.00, a_bnd_v=11.40,
    a_drift_dd=-24.92, a_ann_dd=-30.01, a_qtr_dd=-31.74, a_bnd_dd=-31.86,
    a_ann_to=0.079, a_qtr_to=0.157, a_bnd_to=0.114,
    a_ann_n=19, a_qtr_n=76, a_bnd_n=22,
    a_diff_ba=-0.0261, a_t_ba=0.82, a_cagr_ba=0.092,
    a_diff_bq=-0.0158, a_t_bq=0.64,
    a_diff_qa=-0.0103, a_t_qa=0.53,
    # Same window, same trades, run GROSS-of-cash: the HAC t is identical (cash cancels
    # exactly in the return difference) but the Sharpe DIFFERENCE is not cash-invariant.
    # These are the only numbers the pre-2007 cross-check may be compared against.
    a_diff_ba_gross=-0.0340, a_diff_bq_gross=-0.0204, a_diff_qa_gross=-0.0136,
    a_ci_lo=-0.0674, a_ci_hi=0.0142, a_ci_neg=87.9,
    # Book B: 50/30/20 SPY/IEF/GLD
    b_drift_s=0.7393, b_ann_s=0.7718, b_qtr_s=0.7559, b_bnd_s=0.7441,
    b_diff_ba=-0.0277, b_t_ba=0.80, b_cagr_ba=0.122,
    b_ci_lo=-0.0685, b_ci_hi=0.0148,
    # Eras (60/40, split 2016-01-01)
    e1_n=2164, e1_ann=0.584, e1_qtr=0.555, e1_bnd=0.535, e1_diff=-0.0491, e1_t=0.15,
    e2_n=2637, e2_ann=0.741, e2_qtr=0.751, e2_bnd=0.724, e2_diff=-0.0170, e2_t=0.83,
    # Cost sweep (60/40)
    c0_bnd=0.6416, c0_diff=-0.0260, c0_t=0.83,
    c50_ann=0.6640, c50_qtr=0.6505, c50_bnd=0.6367, c50_diff=-0.0273, c50_t=0.73,
    drag50_ann=3.94, drag50_qtr=7.86, drag50_bnd=5.69,
    # Band-width sweep (60/40)
    bw=[(2, 10, 0.6346, 98, 0.221, -0.0327, 0.38),
        (3, 15, 0.6334, 44, 0.146, -0.0339, 0.31),
        (5, 25, 0.6412, 22, 0.114, -0.0261, 0.82),
        (8, 40, 0.6544, 9, 0.076, -0.0129, 0.93),
        (10, 50, 0.6488, 6, 0.063, -0.0185, 1.42)],
    # Weight discipline (60/40, SPY sleeve)
    w_drift_min=0.357, w_drift_max=0.850, w_drift_fin=0.846, w_drift_dev=0.1016,
    w_ann_min=0.418, w_ann_max=0.672, w_ann_dev=0.0219,
    w_qtr_min=0.481, w_qtr_max=0.666, w_qtr_dev=0.0133,
    w_bnd_min=0.536, w_bnd_max=0.652, w_bnd_dev=0.0178,
    # Long cross-check (SPY/IEF 2003-2026, gross of cash)
    l_start="2003-01-03", l_n=5909, l_fp="7788344e5927",
    l_drift=0.8120, l_ann=0.8682, l_qtr=0.8558, l_bnd=0.8361,
    l_diff_ba=-0.0321, l_t_ba=1.04,
    # Synthetic control
    syn_pl_diff=0.0414, syn_pl_sd=0.0093, syn_pl_fire=7,
    syn_nl_diff=0.0044, syn_nl_sd=0.0142, syn_nl_fire=0,
    # The published claim we are trying to reproduce
    claim_pp=0.5,
)

BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
)


HEADER = f"""# Study 936 — Tolerance Bands ⚖️

**Does rebalancing on 5/25 tolerance bands beat rebalancing on the calendar?**

Every allocation guide eventually reaches the rebalancing chapter, and the answer it gives
is almost always the same: don't trade on the calendar, trade on **tolerance bands** — act
when a sleeve is off target by *5 percentage points absolute or 25% of its own weight*,
whichever binds first. The promise: better risk-adjusted returns **and** less turnover,
because you only trade when the book has actually moved.

We test it on **SPY / IEF / GLD** with **BIL** as the cash leg,
{R['start']} → {R['end']} ({R['n_days']:,} days of total-return closes), one execution lag
(breach seen at the close of day *t*, traded at the close of *t+1*), one-way cost charged
on traded notional, every Sharpe **excess-of-cash**.

*Real-tape numbers below are the frozen headline from `docs/results.md`
(returns fingerprint `{R['fp']}`, as-of 2026-06-30). The live cells at the end run the
**synthetic** control and are labelled as such.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The question, stated fairly\n\n"
           "A 60/40 book does not stay 60/40. Stocks out-run bonds, the equity sleeve grows, "
           "and within a few years you own a portfolio you never chose. So you rebalance. "
           "The only real decision is **when**: on a date (every January; every quarter), or "
           "on a **condition** (whenever the equity sleeve is more than 5 points off).\n\n"
           "The condition sounds obviously better. It carries information about your actual "
           "book; a date carries none. Let's see how much that is worth."),
        md("## 2. The answer: about a third of one percent of a Sharpe point\n\n"
           "Here are the four schedules on the same 60/40 SPY/IEF book, all excess-of-cash, "
           "all costed."),
        code(
            "# Frozen real-tape headline (docs/results.md). Nothing is recomputed here.\n"
            "R = " + repr({k: R[k] for k in (
                "a_drift_s", "a_ann_s", "a_qtr_s", "a_bnd_s",
                "a_drift_c", "a_ann_c", "a_qtr_c", "a_bnd_c",
                "a_ann_to", "a_qtr_to", "a_bnd_to",
                "a_diff_ba", "a_t_ba", "a_ci_lo", "a_ci_hi",
                "drag50_ann", "drag50_qtr", "drag50_bnd")}) + "\n"
            "rows = [('never rebalance', R['a_drift_s'], R['a_drift_c'], 0.0),\n"
            "        ('annual',          R['a_ann_s'],   R['a_ann_c'],   R['a_ann_to']),\n"
            "        ('quarterly',       R['a_qtr_s'],   R['a_qtr_c'],   R['a_qtr_to']),\n"
            "        ('5/25 bands',      R['a_bnd_s'],   R['a_bnd_c'],   R['a_bnd_to'])]\n"
            "print(f\"{'schedule':<16}{'excess Sharpe':>15}{'CAGR':>9}{'traded/yr':>12}\")\n"
            "for name, s, c, to in rows:\n"
            "    print(f'{name:<16}{s:>15.3f}{c:>8.2f}%{to:>11.1%}')\n"
            "print()\n"
            "print(f\"bands minus annual: {R['a_diff_ba']:+.3f} Sharpe  \"\n"
            "      f\"(HAC t = {R['a_t_ba']:+.2f}, 95% CI \"\n"
            "      f\"[{R['a_ci_lo']:+.3f}, {R['a_ci_hi']:+.3f}])\")"
        ),
        md(f"All four are within **{abs(R['a_drift_s'] - R['a_bnd_s']):.3f} of a Sharpe "
           f"point** of each other over nineteen years. The band rule comes out "
           f"**{R['a_diff_ba']:+.3f}** against annual — the *wrong* side of zero for the "
           f"claim, and nowhere near big enough to call. The confidence interval "
           f"**[{R['a_ci_lo']:+.3f}, {R['a_ci_hi']:+.3f}]** is not wide because the test is "
           "weak; it is narrow, and it is centred on nothing.\n\n"
           "> 🔬 **For the quants:** the sign is split. The band book earns a *fractionally "
           "higher* mean daily return (which is why the HAC *t* on the return difference is "
           f"**{R['a_t_ba']:+.2f}**, positive) and runs *more* volatility "
           f"({R['a_bnd_v']:.2f}% vs {R['a_ann_v']:.2f}%), so its Sharpe lands fractionally "
           "lower. Both effects are noise; they happen to point opposite ways."),
        md(f"## 3. The published claim, and what actually turned up\n\n"
           f"The strongest version of the band case in print — Daryanani's 2008 "
           f"*Opportunistic Rebalancing* — reports roughly **{R['claim_pp']:.1f} percentage "
           f"points a year** for bands over annual calendar rebalancing. On this tape the "
           f"gap is **+{R['a_cagr_ba']:.2f} pp/yr**: about a fifth of the claim, and well "
           f"inside the noise. The 3-asset 50/30/20 SPY/IEF/GLD book gives "
           f"**{R['b_diff_ba']:+.3f}** Sharpe (*t* = {R['b_t_ba']:+.2f}); the 23-year "
           f"SPY/IEF cross-check gives **{R['l_diff_ba']:+.3f}** (*t* = {R['l_t_ba']:+.2f}) "
           f"— that one is measured *gross of cash*, because no tradable cash ETF existed "
           f"before 2007, so its like-for-like comparator is the "
           f"**{R['a_diff_ba_gross']:+.3f}** you get by running the headline window the same "
           f"way. Three shots, same nothing."),
        md("## 4. Costs are not the argument\n\n"
           "The usual defence of bands is that they save you turnover. They do — and it "
           "does not matter, because a 60/40 ETF book barely trades under *any* schedule."),
        code(
            "sched = [('annual', R['a_ann_to'], R['drag50_ann']),\n"
            "         ('quarterly', R['a_qtr_to'], R['drag50_qtr']),\n"
            "         ('5/25 bands', R['a_bnd_to'], R['drag50_bnd'])]\n"
            "print('traded notional per year, and the annual drag at a PUNITIVE 50 bps one-way')\n"
            "print('(ten times a realistic SPY/IEF spread)')\n"
            "print()\n"
            "for name, turn, drag in sched:\n"
            "    print(f'{name:<12} {turn:>6.1%} of NAV traded per year  ->  {drag:>4.1f} bp/yr of drag')\n"
            "gap = R['drag50_qtr'] - R['drag50_ann']\n"
            "print(f'\\nThe gap between the most and least expensive schedule: {gap:.1f} bp/yr.')"
        ),
        md("Four basis points a year, at ten times realistic cost. The turnover argument is "
           "true and irrelevant.\n\n"
           "> 🔬 **For the quants:** this is why the cost sweep in `docs/results.md` is flat "
           f"from 0 to 50 bps — bands minus annual moves from {R['c0_diff']:+.4f} to "
           f"{R['c50_diff']:+.4f}. Friction is simply not the binding constraint at ETF "
           "spreads; it would be at 1990s mutual-fund loads, which is the era the folklore "
           "was formed in."),
        md(f"## 5. What rebalancing *does* buy — and here bands genuinely win\n\n"
           f"There is one axis on which the schedules differ enormously, and it has nothing "
           f"to do with return. Left alone, the 60/40 book ended the sample holding "
           f"**{R['w_drift_fin']:.1%} equity** — a portfolio nobody chose, with a completely "
           f"different risk profile from the one that was signed off.\n\n"
           f"| schedule | equity sleeve stayed between | traded/yr |\n"
           f"|---|---|---|\n"
           f"| never rebalance | {R['w_drift_min']:.1%} – {R['w_drift_max']:.1%} | 0.0% |\n"
           f"| annual | {R['w_ann_min']:.1%} – {R['w_ann_max']:.1%} | {R['a_ann_to']:.1%} |\n"
           f"| quarterly | {R['w_qtr_min']:.1%} – {R['w_qtr_max']:.1%} | {R['a_qtr_to']:.1%} |\n"
           f"| **5/25 bands** | **{R['w_bnd_min']:.1%} – {R['w_bnd_max']:.1%}** | "
           f"**{R['a_bnd_to']:.1%}** |\n\n"
           f"The band rule holds the **tightest envelope** for "
           f"**{1 - R['a_bnd_to'] / R['a_qtr_to']:.0%} less traded notional than quarterly**. "
           f"Two things that claim does *not* say, and that the table above will tell you if "
           f"you read it carefully: annual rebalancing is **cheaper still** "
           f"({R['a_ann_to']:.1%} a year — it just lets the book roam nearly twice as far), "
           f"and quarterly sits **closer to target on an average day**. What bands minimise "
           f"is the *worst* excursion, which is the number a risk mandate is written on.\n\n"
           f"That is a real, reproducible advantage — it is just an advantage in *governance*, "
           f"not in return. If your reason for rebalancing is 'I want to still own the "
           f"portfolio I chose', bands are the best tool here. If your reason is 'it will "
           f"make me more money', the tape says no."),
        md("## 6. Live check — the machinery is not broken (offline synthetic)\n\n"
           "**This cell runs on a synthetic tape, not on the real one.** We build a world "
           "where the two sleeves' relative performance genuinely mean-reverts (a 60-day "
           "half-life) — exactly the world in which trading on dispersion *should* pay — and "
           "a null world where dispersion is a random walk. The band rule must win in the "
           "first and not in the second."),
        code(
            BOOT +
            "import numpy as np\n"
            "from rebal_bands import data, strategy as st\n"
            "for tag, ss in [('planted (mean-reverting)', 1.0), ('null (random walk)', 0.0)]:\n"
            "    d = st.synthetic_detect(data.synthetic_panel(signal_strength=ss, seed=936)[0])\n"
            "    print(f\"{tag:26s}: bands - annual Sharpe {d['diff_bands_annual']:+.4f}  \"\n"
            "          f\"(HAC t {d['t_bands_annual']:+.2f})\")"
        ),
        md(f"The detector fires on the planted world (**{R['syn_pl_diff']:+.4f}**, HAC "
           f"*t* >= 2 on **{R['syn_pl_fire']}/8** seeds) and stays quiet on the null "
           f"(**{R['syn_nl_diff']:+.4f}**, **{R['syn_nl_fire']}/8**). So the flat real-tape "
           "answer is a fact about stocks, bonds and gold — their relative performance does "
           "not mean-revert on the horizon a 5-point band operates over — not a bug in the "
           "backtest."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Bands minus annual is **{R['a_diff_ba']:+.3f}** excess "
           f"Sharpe (HAC *t* = {R['a_t_ba']:+.2f}), CI "
           f"**[{R['a_ci_lo']:+.3f}, {R['a_ci_hi']:+.3f}]**, and the same non-result holds on "
           f"the 3-asset book, in both eras, at every cost from 0 to 50 bps, and across band "
           f"widths from 2/10 to 10/50. The famous ~{R['claim_pp']:.1f} pp/yr advantage comes "
           f"out at **+{R['a_cagr_ba']:.2f} pp/yr**.\n"
           f"- **Tradability — Mirage.** Nothing to bank. The schedules differ by under "
           f"4 bp/yr of cost even at ten times realistic spreads — and tax, the one cost we "
           f"do *not* model, runs against the busier schedules, not for them.\n"
           f"- **What is real.** Rebalance on *something*: the do-nothing book finished at "
           f"{R['w_drift_fin']:.0%} equity. Among schedules that do the job, 5/25 bands give "
           f"the tightest *worst-case* weight envelope for less trading than quarterly — not "
           f"the least trading outright (annual is cheaper), and not the closest average "
           f"tracking (quarterly is closer). Pick your rebalancing rule for the portfolio you "
           f"want to still own, not for the return you hope to add.")
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    bw_lines = "\n".join(
        f"| {a}% / {b}% | {s:+.4f} | {n} | {to:.3f} | {d:+.4f} (*t* = {t:+.2f}) |"
        for a, b, s, n, to, d, t in R["bw"]
    )
    cells = [
        md("# Study 936 — Tolerance Bands — the teardown\n\n"
           "One book, four schedules, one execution lag. The excess-of-cash race, the "
           "Newey-West *t* on the daily return difference, paired block-bootstrap CIs on the "
           "Sharpe difference, the era cut, the cost and band-width sweeps, the 23-year "
           "cross-check, and the live synthetic control.\n\n"
           "**Design.** All four schedules hold the *identical* sleeves in the *identical* "
           "target mix, so the market factor and the cash leg cancel exactly in every "
           "pairwise **return** difference. That difference series is therefore the clean "
           "statistic: it isolates *when each rule trades* from *what the book owns*. "
           "(Careful: a Sharpe **ratio** is nonlinear, so the *Sharpe* difference is **not** "
           "cash-invariant — excess-of-cash and gross-of-cash Sharpe gaps differ by ~0.008 "
           "here. Sharpe gaps are only ever compared on a matched cash basis.) "
           "Signal read at the "
           "close of day *t*, executed at the close of *t+1* — one lag, nowhere else. Cost "
           "is one-way bps x traded notional / NAV; nothing is shorted, so no borrow leg. "
           "**Tax is not modelled** and is the one omitted cost that is first-order relative "
           "to the effects measured here.\n\n"
           "Every real number is frozen from `docs/results.md` (returns fingerprint `%s`; "
           "the levels fingerprint `%s` is printed too but churns on every dividend "
           "re-adjustment, so it is a diagnostic, not the key), as-of 2026-06-30."
           % (R["fp"], R["fp_levels"])),
        code("R = %r" % (R,)),
        md("## Book A — 60/40 SPY/IEF, excess-of-cash, 5 bps one-way"),
        code(
            "hdr = f\"{'schedule':<11}{'exSharpe':>10}{'CAGR':>9}{'vol':>8}{'maxDD':>9}\"\n"
            "hdr += f\"{'traded/yr':>11}{'rebals':>8}\"\n"
            "print(hdr)\n"
            "rows = [('drift', R['a_drift_s'], R['a_drift_c'], R['a_drift_v'], R['a_drift_dd'], 0.0, 0),\n"
            "        ('annual', R['a_ann_s'], R['a_ann_c'], R['a_ann_v'], R['a_ann_dd'], R['a_ann_to'], R['a_ann_n']),\n"
            "        ('quarterly', R['a_qtr_s'], R['a_qtr_c'], R['a_qtr_v'], R['a_qtr_dd'], R['a_qtr_to'], R['a_qtr_n']),\n"
            "        ('bands', R['a_bnd_s'], R['a_bnd_c'], R['a_bnd_v'], R['a_bnd_dd'], R['a_bnd_to'], R['a_bnd_n'])]\n"
            "for name, s, c, v, dd, to, n in rows:\n"
            "    print(f'{name:<11}{s:>+10.4f}{c:>8.2f}%{v:>7.2f}%{dd:>8.2f}%{to:>11.3f}{n:>8d}')\n"
            "print()\n"
            "for tag, d, t in [('bands - annual', R['a_diff_ba'], R['a_t_ba']),\n"
            "                  ('bands - quarterly', R['a_diff_bq'], R['a_t_bq']),\n"
            "                  ('quarterly - annual', R['a_diff_qa'], R['a_t_qa'])]:\n"
            "    print(f'{tag:<20} Sharpe diff {d:+.4f}   HAC t (return diff) {t:+.2f}')"
        ),
        md("> 💡 **In plain words:** the four ways of rebalancing land within three "
           "hundredths of a Sharpe point of each other over nineteen years. Whatever you "
           "were arguing about in the rebalancing chapter, it was not this.\n\n"
           "The sign split is worth being explicit about. `bands - annual` has a **positive** "
           f"HAC *t* ({R['a_t_ba']:+.2f}) on the daily return difference and a **negative** "
           f"Sharpe difference ({R['a_diff_ba']:+.4f}): the band book earns marginally more "
           f"per day and carries marginally more variance ({R['a_bnd_v']:.2f}% vs "
           f"{R['a_ann_v']:.2f}% annualised). Both are noise. Reporting only the one with the "
           "convenient sign would be the classic way to manufacture a result here."),
        md("## Book B — 50/30/20 SPY/IEF/GLD\n\n"
           "The 3-asset book is where the 5/25 rule's *relative* leg finally binds: the 20% "
           "gold sleeve gets a 2.5 pp band, not a 5 pp one. If the relative leg is what makes "
           "the rule special, it should show up here."),
        code(
            "print(f\"drift {R['b_drift_s']:+.4f}   annual {R['b_ann_s']:+.4f}   \"\n"
            "      f\"quarterly {R['b_qtr_s']:+.4f}   bands {R['b_bnd_s']:+.4f}\")\n"
            "print(f\"bands - annual: {R['b_diff_ba']:+.4f}  (HAC t {R['b_t_ba']:+.2f}, \"\n"
            "      f\"CI [{R['b_ci_lo']:+.4f}, {R['b_ci_hi']:+.4f}])\")\n"
            "print(f\"CAGR gap: {R['b_cagr_ba']:+.3f} pp/yr  vs the {R['claim_pp']:.1f} pp/yr claimed in the literature\")"
        ),
        md("## Paired block-bootstrap CI on the Sharpe difference\n\n"
           "2,000 circular block resamples, 21-day blocks, the **same** block indices applied "
           "to both arms so the day-by-day pairing survives. Pairing is what makes these "
           "intervals tight: the common market factor cancels inside every resample."),
        code(
            "print(f\"60/40    bands - annual: {R['a_diff_ba']:+.4f}  \"\n"
            "      f\"95% CI [{R['a_ci_lo']:+.4f}, {R['a_ci_hi']:+.4f}]  share<0 {R['a_ci_neg']:.1f}%\")\n"
            "print(f\"50/30/20 bands - annual: {R['b_diff_ba']:+.4f}  \"\n"
            "      f\"95% CI [{R['b_ci_lo']:+.4f}, {R['b_ci_hi']:+.4f}]\")\n"
            "width = R['a_ci_hi'] - R['a_ci_lo']\n"
            "print(f'\\ninterval width {width:.3f} Sharpe points -> a PRECISE null, not an underpowered test')"
        ),
        md("## Era cut (split 2016-01-01), 60/40\n\n"
           "The 2022 regime break flipped the stock/bond correlation, which is the single "
           "biggest driver of how far a 60/40 book drifts between rebalances. If the schedule "
           "choice ever mattered, it should matter differently on the two sides of that."),
        code(
            "print(f\"2007-2015 (n={R['e1_n']}): annual {R['e1_ann']:+.3f}  quarterly {R['e1_qtr']:+.3f}  \"\n"
            "      f\"bands {R['e1_bnd']:+.3f}  |  bands-annual {R['e1_diff']:+.4f} (t={R['e1_t']:+.2f})\")\n"
            "print(f\"2016-2026 (n={R['e2_n']}): annual {R['e2_ann']:+.3f}  quarterly {R['e2_qtr']:+.3f}  \"\n"
            "      f\"bands {R['e2_bnd']:+.3f}  |  bands-annual {R['e2_diff']:+.4f} (t={R['e2_t']:+.2f})\")\n"
            "print('\\nsame sign, same non-significance in both halves')"
        ),
        md("## Cost sweep — friction is not the binding constraint"),
        code(
            "print(f\"  0 bps: bands {R['c0_bnd']:+.4f}   bands-annual {R['c0_diff']:+.4f} (t={R['c0_t']:+.2f})\")\n"
            "print(f\" 50 bps: annual {R['c50_ann']:+.4f}  quarterly {R['c50_qtr']:+.4f}  \"\n"
            "      f\"bands {R['c50_bnd']:+.4f}   bands-annual {R['c50_diff']:+.4f} (t={R['c50_t']:+.2f})\")\n"
            "print()\n"
            "print('annual drag at 50 bps one-way (ten times realistic):')\n"
            "print(f\"  annual {R['drag50_ann']:.2f} bp/yr   quarterly {R['drag50_qtr']:.2f} bp/yr   \"\n"
            "      f\"bands {R['drag50_bnd']:.2f} bp/yr\")"
        ),
        md("> 💡 **In plain words:** the whole cost argument for tolerance bands is worth "
           "about four basis points a year, at ten times realistic trading costs. It is real "
           "and it is negligible — the folklore was formed when retail rebalancing meant "
           "loaded mutual funds, not penny-spread ETFs."),
        md("## The 5/25 widths are an ASSUMPTION — sweep them\n\n"
           "5/25 is a memorable round number, not a calibrated parameter. If the result "
           "depended on it, that dependence would be the finding.\n\n"
           "| Band | Excess Sharpe | Rebalances | Traded/yr | vs annual (*t*) |\n"
           "|---|--:|--:|--:|--:|\n" + bw_lines + "\n\n"
           "A five-fold range of trigger widths moves the excess Sharpe by 0.02 with no "
           "monotone pattern beyond *trade less, keep a hair more*. 5/25 sits inside the flat "
           "region — neither optimal nor harmful."),
        md("## Weight discipline — the one axis where the schedules genuinely separate"),
        code(
            "print(f\"{'schedule':<11}{'SPY min':>9}{'SPY max':>9}{'mean |dev|':>12}\")\n"
            "for name, lo, hi, dev in [('drift', R['w_drift_min'], R['w_drift_max'], R['w_drift_dev']),\n"
            "                          ('annual', R['w_ann_min'], R['w_ann_max'], R['w_ann_dev']),\n"
            "                          ('quarterly', R['w_qtr_min'], R['w_qtr_max'], R['w_qtr_dev']),\n"
            "                          ('bands', R['w_bnd_min'], R['w_bnd_max'], R['w_bnd_dev'])]:\n"
            "    print(f'{name:<11}{lo:>9.1%}{hi:>9.1%}{dev:>12.4f}')\n"
            "print(f\"\\ndrift book final equity weight: {R['w_drift_fin']:.1%} \"\n"
            "      f\"(it started at 60.0%)\")\n"
            "print(f\"bands traded {1 - R['a_bnd_to']/R['a_qtr_to']:.0%} less notional than quarterly \"\n"
            "      f\"for a TIGHTER envelope\")\n"
            "print(f\"but ANNUAL is cheaper still ({R['a_ann_to']:.1%}/yr) at a much wider envelope,\")\n"
            "print(f\"and QUARTERLY tracks target closer on an AVERAGE day \"\n"
            "      f\"({R['w_qtr_dev']:.4f} vs {R['w_bnd_dev']:.4f}) — bands minimise the WORST excursion\")"
        ),
        md(f"## Longer-window cross-check — SPY/IEF from {R['l_start']}, gross of cash\n\n"
           f"BIL does not exist before 2007, so this window has **no tradable cash leg** and is "
           f"run gross of cash; that buys four extra years of sample. The trap — and the first "
           f"draft of this study walked into it — is comparing its Sharpe gap with the "
           f"excess-of-cash headline. The cash leg cancels **exactly** in the daily *return* "
           f"difference (the HAC *t* is bit-identical either way), but a Sharpe *ratio* is "
           f"nonlinear, so the Sharpe *difference* moves. The cell below prints the matched "
           f"comparator. (Returns fingerprint `{R['l_fp']}`, n = {R['l_n']:,}.)"),
        code(
            "print('headline window 2007-2026, 60/40 — the SAME trades, two cash treatments:')\n"
            "for tag, ex, gr in [('bands - annual', R['a_diff_ba'], R['a_diff_ba_gross']),\n"
            "                    ('bands - quarterly', R['a_diff_bq'], R['a_diff_bq_gross']),\n"
            "                    ('quarterly - annual', R['a_diff_qa'], R['a_diff_qa_gross'])]:\n"
            "    print(f'  {tag:<20} excess {ex:+.4f}   gross {gr:+.4f}   shift {gr-ex:+.4f}')\n"
            "print()\n"
            "print(f\"long window {R['l_start']}-2026 (gross of cash, n={R['l_n']:,}):\")\n"
            "print(f\"  drift {R['l_drift']:+.4f}   annual {R['l_ann']:+.4f}   \"\n"
            "      f\"quarterly {R['l_qtr']:+.4f}   bands {R['l_bnd']:+.4f}\")\n"
            "print(f\"  bands - annual: {R['l_diff_ba']:+.4f}  (HAC t {R['l_t_ba']:+.2f})\")\n"
            "print(f\"  vs the matched gross comparator {R['a_diff_ba_gross']:+.4f} on the headline window\")\n"
            "print('\\n23 years, same answer — and, on a matched basis, a FRACTIONALLY SMALLER')\n"
            "print('adverse gap than the headline window, not a larger one')"
        ),
        md("## Live synthetic control — the harness reads the sign correctly\n\n"
           "**Synthetic tape, not the real one.** Two sleeves whose idiosyncratic components "
           "follow an AR(1): at `signal_strength=1` dispersion mean-reverts with a 60-day "
           "half-life (acting on dispersion *should* pay — the planted world); at "
           "`signal_strength=0` dispersion is a random walk (nothing to harvest — the null). "
           "The panel is deliberately favourable to bands: equal drifts, equal vols, almost "
           "no common factor. If the band rule could not win *there*, the detector would be "
           "broken."),
        code(
            BOOT +
            "import numpy as np\n"
            "from rebal_bands import data, strategy as st\n"
            "for tag, ss in [('planted (reverting)', 1.0), ('null (random walk)', 0.0)]:\n"
            "    diffs, ts = [], []\n"
            "    for s in range(4):\n"
            "        d = st.synthetic_detect(data.synthetic_panel(signal_strength=ss, seed=936+s)[0])\n"
            "        diffs.append(d['diff_bands_annual']); ts.append(d['t_bands_annual'])\n"
            "    diffs, ts = np.array(diffs), np.array(ts)\n"
            "    print(f\"{tag:22s}: bands-annual {diffs.mean():+.4f} (sd {diffs.std(ddof=1):.4f}), \"\n"
            "          f\"HAC t >= 2 in {(ts >= 2).sum()}/4 seeds\")"
        ),
        md(f"Over the full 8-seed run in `docs/results.md`: planted "
           f"**{R['syn_pl_diff']:+.4f}** (sd {R['syn_pl_sd']:.4f}), fires "
           f"**{R['syn_pl_fire']}/8**; null **{R['syn_nl_diff']:+.4f}** (sd "
           f"{R['syn_nl_sd']:.4f}), fires **{R['syn_nl_fire']}/8**. The detector recovers a "
           "planted effect and is silent on the null, so the real-tape flatness is a property "
           "of the asset tape: **stock/bond/gold relative performance does not mean-revert on "
           "the horizon a 5-point band operates over.** That is the economic reason the rule "
           "has nothing to harvest, and it is why no amount of band tuning rescues it."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Across two books, two windows, two eras, six cost levels "
           f"and five band widths, no schedule difference clears |*t*| = 2 in any "
           f"specification. Headline: bands − annual **{R['a_diff_ba']:+.4f}** excess Sharpe, "
           f"HAC *t* on the daily return difference **{R['a_t_ba']:+.2f}**, paired bootstrap "
           f"CI **[{R['a_ci_lo']:+.4f}, {R['a_ci_hi']:+.4f}]** (share < 0 = "
           f"{R['a_ci_neg']:.1f}%). 3-asset book {R['b_diff_ba']:+.4f} "
           f"(*t* = {R['b_t_ba']:+.2f}); 2003-2026 cross-check {R['l_diff_ba']:+.4f} "
           f"*gross-of-cash* (*t* = {R['l_t_ba']:+.2f}) against a "
           f"{R['a_diff_ba_gross']:+.4f} gross-of-cash comparator on the headline window. "
           f"Daryanani's ~{R['claim_pp']:.1f} pp/yr reproduces as "
           f"**+{R['a_cagr_ba']:.2f} pp/yr**. Intervals are narrow — a precise null. "
           f"Survivorship: the sleeves are surviving mega-ETFs picked with hindsight, which "
           f"can only flatter a result, and there is no result to flatter.\n"
           f"- **Tradability — Mirage.** Turnover is 7-17% of NAV/yr under every schedule; at "
           f"a punitive 50 bps one-way the schedules separate by under 4 bp/yr. The only "
           f"first-order unmodelled cost is **tax**, and it penalises the higher-turnover "
           f"schedules — so modelling it would widen the gap *against* quarterly, not open "
           f"one for bands.\n"
           f"- **What survives.** Weight discipline, on one statistic. 5/25 bands hold the "
           f"equity sleeve inside {R['w_bnd_min']:.1%}-{R['w_bnd_max']:.1%} for "
           f"{1 - R['a_bnd_to']/R['a_qtr_to']:.0%} less traded notional than quarterly, while "
           f"the do-nothing book finishes at {R['w_drift_fin']:.1%} equity. State the "
           f"statistic honestly, though: annual trades *less* than bands "
           f"({R['a_ann_to']:.1%} vs {R['a_bnd_to']:.1%}) and quarterly tracks target closer "
           f"*on average* ({R['w_qtr_dev']:.4f} vs {R['w_bnd_dev']:.4f}). Bands win the "
           f"**worst-excursion** contest and nothing else. Choose the rule for the risk "
           f"profile you want to keep, and do not budget a return for it.")
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
