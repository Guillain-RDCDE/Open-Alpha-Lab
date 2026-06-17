"""Build the two narrative notebooks for Study 247 (Bond-Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both walk the seven desk beats at two altitudes. Figures are computed live from the
``bond_seasonality`` package on the cached real tapes; the frozen ``R`` dict mirrors
``docs/results.md`` for the verdict prose so the two never drift.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers - mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    as_of="2026-06-12",
    # TLT total-return
    tlt_n=6007, tlt_fp="f83a9db36f17", tlt_n_tom=1150,
    tlt_mu_tom=6.67, tlt_mu_rest=0.71, tlt_gap=5.95,
    tlt_t_tom=2.465, tlt_t_gap=1.987,
    tlt_win_tom=54.7, tlt_win_rest=51.5,
    tlt_book_cagr=2.81, tlt_bh_cagr=3.71,
    tlt_book_sharpe_whole=0.470, tlt_book_sharpe_inv=1.119,
    tlt_book_pct=19.1, tlt_book_mdd=-18.4,
    # IEF total-return
    ief_n=6007, ief_fp="a83262b21d6b", ief_n_tom=1150,
    ief_mu_tom=4.62, ief_mu_rest=0.76, ief_gap=3.86,
    ief_t_tom=3.515, ief_t_gap=2.664,
    ief_win_tom=54.7, ief_win_rest=51.1,
    ief_book_cagr=1.96, ief_bh_cagr=3.60,
    ief_book_sharpe_whole=0.657, ief_book_sharpe_inv=1.599,
    ief_book_pct=19.1, ief_book_mdd=-7.5,
    # Turn-of-year
    tlt_toy_gap=-2.38, tlt_toy_t=-0.40,
    ief_toy_gap=1.48, ief_toy_t=0.53,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study root (bond_seasonality/)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from bond_seasonality import data, strategy

AS_OF = "2026-06-12"
tlt = data.load_real("TLT", start="2002-07-30").loc[:AS_OF]
tlt_close = tlt["close"]
tlt_tom = data.tom_mask(tlt_close.index, n_end=2, n_start=2)
ief = data.load_real("IEF", start="2002-07-30").loc[:AS_OF]
ief_close = ief["close"]
ief_tom = data.tom_mask(ief_close.index, n_end=2, n_start=2)
et = strategy.effect_stats(tlt_close, tlt_tom)
ei = strategy.effect_stats(ief_close, ief_tom)
print(f"TLT: {len(tlt_close):,} rows  {tlt_close.index[0].date()} -> {tlt_close.index[-1].date()}  fp={data.fingerprint(tlt)}")
print(f"  TOM: {et['mu_tom']*1e4:.2f} bps vs rest {et['mu_rest']*1e4:.2f} bps  gap {et['gap']*1e4:+.2f} bps  HAC t(gap)={et['t_gap']:.2f}")
print(f"IEF: {len(ief_close):,} rows  fp={data.fingerprint(ief)}")
print(f"  TOM: {ei['mu_tom']*1e4:.2f} bps vs rest {ei['mu_rest']*1e4:.2f} bps  gap {ei['gap']*1e4:+.2f} bps  HAC t(gap)={ei['t_gap']:.2f}")
"""

HERO_BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Calendar bid confirmed?: Mixed](https://img.shields.io/badge/Calendar_bid_confirmed%3F-Mixed-8b949e?style=flat-square)\n\n"
)

FIG_BAR = """\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, e, label in zip(axes, [et, ei], ["TLT (20+ yr)", "IEF (7-10 yr)"]):
    vals = [e["mu_tom"]*1e4, e["mu_rest"]*1e4]
    bars = ax.bar(["TOM days", "Other days"], vals, color=["C3", "C0"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.1f} bps", ha="center", va="bottom")
    ax.set_ylabel("Mean daily return (bps)")
    ax.set_title(f"{label}: TOM vs mid-month (t(gap)={e['t_gap']:+.2f})")
plt.suptitle("Treasury turn-of-month: mean daily return on TOM vs other days", fontsize=12)
plt.tight_layout(); plt.show()
print(f"TLT gap = {et['gap']*1e4:+.2f} bps  HAC t = {et['t_gap']:+.2f}")
print(f"IEF gap = {ei['gap']*1e4:+.2f} bps  HAC t = {ei['t_gap']:+.2f}")
print("IEF clears t=2; TLT just misses - the bond calendar effect is soft.")
"""

FIG_CUM = """\
book_t = strategy.tom_only_book(tlt_close, tlt_tom, cost_bps=1.0)
bh_t   = strategy.buy_and_hold(tlt_close)
book_i = strategy.tom_only_book(ief_close, ief_tom, cost_bps=1.0)
bh_i   = strategy.buy_and_hold(ief_close)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, bk, bh, label in zip(axes, [book_t, book_i], [bh_t, bh_i], ["TLT", "IEF"]):
    ax.plot(bh["equity"], label=f"Buy & hold ({bh['cagr']*100:.1f}%/yr)", lw=1.3)
    ax.plot(bk["equity"], label=f"TOM-only ({bk['cagr']*100:.1f}%/yr, in mkt {bk['time_in_market']*100:.0f}%)", lw=1.3, color="C3")
    ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log)")
    ax.set_title(f"{label}: TOM-only vs buy-and-hold"); ax.legend()
plt.suptitle("TOM book trails B&H despite higher Sharpe on IEF", fontsize=12)
plt.tight_layout(); plt.show()
print(f"TLT - TOM-only CAGR {book_t['cagr']*100:.2f}% vs B&H {bh_t['cagr']*100:.2f}%; Sharpe(whole) {book_t['sharpe_whole']:.3f} vs {bh_t['sharpe']:.3f}")
print(f"IEF - TOM-only CAGR {book_i['cagr']*100:.2f}% vs B&H {bh_i['cagr']*100:.2f}%; Sharpe(whole) {book_i['sharpe_whole']:.3f} vs {bh_i['sharpe']:.3f}")
"""

FIG_TOY = """\
toy_tlt = pd.Series(False, index=tlt_close.pct_change().dropna().index)
for yr in toy_tlt.index.year.unique():
    days = toy_tlt.index[toy_tlt.index.year == yr].sort_values()
    for d in list(days[:5]) + list(days[-5:]): toy_tlt.loc[d] = True
toy_ief = pd.Series(False, index=ief_close.pct_change().dropna().index)
for yr in toy_ief.index.year.unique():
    days = toy_ief.index[toy_ief.index.year == yr].sort_values()
    for d in list(days[:5]) + list(days[-5:]): toy_ief.loc[d] = True
etoy_t = strategy.effect_stats(tlt_close, toy_tlt)
etoy_i = strategy.effect_stats(ief_close, toy_ief)
print("=== Turn-of-year ===")
print(f"TLT: gap={etoy_t['gap']*1e4:+.2f} bps  HAC t(gap)={etoy_t['t_gap']:+.2f}  n_toy={etoy_t['n_tom']}")
print(f"IEF: gap={etoy_i['gap']*1e4:+.2f} bps  HAC t(gap)={etoy_i['t_gap']:+.2f}  n_toy={etoy_i['n_tom']}")
print("Neither clears t=2. The turn-of-year effect in bonds is absent.")
"""


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Bond-Seasonality\n"
            "### Do Treasuries have their own calendar - a turn-of-year or month-end bid?\n\n"
            + HERO_BADGES +
            "The equity world has a well-documented turn-of-month (TOM) premium. Does a "
            "similar pattern show up in long Treasuries? The story is compelling: pension "
            "funds rebalance at month-end, coupons get reinvested, risk-off flows into bonds "
            "at the turn. We take the claim literally and test it.\n\n"
            "> This is the plain-language layer. The HAC *t*-stats, win-rate intervals, and "
            "capacity arithmetic are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n>\n"
            "> Not investment advice. Educational, reproducible research - every chart below is "
            "generated by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is there a turn-of-month effect in bonds? | **Soft yes** - TLT earns "
            f"**{R['tlt_mu_tom']:.1f} bps** on TOM days vs **{R['tlt_mu_rest']:.1f} bps** "
            f"on other days, and IEF shows a similar gap. But TLT's *t*-stat ({R['tlt_t_gap']:.2f}) "
            f"just misses the bar; IEF clears it ({R['ief_t_gap']:.2f}). **Weak** evidence. |\n"
            f"| Can you trade it? | **No** - the TOM-only book earns **{R['tlt_book_cagr']:.1f}%/yr** "
            f"(TLT) and **{R['ief_book_cagr']:.1f}%/yr** (IEF) vs **{R['tlt_bh_cagr']:.1f}%** and "
            f"**{R['ief_bh_cagr']:.1f}%** for just holding - trailing buy-and-hold on both. |\n"
            f"| Is there a turn-of-year effect? | **No** - the TOY gap is "
            f"**{R['tlt_toy_gap']:+.1f} bps** on TLT (*t* = {R['tlt_toy_t']:.2f}) "
            f"and **+{R['ief_toy_gap']:.1f} bps** on IEF (*t* = {R['ief_toy_t']:.2f}). Noise. |"
        ),
        md(
            "## 1 · The claim\n\n"
            "*\"Treasuries have their own calendar - a turn-of-month or turn-of-year bid. Buy "
            "TLT (or IEF) at the last close of the month, sell at the first close of the next "
            "month.\"* The theoretical story: institutional fixed-income rebalancing at month-end, "
            "coupon reinvestment, and risk-off portfolio flows into bonds at predictable windows."
        ),
        md(
            "## 2 · So what?\n\n"
            "If bonds reliably rallied on a handful of predictable dates, you could beat a "
            "Treasury buy-and-hold with a simple calendar rule - no yield forecasting, no credit "
            "risk, no model. That is a large and testable claim worth taking seriously."
        ),
        md(
            "## 3 · How we test it\n\n"
            "We define the **TOM window** as the last 2 + first 2 trading days of each calendar "
            "month - derived from the trading index itself, no external calendar needed. We "
            "compare mean daily returns on TOM vs all other days, run a TOM-only book against "
            "buy-and-hold, and separately test the turn-of-year (last 5 + first 5 trading days "
            "per year). Both TLT (20+ year) and IEF (7-10 year) are tested in parallel."
        ),
        md("## 4 · The teardown\n\nFirst: how big is the TOM day in bonds, really?"),
        code(FIG_BAR),
        md(
            "The gap is positive and consistent across both durations - TLT earns nearly "
            f"**{R['tlt_mu_tom']:.0f} bps** on TOM days vs **{R['tlt_mu_rest']:.1f} bps** at "
            "other times, and IEF shows a similar pattern. But the t-stats tell the truth: IEF "
            "clears the bar, TLT barely misses. This is a **soft** effect, not a clean anomaly.\n\n"
            "Now the tradability test - can the TOM-only book keep up?"
        ),
        code(FIG_CUM),
        md(
            "The TOM-only book is in the market 19% of the time - much more than an extreme "
            "calendar rule like the pre-holiday effect (4%). Yet it still **trails buy-and-hold** "
            "in absolute CAGR on both instruments. The reason is simple: Treasuries earn the term "
            "premium on every single day you hold them. A book that sits in cash 81% of the time "
            "gives up most of that premium and can not recover it from a modest TOM bump."
        ),
        md("And the turn-of-year hypothesis?"),
        code(FIG_TOY),
        md(
            "The turn-of-year effect is entirely absent in both Treasuries. The t-stats are near "
            "zero and the signs are not even consistent. The bond calendar, such as it is, is a "
            "month-end phenomenon at most - and even that is too soft to exploit."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"- **Weak TOM signal.** IEF: gap +{R['ief_gap']:.1f} bps, *t* = {R['ief_t_gap']:.2f} "
            f"(clears bar). TLT: gap +{R['tlt_gap']:.1f} bps, *t* = {R['tlt_t_gap']:.2f} "
            f"(just misses). Inconsistent across durations = Weak.\n"
            f"- **Not tradable.** TOM-only book earns {R['tlt_book_cagr']:.1f}%/yr (TLT) and "
            f"{R['ief_book_cagr']:.1f}%/yr (IEF) vs {R['tlt_bh_cagr']:.1f}% and "
            f"{R['ief_bh_cagr']:.1f}% B&H. The term premium is not concentrated in four days/month.\n"
            "- **TOY absent.** Neither TLT nor IEF shows a turn-of-year effect."
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Mechanically yes - buy at the last close of each month, sell at the first close of "
            "the next. Economically, the book trails buy-and-hold in absolute return. The Sharpe "
            "ratio on IEF TOM-only (0.66) edges above B&H (0.55), but only because the book avoids "
            "mid-month volatility - it earns less, not more. As a rule that *beats* the index in "
            "total return, it is a **Mirage**."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Does the TOM effect concentrate in specific months (January, year-end)?\n"
            "- Is there a matching *intramonth* giveback that would eat the edge?\n"
            "- Does a hedged version (long TOM Treasuries, short mid-month) reduce duration risk "
            "enough to make a risk-adjusted case? Fork it and try."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Bond-Seasonality - a quantitative teardown\n"
            "### Turn-of-month calendar · HAC inference · Wilson win-rates · capacity arithmetic\n\n"
            + HERO_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) - "
            "*same seven beats, every claim now carrying its standard error.* We separate the three "
            "questions: is the TOM gap in Treasuries **statistically real** (weak evidence), is it "
            "**tradable** (no - trails buy-and-hold in CAGR), and is there a turn-of-year effect "
            "(absent).\n\n"
            "> Not investment advice. TLT and IEF **total-return adjusted** (dividends reinvested) "
            "via quantlab. Sources in [`docs/references.md`](../docs/references.md), reproducible "
            "run in [`docs/results.md`](../docs/results.md).\n>\n"
            "> The **In plain words** notes translate each result back into intuition."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | TOM gap: TLT +{R['tlt_gap']:.1f} bps (*t* = {R['tlt_t_gap']:.2f}, "
            f"misses bar), IEF +{R['ief_gap']:.1f} bps (*t* = **{R['ief_t_gap']:.2f}**, clears bar). "
            "One instrument in, one out. TOY: absent (*t* near zero). |\n"
            f"| **Tradability** | `MIRAGE` | TOM book CAGR {R['tlt_book_cagr']:.1f}%/yr (TLT) "
            f"and {R['ief_book_cagr']:.1f}%/yr (IEF) vs {R['tlt_bh_cagr']:.1f}% and "
            f"{R['ief_bh_cagr']:.1f}% B&H. Trails in absolute return on both instruments. |\n"
            f"| **Bond calendar bid?** | `MIXED` | Month-end institutional rebalancing channel is "
            "theoretically coherent and shows a suggestive positive signal, but it is too soft and "
            "inconsistent across durations to confirm. |\n\n"
            "> In plain words: a positive TOM tilt exists in bonds at low confidence - not strong "
            "enough to bet on, not absent enough to dismiss."
        ),
        md(
            "## 1 · The hypotheses\n\n"
            "- **H1 (TOM):** TOM days earn more than other days in TLT and IEF.\n"
            "- **H2 (TOY):** Turn-of-year days earn more than other days in TLT and IEF.\n"
            "- **H3 (tradability):** A TOM-only book competes with buy-and-hold.\n\n"
            "We find H1 weakly positive (one instrument clears the bar), H2 absent, and H3 false."
        ),
        md(
            "## 2 · So what?\n\n"
            "If H3 held, a calendar lookup would beat holding Treasuries. It does not - and the "
            "*why* (term premium is diffuse, not concentrated in four days/month) is the lesson."
        ),
        md(
            "## 3 · Protocol\n\n"
            "TOM window: last 2 + first 2 trading days of each calendar month (~19% of all days). "
            "TOY window: last 5 + first 5 per year (~4%). Both derived from the trading index, no "
            "external calendar. HAC *t*-stats on the gap, Wilson win-rate intervals. TOM-only book "
            "vs buy-and-hold (total-return, net of 1 bp/leg). Both TLT and IEF independently."
        ),
        md("## 4 · The teardown\n\nEffect stats, both instruments:"),
        code(
            "tbl = pd.DataFrame({\n"
            "  'TOM bps/day': [et['mu_tom']*1e4, ei['mu_tom']*1e4],\n"
            "  'Rest bps/day': [et['mu_rest']*1e4, ei['mu_rest']*1e4],\n"
            "  'Gap bps': [et['gap']*1e4, ei['gap']*1e4],\n"
            "  'HAC t(TOM)': [et['t_tom'], ei['t_tom']],\n"
            "  'HAC t(gap)': [et['t_gap'], ei['t_gap']],\n"
            "  'Win% TOM': [et['win_tom']*100, ei['win_tom']*100],\n"
            "  'Win% rest': [et['win_rest']*100, ei['win_rest']*100],\n"
            "}, index=['TLT (20+ yr)', 'IEF (7-10 yr)'])\n"
            "tbl.round(2)"
        ),
        code(FIG_BAR),
        md(
            "> In plain words: IEF clears the t=2 bar, TLT misses by 0.01. The bar requires "
            "both instruments to agree to call this Real. They do not - hence Weak."
        ),
        md("**Wilson win-rate intervals** - up-rate on TOM vs other days:"),
        code(
            "for e, label in zip([et, ei], ['TLT', 'IEF']):\n"
            "    wt, wr = e['wilson_tom'], e['wilson_rest']\n"
            "    print(f\"{label}: TOM {e['win_tom']*100:.1f}% [{wt[0]*100:.1f}, {wt[1]*100:.1f}]  \"\n"
            "          f\"rest {e['win_rest']*100:.1f}% [{wr[0]*100:.1f}, {wr[1]*100:.1f}]\")\n"
            "print('TOM win-rates are mildly higher but the intervals overlap for TLT.')"
        ),
        md("**Capacity** - TOM-only book vs buy-and-hold:"),
        code(FIG_CUM),
        md(
            "> In plain words: the TOM book avoids mid-month bond volatility (higher Sharpe for "
            "IEF), but it earns less in absolute return. The term premium is not concentrated in "
            "four days per month - it accrues daily. Sitting in cash for 81% of the tape destroys "
            "more in foregone carry than the TOM bump can recover."
        ),
        md("**Turn-of-year** - is there a January / year-end bond bid?"),
        code(FIG_TOY),
        md(
            "> In plain words: the TOY effect in bonds is absent and directionally inconsistent. "
            "TLT shows a slightly *negative* gap at year-turn; IEF shows a small positive one. "
            "Neither is close to the inference bar. This hypothesis is rejected."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"Signal `WEAK` (IEF HAC *t* {R['ief_t_gap']:.2f}, TLT {R['tlt_t_gap']:.2f}; TOY "
            "absent). Tradability `MIRAGE` (TOM book trails B&H in CAGR on both instruments). "
            "Bond calendar bid? `MIXED` - the month-end channel is theoretically coherent but "
            "the signal is too soft and duration-inconsistent to confirm."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            "The strategy is mechanically simple (buy TLT/IEF at month-end close, sell at "
            "first-of-month close, repeat). The economics are unfavorable: Treasury buy-and-hold "
            "already earns the term premium continuously, and a TOM-only book gives up 81% of "
            "that carry without recovering it from the TOM bump. Net of any realistic friction "
            "the gap to buy-and-hold only widens."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Sub-period analysis: is the TOM gap stable across the rate cycles (2002-2007 "
            "flat, 2008-2020 low-rate, 2022+ hiking)?\n"
            "- A hedged book (long TOM, short mid-month via futures) eliminates duration risk "
            "and isolates the calendar alpha - but the expected alpha is too small to survive costs.\n"
            "- Cross-country: do European or Japanese government bond ETFs show a TOM effect?\n"
            "- The theory predicts *equity selling* at month-end drives *bond buying*. Does the "
            "TOM gap in bonds correlate with the equity TOM gap?"
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


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
