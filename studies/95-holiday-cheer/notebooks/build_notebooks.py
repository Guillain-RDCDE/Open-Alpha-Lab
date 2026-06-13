"""Build the two narrative notebooks for Study 95 (Holiday-Cheer).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both walk the seven desk beats at two altitudes. Figures are computed live from the
``holiday_cheer`` package on the cached real tapes; the frozen ``R`` dict mirrors
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
    # SPY total-return (fair tape, 1993+)
    spy_n=8400, spy_fp="eb1569cc9e97", spy_pre=301,
    spy_mu_pre=9.60, spy_mu_rest=4.58, spy_ratio=2.1, spy_gap=5.02,
    spy_t_pre=1.94, spy_t_gap=0.98, spy_win_pre=57.1, spy_win_rest=54.0,
    spy_book_cagr=0.65, spy_bh_cagr=10.82,
    # ^GSPC price-only (long sample, 1950+)
    g_n=19233, g_fp="0198a54a32c0", g_pre=702,
    g_mu_pre=18.89, g_mu_rest=3.09, g_ratio=6.1, g_gap=15.80,
    g_t_pre=6.21, g_t_gap=5.05, g_win_pre=64.2, g_win_rest=52.7,
    g_book_cagr=1.54, g_book_sharpe_inv=3.571, g_bh_cagr=8.32,
    g_book_days=702, g_book_pct=3.6,
    # decay (^GSPC price-only)
    gap_early=25.13, t_early=7.25, n_pre_early=377,
    gap_late=5.02, t_late=1.96, n_pre_late=325,
    decay=20.11, p_decay=0.999, decay_ci_lo=6.78, decay_ci_hi=32.76,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study root (holiday_cheer/)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from holiday_cheer import data, strategy

AS_OF = "2026-06-12"
# Long price-only tape for the magnitude + decay story; total-return SPY for the fair compare.
g = data.load_real("^GSPC", start="1950-01-01", mode="split_only").loc[:AS_OF]
gclose = g["close"]; gpre = data.pre_holiday_mask(gclose.index)
spy = data.load_real("SPY", mode="total_return").loc[:AS_OF]
sclose = spy["close"]; spre = data.pre_holiday_mask(sclose.index)
ge = strategy.effect_stats(gclose, gpre)
se = strategy.effect_stats(sclose, spre)
print(f"^GSPC price-only: {len(gclose):,} rows  {gclose.index[0].date()} -> {gclose.index[-1].date()}  fp={data.fingerprint(g)}")
print(f"  pre-holiday {ge['mu_pre']*1e4:.2f} bps vs rest {ge['mu_rest']*1e4:.2f} bps  ratio {ge['ratio']:.1f}x  HAC t(gap)={ge['t_gap']:.2f}")
print(f"SPY total-return: {len(sclose):,} rows  fp={data.fingerprint(spy)}  pre {se['mu_pre']*1e4:.2f} vs {se['mu_rest']*1e4:.2f} bps  t(gap)={se['t_gap']:.2f}")
"""


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


HERO_BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Faded since 1990?: Confirmed](https://img.shields.io/badge/Faded_since_1990%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

# --- shared live figures ---------------------------------------------------
FIG_BAR = (
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "labels = ['Pre-holiday', 'Every other day']\n"
    "vals = [ge['mu_pre']*1e4, ge['mu_rest']*1e4]\n"
    "bars = ax.bar(labels, vals, color=['C3', 'C0'])\n"
    "for b, v in zip(bars, vals):\n"
    "    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f} bps', ha='center', va='bottom')\n"
    "ax.set_ylabel('Mean daily return (bps)')\n"
    "ax.set_title(f\"S&P 500 (price-only, 1950+): pre-holiday day is {ge['ratio']:.1f}x the normal day\")\n"
    "plt.tight_layout(); plt.show()\n"
    "print(f\"gap = {ge['gap']*1e4:+.2f} bps   HAC t(gap) = {ge['t_gap']:+.2f}   win-rate {ge['win_pre']*100:.0f}% vs {ge['win_rest']*100:.0f}%\")"
)

FIG_CUM = (
    "book = strategy.pre_holiday_only(gclose, gpre, cost_bps=1.0)\n"
    "bh = strategy.buy_and_hold(gclose)\n"
    "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
    "ax.plot(bh['equity'], label=f\"Buy & hold ({bh['cagr']*100:.1f}%/yr)\", lw=1.3)\n"
    "ax.plot(book['equity'], label=f\"Pre-holiday-only ({book['cagr']*100:.1f}%/yr, in mkt {book['time_in_market']*100:.0f}%)\", lw=1.3, color='C3')\n"
    "ax.set_yscale('log'); ax.set_ylabel('Growth of $1 (log)')\n"
    "ax.set_title('A pre-holiday-only book vs simply holding the index (price-only)'); ax.legend()\n"
    "plt.tight_layout(); plt.show()\n"
    "print(f\"pre-only days invested = {book['days_invested']} ({book['time_in_market']*100:.1f}% of tape); \"\n"
    "      f\"per-invested-day Sharpe = {book['sharpe_invested']:.2f}, but whole-tape CAGR {book['cagr']*100:.2f}% vs B&H {bh['cagr']*100:.2f}%\")"
)

FIG_DECAY = (
    "c = strategy.subperiod_contrast(gclose, gpre, split_date='1990-01-01')\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "labels = ['pre-1990', 'post-1990']\n"
    "vals = [c['gap_early']*1e4, c['gap_late']*1e4]\n"
    "bars = ax.bar(labels, vals, color=['C2', 'C1'])\n"
    "for b, v, t, n in zip(bars, vals, [c['t_early'], c['t_late']], [c['n_pre_early'], c['n_pre_late']]):\n"
    "    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f} bps\\n(t={t:.1f}, n={n})', ha='center', va='bottom')\n"
    "ax.axhline(0, color='k', lw=.6)\n"
    "ax.set_ylabel('Pre-holiday minus rest (bps/day)')\n"
    "ax.set_title('The fade: pre-holiday gap before vs after Ariel (1990)')\n"
    "plt.tight_layout(); plt.show()\n"
    "print(f\"decay = {c['decay']*1e4:.2f} bps   P(decay>0) = {c['p_decay_gt0']:.3f}   \"\n"
    "      f\"95% CI [{c['decay_ci'][0]*1e4:.1f}, {c['decay_ci'][1]*1e4:.1f}] bps\")"
)


def build_curious():
    cells = [
        md(
            "# Holiday Cheer 🦃\n"
            "### Do stocks really pop the day before every market holiday - and can you buy it?\n\n"
            + HERO_BADGES +
            "One of the oldest market superstitions: shares drift up on the trading day *before* a "
            "holiday - the 'pre-holiday effect' - so you should buy the day before Thanksgiving, the "
            "Fourth of July, Christmas. We take it literally and ask the two questions that matter: is "
            "the pre-holiday day really special (**it was - 6x the normal day**), and can you turn it "
            "into money (**no - there are only ~9 of them a year, and they stopped being special after "
            "1990**).\n\n"
            "> 📓 **This is the plain-language layer.** The HAC *t*-stats, the win-rate intervals and "
            "the bootstrap decay test are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** - "
            "same story, deeper.\n>\n"
            "> ⚠️ **Not investment advice.** Educational, reproducible research - every chart below is "
            "generated by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the day before a holiday really special? | **Yes (historically)** - over 1950+ it "
            f"earned **{R['g_mu_pre']:.1f} bps** vs **{R['g_mu_rest']:.1f} bps** on a normal day, about "
            f"**{R['g_ratio']:.0f}x** as much. |\n"
            f"| Can you trade it? | **No** - there are only ~9 pre-holiday days a year; a buy-the-day-"
            f"before book earns **{R['g_book_cagr']:.1f}%/yr** vs **{R['g_bh_cagr']:.1f}%/yr** for just "
            f"holding the index. |\n"
            f"| Does it still work? | **Barely** - the gap fell from **{R['gap_early']:.0f} bps** before "
            f"1990 to **{R['gap_late']:.0f} bps** after. It faded right when it got famous. |"
        ),
        md(
            "## 1 · The claim\n\n"
            "*\"Stocks reliably drift up the trading day **before** a market holiday - several times the "
            "normal daily average. Buy the day before every holiday.\"* It goes back to the 1930s and was "
            "made famous by Ariel's 1990 paper *High Stock Returns before Holidays*."
        ),
        md(
            "## 2 · So what?\n\n"
            "If a handful of dates on the calendar reliably printed money, you wouldn't need to forecast "
            "anything - you'd just mark your diary. That's a big enough claim to deserve a fair, literal "
            "test rather than a cherry-picked good year."
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "Here's the neat trick: we don't download a holiday calendar at all. We look at the market's "
            "**own trading days** - if a weekday is missing from the trading record (and it isn't a "
            "weekend), the exchange was closed: that's a holiday. The trading day right before it is a "
            "*pre-holiday* day. Then we compare the average pre-holiday return to every other day, see if "
            "a buy-the-day-before book keeps up with the index, and check whether the effect held up after "
            "1990."
        ),
        md("## 4 · The teardown - let's actually look\n\nFirst: how big is the pre-holiday day, really?"),
        code(FIG_BAR),
        md(
            "Six times the normal day, and it's *up* 64% of the time vs 53% otherwise. That's a real, "
            "eye-catching gap. Now the catch - **how often does it actually happen?**"
        ),
        code(FIG_CUM),
        md(
            "There it is. The pre-holiday day is a *fantastic* day to be invested - but you're only in the "
            "market **3.6% of the time**, so the book crawls along far below buy-and-hold. A brilliant bet "
            "you can almost never place isn't a strategy."
        ),
        md("And the last twist - did the effect even survive being published?"),
        code(FIG_DECAY),
        md(
            f"The pre-holiday premium was **{R['gap_early']:.0f} bps** before 1990 and only "
            f"**{R['gap_late']:.0f} bps** after - the classic story of an anomaly that fades once everyone "
            "knows about it. (The bootstrap says that drop is real, not noise.)"
        ),
        md(
            "## 5 · The verdict\n\n"
            f"- **Real, historically.** {R['g_ratio']:.0f}x the normal day across 1950+, up 64% of the "
            "time - not a fluke.\n"
            f"- **But you can't trade it.** ~9 days a year -> {R['g_book_cagr']:.1f}%/yr vs "
            f"{R['g_bh_cagr']:.1f}%/yr for the index.\n"
            f"- **And it faded.** {R['gap_early']:.0f} bps -> {R['gap_late']:.0f} bps after 1990."
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Mechanically, yes - buy SPY at the close two days before a holiday, sell at the next close. "
            "Economically, no: nine days a year can't compound into a competitive return no matter how "
            "good each one is, and the modern (post-1990) pre-holiday day no longer reliably pays. As "
            "*\"a rule that beats the market,\"* it's a mirage."
        ),
        md(
            "## 7 · Going further 🚪\n\n"
            "- Does the effect concentrate in *specific* holidays (Thanksgiving, July 4) more than others?\n"
            "- Is there a matching *post*-holiday giveback that would eat the edge?\n"
            "- Does it survive on small-caps or other countries' calendars? Fork it and try."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Holiday-Cheer - a quantitative teardown 🔬\n"
            "### Holiday-from-gaps calendar · HAC inference · Wilson win-rates · bootstrap decay test\n\n"
            + HERO_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) - *same seven "
            "beats, every claim now carrying its standard error.* We separate the three facts about the "
            "pre-holiday effect: it was **real and large** (long sample), it **faded after 1990** (tested), "
            "and it is **un-tradable** (capacity).\n\n"
            "> ⚠️ **Not investment advice.** ^GSPC daily **price-only** (no dividends - stated) back to "
            "1950 for magnitude + decay; SPY **total-return** (1993+) for the fair compare. The holiday "
            "calendar is derived offline from the trading index. Sources in "
            "[`docs/references.md`](../docs/references.md), reproducible run in "
            "[`docs/results.md`](../docs/results.md).\n>\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Real on 1950+ (gap **+{R['g_gap']:.1f} bps**, **{R['g_ratio']:.1f}x**, "
            f"HAC *t* = **{R['g_t_gap']:.2f}**), but carried by the early sample: pre-1990 *t* = "
            f"**{R['t_early']:.2f}**, post-1990 *t* = **{R['t_late']:.2f}**, modern SPY-TR *t* = "
            f"**{R['spy_t_gap']:.2f}**. Real, then faded. |\n"
            f"| **Tradability** | `MIRAGE` | ~9 days/yr; pre-holiday-only book **{R['g_book_cagr']:.1f}%/yr** "
            f"vs **{R['g_bh_cagr']:.1f}%** B&H, in market {R['g_book_pct']:.1f}% of the time. |\n"
            f"| **Faded since 1990?** | `CONFIRMED` | decay **{R['decay']:.1f} bps**, "
            f"P(decay>0) = **{R['p_decay']:.3f}**, 95% CI [{R['decay_ci_lo']:.1f}, {R['decay_ci_hi']:.1f}] bps. |\n\n"
            "> 💡 **In plain words:** a genuine anomaly that paid handsomely on the few days it touched - "
            "and then mostly arbitraged itself away once it was published."
        ),
        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (existence):** pre-holiday days earn more than other days.\n"
            "- **H₂ (decay):** the effect is smaller after Ariel (1990) than before.\n"
            "- **H₃ (tradability):** a pre-holiday-only book competes with buy-and-hold.\n\n"
            "We find H₁ true on the long tape, H₂ true and significant, and H₃ false - the textbook "
            "'real-but-untradable, and fading' anomaly profile."
        ),
        md("## 2 · So what?\n\nIf H₃ held, a calendar lookup would beat indexing. It doesn't - and the "
           "*why* (too few days, plus post-publication decay) is the lesson about calendar anomalies."),
        md("## 3 · How we'd know - the protocol\n\nDerive holidays from index gaps · pre-holiday vs rest "
           "with **HAC** *t* and **Wilson** win-rate intervals · pre-holiday-only book vs B&H (capacity) · "
           "**pre/post-1990** split with a **circular-block-bootstrap** test of the gap *difference*."),
        md("## 4 · The teardown\n\nThe effect, both tapes (price-only 1950+ and total-return 1993+):"),
        code(
            "tbl = pd.DataFrame({\n"
            "  'pre-holiday bps': [ge['mu_pre']*1e4, se['mu_pre']*1e4],\n"
            "  'rest bps':        [ge['mu_rest']*1e4, se['mu_rest']*1e4],\n"
            "  'ratio':           [ge['ratio'], se['ratio']],\n"
            "  'gap bps':         [ge['gap']*1e4, se['gap']*1e4],\n"
            "  'HAC t(gap)':      [ge['t_gap'], se['t_gap']],\n"
            "  'win% pre':        [ge['win_pre']*100, se['win_pre']*100],\n"
            "  'win% rest':       [ge['win_rest']*100, se['win_rest']*100],\n"
            "}, index=['^GSPC price-only 1950+', 'SPY total-return 1993+'])\n"
            "tbl.round(2)"
        ),
        code(FIG_BAR),
        md(
            "> 💡 **In plain words:** on the long tape the pre-holiday day is unmistakably special; on the "
            "modern total-return tape alone the gap is half as big and the *t* (0.98) can't certify it - "
            "which is the first hint the effect is mostly an old-sample phenomenon."
        ),
        md("**Wilson win-rate intervals** - the up-rate on pre-holiday vs other days (do the CIs overlap?):"),
        code(
            "wp, wr = ge['wilson_pre'], ge['wilson_rest']\n"
            "print(f\"^GSPC pre-holiday up-rate {ge['win_pre']*100:.1f}% [{wp[0]*100:.1f}, {wp[1]*100:.1f}]\")\n"
            "print(f\"^GSPC other-day  up-rate {ge['win_rest']*100:.1f}% [{wr[0]*100:.1f}, {wr[1]*100:.1f}]\")\n"
            "print('=> non-overlapping intervals: the pre-holiday up-rate is reliably higher (long sample).')"
        ),
        md("**Capacity** - the pre-holiday-only book vs buy-and-hold:"),
        code(FIG_CUM),
        md(
            "> 💡 **In plain words:** per *invested* day the book's Sharpe is huge (~3.6), but it's in cash "
            "96% of the time, so its whole-tape return is a small fraction of the index. Capacity, not "
            "cost, is what kills it."
        ),
        md("**The decay** - pre/post-1990 with a bootstrap test of the *difference*:"),
        code(FIG_DECAY),
        md(
            "> 💡 **In plain words:** the gap roughly quartered after the effect was published, and the "
            "bootstrap says that drop is real (P = 0.999), not sampling noise. This is the canonical "
            "'anomaly fades once arbitraged' pattern - on a split justified by the publication date, not "
            "snooped."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"Signal `MIXED` (1950+ HAC *t* {R['g_t_gap']:.2f}; pre-1990 {R['t_early']:.2f}, post-1990 "
            f"{R['t_late']:.2f}), Tradability `MIRAGE` ({R['g_book_cagr']:.1f}%/yr vs {R['g_bh_cagr']:.1f}%), "
            f"Faded since 1990? `CONFIRMED` (decay {R['decay']:.1f} bps, P {R['p_decay']:.3f}). Real, rare, "
            "and mostly gone."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            "Execution is trivial (SPY, calendar-known, no signal to peek at - the position needs **no "
            "lag**). The binding facts are capacity and decay: ~9 days a year cannot compound into a "
            "competitive return, and the modern pre-holiday day no longer clears the bar. Net of any "
            "cost the gap to buy-and-hold only widens."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Per-holiday decomposition: is the effect concentrated in a few holidays (Thanksgiving)?\n"
            "- A long-pre-holiday / short-post-holiday spread - does a giveback exist?\n"
            "- Re-derive the calendar for non-US indices and test the effect cross-country."
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
