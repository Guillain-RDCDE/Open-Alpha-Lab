"""Build the two narrative notebooks for Study 89 (Turn-of-the-Month).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both walk the seven desk beats at two altitudes. Figures are computed live from the
``turn_of_the_month`` package on the cached real tape; the frozen ``R`` dict mirrors
``docs/results.md`` for the verdict prose so the two never drift.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12", fp="eb1569cc9e97",
    g_n=19233, g_start="1950-01-03", g_fp="0198a54a32c0",
    tom_bps=7.68, rest_bps=4.07, diff_bps=3.61, hac_t=1.17,
    frac_days=19.1, share=32.9,
    g_tom_bps=11.02, g_rest_bps=1.93, g_diff_bps=9.08, g_hac_t=4.97, g_share=63.4,
    t_cagr=2.20, t_vol=7.9, t_sharpe=0.315, t_sharpe_inv=0.895, t_dd=-19.3, t_tim=19, t_sw=803,
    b_cagr=10.82, b_vol=18.6, b_sharpe=0.646, b_dd=-55.2,
    cagr_gap=-8.62, sharpe_gap=-0.331, sharpe_inv_edge=0.249,
    early_bps=5.77, early_t=1.26, late_bps=1.37, late_t=0.33, decay_bps=-4.40, decay_p=0.487,
    split="2010-01-01",
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study root (turn_of_the_month/)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from turn_of_the_month import data, strategy

AS_OF = "2026-06-12"
frame = data.load_real("SPY", mode="total_return").loc[:AS_OF]
close = frame["close"]
wc = strategy.window_contrast(close, last=1, first=3)
pos = strategy.tom_position(close.index, last=1, first=3)
timer = strategy.backtest(close, pos, cost_bps=5.0)
bh = strategy.buy_and_hold(close)
print(f"SPY total return: {len(close):,} rows  {close.index[0].date()} -> {close.index[-1].date()}  fingerprint={data.fingerprint(frame)}")
print(f"TOM mean {wc['tom_mean_bps']:+.2f} bps/day  rest {wc['rest_mean_bps']:+.2f} bps/day  diff {wc['diff_bps']:+.2f}  HAC t {wc['t_diff']:+.2f}")
print(f"window = {wc['frac_tom_days']*100:.1f}% of days, earns {wc['share_of_total_logret']*100:.1f}% of total return")
print(f"timer  CAGR {timer['cagr']*100:.2f}%  Sharpe {timer['sharpe']:.3f}  Sharpe(inv) {timer['sharpe_invested']:.3f}  maxDD {timer['max_dd']*100:.1f}%")
print(f"B&H    CAGR {bh['cagr']*100:.2f}%  Sharpe {bh['sharpe']:.3f}  maxDD {bh['max_dd']*100:.1f}%")
"""


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


HERO_BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Almost all the gains?: Busted](https://img.shields.io/badge/Almost_all_the_gains%3F-Busted-8b949e?style=flat-square)\n\n"
)

# Reusable figure code -----------------------------------------------------------------
FIG_CUMULATIVE = (
    "# Cumulative return split: what the TOM window contributes vs the rest of the month.\n"
    "ret = close.pct_change().fillna(0.0)\n"
    "mask = strategy.tom_mask(ret.index, 1, 3).reindex(ret.index).fillna(False).to_numpy()\n"
    "tom_only = pd.Series(np.where(mask, ret.to_numpy(), 0.0), index=ret.index)\n"
    "rest_only = pd.Series(np.where(~mask, ret.to_numpy(), 0.0), index=ret.index)\n"
    "fig, ax = plt.subplots(figsize=(10,5.5))\n"
    "ax.plot((1+ret).cumprod(), label=f\"full market (buy & hold, Sharpe {bh['sharpe']:.2f})\", lw=1.4)\n"
    "ax.plot((1+tom_only).cumprod(), label=f\"only the TOM window ({wc['frac_tom_days']*100:.0f}% of days)\", lw=1.4)\n"
    "ax.plot((1+rest_only).cumprod(), label='only the rest of the month', lw=1.4, alpha=.8)\n"
    "ax.set_yscale('log'); ax.set_ylabel('Growth of $1 (log)'); ax.legend()\n"
    "ax.set_title('SPY total return decomposed: turn-of-month vs rest-of-month'); plt.tight_layout(); plt.show()\n"
    "print(f\"TOM window earns {wc['share_of_total_logret']*100:.1f}% of the total cumulative return, in {wc['frac_tom_days']*100:.1f}% of days\")"
)

FIG_BARCHART = (
    "# Mean daily return by signed trading-day-of-month, centred on the turn.\n"
    "ret = close.pct_change().dropna()\n"
    "sd = strategy.signed_trading_day(ret.index, around=6).reindex(ret.index)\n"
    "by_day = (ret.groupby(sd).mean() * 1e4).sort_index()\n"
    "colors = ['C1' if (d <= -1 or 1 <= d <= 3) else 'C0' for d in by_day.index]\n"
    "fig, ax = plt.subplots(figsize=(10,5))\n"
    "ax.bar([str(int(d)) for d in by_day.index], by_day.values, color=colors)\n"
    "ax.axhline(ret.mean()*1e4, color='grey', ls='--', lw=1, label=f'all-day mean ({ret.mean()*1e4:.1f} bps)')\n"
    "ax.set_xlabel('trading day relative to month-end  (-1 = last day of month, +1..+3 = first days)')\n"
    "ax.set_ylabel('mean daily return (bps)'); ax.legend()\n"
    "ax.set_title('The turn-of-month bump: orange bars are the [-1,+3] window'); plt.tight_layout(); plt.show()"
)


def build_curious():
    cells = [
        md(
            "# Turn-of-the-Month 📅\n"
            "### Do almost all the market's gains really land in the ~4 days around month-end?\n\n"
            + HERO_BADGES +
            "An old desk saying: *the market makes almost all its money in a tiny window straddling "
            "the turn of the month — the last day of the old month and the first three of the new "
            "one — so just own stocks then and skip the rest.* We take it literally and ask the two "
            "questions that matter: is that window **really** where the money is (**partly** — about "
            "a third of the return in a fifth of the days), and could you **get richer** owning only "
            "those days (**no** — far poorer).\n\n"
            "> 📓 **This is the plain-language layer.** The statistics, the long-history sample and "
            "the per-day-invested maths are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** "
            "— same story, deeper.\n>\n"
            "> ⚠️ **Not investment advice.** Educational, reproducible research — every chart below is "
            "generated by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first 🎯\n\n"
            f"| Question | Answer |\n|---|---|\n"
            f"| Are the gains concentrated around the turn? | **Partly** — the {R['frac_days']:.0f}% of "
            f"days in the window earn **{R['share']:.0f}%** of the total return (~1.7x their share). |\n"
            f"| Is it *almost all* the gains, as the saying goes? | **No** — the other ~80% of days "
            f"still earn most of the return. |\n"
            f"| Could you get richer owning only the window? | **No** — a TOM-only timer makes "
            f"**{R['t_cagr']}%/yr** vs **{R['b_cagr']}%/yr** for just holding — about "
            f"**{abs(R['cagr_gap']):.0f} points/yr less**. |"
        ),
        md(
            "## 1 · The claim\n\n"
            "*\"Almost all of the market's gains happen in a narrow window around the turn of the "
            "month — the last trading day plus the first three of the next. Be invested only then and "
            "you capture the market with a fraction of the risk.\"* It traces back to real academic "
            "papers (Ariel 1987; Lakonishok & Smidt 1988), which is what makes it worth a fair test."
        ),
        md(
            "## 2 · So what?\n\n"
            "If true, it would be a free lunch: the same upside for a quarter of the time exposed to "
            "the market, dodging four-fifths of the risk. Every saver should run it. The stakes are "
            "exactly that big — so it deserves a literal, honest test."
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "The window is set purely by the **calendar** — the last trading day of a month and the "
            "first three of the next — so it's knowable in advance and there's **no guessing and no "
            "peeking** (a calendar rule isn't a forecast). We measure how much of the total return "
            "lands inside it, then run a strategy that owns SPY only during the window (cash, earning "
            "0%, otherwise; 5 bps a switch) against simply holding."
        ),
        md("## 4 · The teardown — let's actually look\n\nFirst: split the market's growth into the "
           "part earned **inside** the window and the part earned the **rest** of the month."),
        code(FIG_CUMULATIVE),
        md(
            f"The TOM window pulls **well above its weight** — {R['share']:.0f}% of the return from "
            f"{R['frac_days']:.0f}% of the days — but the *rest of the month* line still climbs a long "
            "way. This is real concentration, not *\"almost all the gains.\"*"
        ),
        md("Now the same thing as a bar chart: average daily return by where the day sits relative to "
           "month-end. The orange bars are the `[-1, +3]` window."),
        code(FIG_BARCHART),
        md(
            "The last day of the month and the first one or two of the next clearly punch above the "
            "grey all-day average. The effect is **real** — but look at the size: a few extra basis "
            "points a day, not a different universe."
        ),
        md("So does owning *only* the window beat just holding? Here are the two strategies head to head."),
        code(
            "tbl = pd.DataFrame({\n"
            "  'CAGR %':  [timer['cagr']*100, bh['cagr']*100],\n"
            "  'Vol %':   [timer['vol']*100, bh['vol']*100],\n"
            "  'Sharpe':  [timer['sharpe'], bh['sharpe']],\n"
            "  'MaxDD %': [timer['max_dd']*100, bh['max_dd']*100],\n"
            "  'Days invested %': [timer['time_in_market']*100, bh['time_in_market']*100],\n"
            "}, index=['TOM-only timer', 'Buy & hold']).round(2)\n"
            "print(tbl)\n"
            "print(f\"\\nThe timer trails buy-and-hold by {(bh['cagr']-timer['cagr'])*100:.1f} points a year.\")"
        ),
        md(
            "## 5 · The verdict\n\n"
            f"- **Almost all the gains? → Busted.** {R['share']:.0f}% of the return in "
            f"{R['frac_days']:.0f}% of the days is striking, but a long way from *\"almost all.\"*\n"
            f"- **Signal → Weak.** On this modern dividend-inclusive tape the daily bump (+"
            f"{R['diff_bps']:.1f} bps) isn't statistically distinguishable from noise (HAC *t* = "
            f"{R['hac_t']}). Over deep history it is real (see the quants notebook).\n"
            f"- **Beats the market? → Mirage.** Owning only the window makes **{abs(R['cagr_gap']):.0f} "
            "pts/yr less**, because you're in cash 80% of the time and miss the rest of the rally."
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You *can* — it's just SPY, infinitely liquid. But the product is a quarter of the upside "
            "for a quarter of the exposure: the window's per-day-invested Sharpe is genuinely high, "
            "yet you can't compound what you don't hold. Twelve months a year of entering and exiting "
            "(~800 switches over the sample) adds costs and taxes on top. As *\"capture the market "
            "with less risk,\"* it's a mirage — a 25/75 stock/cash blend gets you there more cheaply."
        ),
        md(
            "## 7 · Going further 🚪\n\n"
            "- Does paying the **T-bill** on the 80% of days in cash close the gap? (It helps — it's "
            "still less compounding than holding.)\n"
            "- Different windows (`[-1,+4]`, first-half-of-month à la Ariel) — wider or narrower?\n"
            "- Is the bump bigger in **small caps** or other countries? Fork it and try."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Turn-of-the-Month — a quantitative teardown 🔬\n"
            "### Calendar-known window · HAC inference · total-return vs price-only · per-day-invested Sharpe · decay test\n\n"
            + HERO_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We separate the two things the "
            "turn-of-month does — **concentrate** return (real) and **out-earn the rest of the month "
            "as a tradable timer** (it doesn't) — and show why the per-day-invested Sharpe edge can't "
            "be banked.\n\n"
            "> ⚠️ **Not investment advice.** SPY daily, total-return adjusted (`quantlab.data`, Yahoo) "
            "is the headline; `^GSPC` price-only (no dividends) is the long sample, labelled as such. "
            "Cash earns 0%; 5 bps/switch; the window is **calendar-known so there is no execution "
            "lag**. Sources in [`docs/references.md`](../docs/references.md), reproducible run in "
            "[`docs/results.md`](../docs/results.md).\n>\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | TOM-minus-rest **+{R['diff_bps']:.1f} bps/day**, but HAC *t* = "
            f"**{R['hac_t']}** (\\|t\\| < 2) on SPY total-return (1993–). The long price-only `^GSPC` "
            f"(1950–) clears it (*t* = **{R['g_hac_t']}**) — real over deep history, uncertified here. |\n"
            f"| **Tradability** | `MIRAGE` | TOM-only timer **{R['t_cagr']}%/yr vs {R['b_cagr']}%/yr** "
            f"(−{abs(R['cagr_gap']):.1f} pts/yr); the per-day-invested Sharpe edge (+{R['sharpe_inv_edge']:.2f}) "
            f"can't be compounded out of cash. |\n"
            f"| **Almost all the gains?** | `BUSTED` | Window = **{R['frac_days']:.0f}%** of days, earns "
            f"**{R['share']:.0f}%** of total return — ~1.7x, not \"almost all.\" |\n\n"
            "> 💡 **In plain words:** the turn-of-month is a *real bump* wearing the costume of a "
            "*free lunch*."
        ),
        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (concentration):** the `[-1,+3]` window earns far more than its day-share of return.\n"
            "- **H₂ (significance):** the daily TOM-minus-rest premium is real (HAC *t* > 2).\n"
            "- **H₃ (the sold claim):** owning only the window beats buy-and-hold on a usable basis.\n\n"
            "H₁ is clearly true; H₂ is true over deep history but **not** on the modern total-return "
            "tape; H₃ is false. That split is the whole result."
        ),
        md("## 2 · So what? — what rides on each answer\n\nIf H₃ held, a saver could match the market on "
           "a quarter of the risk. It doesn't — and seeing *why* (you can't compound days you don't "
           "hold) is the lesson."),
        md("## 3 · How we'd know — the protocol\n\nMeasure the share of total return in the window · HAC "
           "*t* on the TOM-minus-rest difference (dummy-regression slope) · total-return vs price-only "
           "samples · per-day-invested Sharpe · a block-bootstrap early-vs-late decay test on a "
           "justified split."),
        md("## 4 · The teardown\n\nThe concentration first — cumulative return, window vs rest:"),
        code(FIG_CUMULATIVE),
        md("Mean daily return by signed trading-day-of-month (the `[-1,+3]` window in orange):"),
        code(FIG_BARCHART),
        md("**HAC inference** on the TOM-minus-rest daily difference — on both the headline total-return "
           "tape and the long price-only history:"),
        code(
            "print(f\"SPY total return (1993-): diff {wc['diff_bps']:+.2f} bps/day  HAC t {wc['t_diff']:+.2f}\")\n"
            "g = data.load_real('^GSPC', start='1950-01-01', mode='split_only').loc[:AS_OF]['close']\n"
            "gwc = strategy.window_contrast(g, 1, 3)\n"
            "print(f\"^GSPC price-only (1950-): diff {gwc['diff_bps']:+.2f} bps/day  HAC t {gwc['t_diff']:+.2f}\")\n"
            "print(f\"^GSPC window earns {gwc['share_of_total_logret']*100:.1f}% of price return in {gwc['frac_tom_days']*100:.1f}% of days\")\n"
            "print('\\n=> real and strongly significant over 76 years of price history; sub-2 on the modern total-return tape.')"
        ),
        md(
            "> 💡 **In plain words:** the bump is real and big over the long run, but on the recent, "
            "dividend-inclusive sample the day-to-day noise is large enough that this tape alone can't "
            "certify it. *(Price-only `^GSPC` has no dividends — it understates the rest-of-month drift, "
            "which flatters the window's share; never call it total return.)*"
        ),
        md("**Per-day-invested Sharpe** — the only comparison on which the timer looks good (normalise "
           "before you marvel):"),
        code(
            "print(f\"TOM-only timer : Sharpe(all days) {timer['sharpe']:.3f}   Sharpe(per-day-invested) {timer['sharpe_invested']:.3f}\")\n"
            "print(f\"Buy & hold     : Sharpe          {bh['sharpe']:.3f}\")\n"
            "print(f\"\\nPer-day-invested, the window beats buy-and-hold by {timer['sharpe_invested']-bh['sharpe']:+.3f} Sharpe.\")\n"
            "print(f\"But whole-period the timer trails by {timer['sharpe']-bh['sharpe']:+.3f} Sharpe and {(timer['cagr']-bh['cagr'])*100:+.1f} pts/yr CAGR\")\n"
            "print('-- because it is in cash ~81% of the time and forgoes the equity premium.')"
        ),
        md(
            "> 💡 **In plain words:** each *invested* day in the window is a better day than the average "
            "buy-and-hold day. But you only get ~50 of them a year; the missing 200 days of equity "
            "premium are what sink the timer."
        ),
        md("**Decay test** — has the premium faded? Early vs late half, block-bootstrap on the difference:"),
        code(
            "sp = strategy.subperiod_contrast(close, split='" + R['split'] + "', last=1, first=3)\n"
            "print(f\"early ({sp['n_early']:,} d): {sp['early_diff_bps']:+.2f} bps/day  HAC t {sp['early_t']:+.2f}\")\n"
            "print(f\"late  ({sp['n_late']:,} d): {sp['late_diff_bps']:+.2f} bps/day  HAC t {sp['late_t']:+.2f}\")\n"
            "print(f\"change (late - early) = {sp['delta_bps']:+.2f} bps/day   block-bootstrap p = {sp['p_delta']:.3f}\")\n"
            "print('=> the premium points down, but the difference is NOT significant -- we say \\'cannot certify\\', not \\'decayed\\'.')"
        ),
        md(
            "> 💡 **In plain words:** the late-sample bump is smaller, but daily returns are noisy "
            "enough that we can't rule out chance — so we report the drop honestly and decline to "
            "stamp it *\"decayed.\"*"
        ),
        md(
            "## 5 · The verdict\n\n"
            f"Signal `WEAK` (SPY TR diff +{R['diff_bps']:.1f} bps, HAC *t* {R['hac_t']}; `^GSPC` "
            f"price-only *t* {R['g_hac_t']}), Tradability `MIRAGE` (−{abs(R['cagr_gap']):.1f} pts/yr; "
            f"per-day-invested Sharpe edge +{R['sharpe_inv_edge']:.2f} can't be banked), Almost all the "
            f"gains? `BUSTED` ({R['share']:.0f}% of return in {R['frac_days']:.0f}% of days)."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            "Capacity is a non-issue (SPY). The binding fact is economic: a TOM-only book is a "
            "low-exposure portfolio, and you cannot compound the equity premium on days you sit in "
            "cash. ~800 switches over the sample add cost and taxes. A 25/75 stock/cash blend "
            "reproduces the risk profile without the churn. Net of friction the gap to buy-and-hold "
            "only widens."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Replace 0% cash with the **T-bill** (`^IRX`) and recompute excess-of-cash for both arms.\n"
            "- Sweep the window definition (`[-1,+2]`, `[-1,+4]`, Ariel's first-half-of-month).\n"
            "- Cross-section: is the bump larger in small caps / international indices, where "
            "month-end flows are a bigger share of volume?"
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
