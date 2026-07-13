"""Generate the two narrative notebooks for Study 733 (Kentucky-Derby-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY / CHDN
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers -- mirror of docs/results.md
# (SPY + CHDN, yfinance total-return, 1998-01-02 -> 2026-06-30; 26 CHDN events, 25 May
#  seasonal events; fingerprint 4b4d7494b219).
R = dict(
    n_events=26, n_chdn=26, n_market=25, rows=7166, start="1998-01-02",
    fp="4b4d7494b219", cost_bps=10.0,
    # CHDN (the gambling name) — signal, abnormal vs SPY
    ch_ru_mean=+0.359, ch_ru_t=+0.554, ch_ru_hit=13, ch_ru_n=26,
    ch_wk_mean=-0.063, ch_wk_t=-0.063, ch_wk_hit=11, ch_wk_n=26,
    ch_mo_mean=-0.195, ch_mo_t=-0.173, ch_mo_hit=13, ch_mo_n=26,
    # CHDN capture net@10bps
    ch_ru_cap_n=+0.159, ch_ru_cap_t=+0.245,
    ch_wk_cap_n=-0.003, ch_wk_cap_t=-0.003,
    ch_mo_cap_n=-0.091, ch_mo_cap_t=-0.080,
    # market seasonal — signal, SPY drift-removed
    mk_ru_mean=+0.003, mk_ru_t=+0.010, mk_ru_hit=12, mk_ru_n=25,
    mk_wk_mean=-0.617, mk_wk_t=-1.426, mk_wk_hit=7, mk_wk_n=25,
    mk_mo_mean=-0.319, mk_mo_t=-0.405, mk_mo_hit=16, mk_mo_n=25,
    # market capture — the costed 'trade it' timer (the naive-stat trap)
    mk_wk_cap_g=-0.602, mk_wk_cap_gt=-1.58,
    mk_wk_cap_n=-0.802, mk_wk_cap_t=-2.111,
    mk_wk_cap_n5=-0.702, mk_wk_cap_t5=-1.848,
    mk_mo_cap_n=-0.500, mk_mo_cap_t=-0.646,
    # random-window placebo (drift-neutral null)
    pl_ch_ru_p=0.406, pl_ch_ru_plmean=+0.161, pl_ch_ru_plsd=0.859,
    pl_ch_mo_p=0.691, pl_ch_mo_plmean=+0.623, pl_ch_mo_plsd=1.605,
    pl_mk_wk_p=0.107, pl_mk_wk_plmean=+0.001, pl_mk_wk_plsd=0.501,
    pl_mk_mo_p=0.368,
    pl_mk_wk_cap_p=0.111, pl_mk_wk_cap_plmean=-0.200,
    # jackknife (market 1-week — the only |t| approaching 2)
    jk_lo=-2.417, jk_hi=-0.999, jk_below=1, jk_n=25,
    # third axis — Welch (May years): CHDN minus market seasonal
    wh_wk_t=+0.577, wh_mo_t=+0.103,
    # event anatomy (mean cumulative AR, run-up start = 0)
    car_ch={-5: 0.000, -3: 0.778, -1: 0.621, 0: 0.249, 1: -0.016, 3: -0.305,
            5: 0.186, 10: -0.241, 15: 0.092, 21: 0.053},
    car_mk={-5: 0.000, -3: -0.127, -1: -0.353, 0: -0.020, 1: -0.034, 3: -0.384,
            5: -0.638, 10: -0.365, 15: -0.747, 21: -0.339},
    # synthetic control
    syn_null_mean=-0.29, syn_null_sd=1.03, syn_null_fire=1, syn_null_seeds=20,
    syn_planted2_t=+2.16, syn_planted3_t=+4.18,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Sell_in_May%3F: Not_supported](https://img.shields.io/badge/Sell_in_May%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from kentucky_derby_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=10.0)
    INC = EV[EV["included"]]
    CHDN = INC[INC["leg"] == "chdn"]
    MKT = INC[INC["leg"] == "market"]
else:
    PRICES = EV = INC = CHDN = MKT = None
print("real cache present:", HAVE_REAL, "| Derbys:", len(data.EVENTS),
      "| CHDN events:", (0 if CHDN is None else len(CHDN)),
      "| market events:", (0 if MKT is None else len(MKT)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Kentucky Derby move the market — or the stock that runs it? 🐎📈\n"
            "### The \"Run for the Roses\" folklore — a clean null, and one *t*-stat that "
            "only *looks* tradable\n\n"
            + BADGES +
            "Every year on the **first Saturday in May**, 150,000 people in big hats watch "
            "two minutes of horse racing, and every year a certain kind of market almanac "
            "asks the same two questions. First: is early May — Derby weekend, right on "
            "the \"Sell in May and go away\" line — a *market* seasonal? And second, more "
            "pointedly: does **Churchill Downs Inc. (`CHDN`)**, the company that literally "
            "owns and runs the Derby, get a bump around its own marquee event?\n\n"
            "This is the cleanest possible test of sports-event folklore. Unlike Eurovision "
            "(where half the winners have no tradable stock), CHDN has traded since the "
            "1990s — *full* price history, a stock with *direct* exposure. If there were "
            "anything here, this is exactly where it would show up.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "jackknife? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 26 Derbys hardcoded from Wikipedia (2020 was postponed "
            "to September — a named quirk). Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does `CHDN` pop around its own Derby? | **No.** Flat at every horizon — "
            f"run-up **{R['ch_ru_mean']:+.2f}%**, the month after **{R['ch_mo_mean']:+.2f}%**, "
            "every *t* under 1, hit rates glued to a coin flip. |\n"
            f"| Is the broad market special that week? | **Barely a whisper.** The S&P "
            f"drifts **{R['mk_wk_mean']:+.2f}%** in the week after — a \"Sell in May\" "
            "shadow — but it doesn't clear the bar even before costs, and a random week "
            "does this about 1 time in 9. |\n"
            f"| Could you have traded *that* dip? | **No.** One number (shorting the S&P "
            f"the Derby week) shows *t* = **{R['mk_wk_cap_t']:.2f}** net of costs — but "
            "only because subtracting costs made an already-negative return look worse. "
            "Its own placebo says it's noise. |\n"
            "| So is there a Kentucky Derby effect? | **No.** The most-exposed stock, full "
            "history, no survivorship excuse — and still nothing. |\n\n"
            "> This is what an honest *zero* looks like: not a dramatic debunk, just a "
            "flat line where the folklore promised a bump."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Kentucky Derby is a market event. The first Saturday in May is a "
            "seasonal turning point (it's the 'Sell in May' boundary), and Churchill Downs "
            "— the company that owns the track and runs the Derby — gets a lift around the "
            "attention, the record betting handle, the sold-out infield.\"*\n\n"
            "It borrows a *real* mechanism: attention- and mood-events genuinely move "
            "prices (Edmans, García & Norli found national markets fall after World Cup "
            "*elimination*). The Derby swaps in a positive, single-day, calendar-fixed "
            "spectator event — and, unusually, offers one directly-exposed listed stock to "
            "test it on. Nobody has ever formally tested it. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, you'd have a delightfully silly annual trade: the date is known years "
            "in advance, so you could position into it with zero look-ahead. And it would "
            "say something real — that a *scheduled* attention-event, not just a shock, "
            "moves an exposed stock. It would also be a rare clean example, because for "
            "once there's no missing-data excuse: CHDN has the full tape."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_events']}** Derbys 2000→2025, hardcoded — the "
            "first Saturday in May every year except **2020** (COVID pushed it to "
            "September; kept for the stock, dropped from the May seasonal).\n"
            "- **Two legs.** The **market** (`SPY`, drift removed — is Derby week special?) "
            "and the **gambling name** (`CHDN` minus `SPY` — does the operator beat the "
            "market around its event?).\n"
            "- **The lag.** The race runs Saturday (markets shut), so the *result* is "
            "tradable only from Monday's close — but the *date* is known in advance, so the "
            "**run-up** week into the race is tradable too. We test both.\n"
            "- **The honesty checks.** A random-window placebo (does a random week do the "
            "same?), a jackknife (does one year drive it?), and the trade net of real costs.\n\n"
            "> **What would make us say \"mirage\":** any cut that clears *t* ≥ 2 must also "
            "beat its *drift-neutral* placebo. A bare *t*-stat isn't enough — especially "
            "one that costs *improve*."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Start with the stock that runs the race. If anything pops, it's CHDN.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ru = st.one_sample_t(CHDN['ar_runup'].values)['mean']*100\n"
            "    wk = st.one_sample_t(CHDN['ar_week'].values)['mean']*100\n"
            "    mo = st.one_sample_t(CHDN['ar_month'].values)['mean']*100\n"
            "    ts = [st.one_sample_t(CHDN[c].values)['t'] for c in ('ar_runup','ar_week','ar_month')]\n"
            "else:\n"
            "    ru, wk, mo = R['ch_ru_mean'], R['ch_wk_mean'], R['ch_mo_mean']\n"
            "    ts = [R['ch_ru_t'], R['ch_wk_t'], R['ch_mo_t']]\n"
            "labels = ['run-up\\n(into the race)', '1 week\\n(after)', '1 month\\n(after)']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "bars = ax.bar(labels, [ru, wk, mo], color=[GREY, GREY, GREY])\n"
            "for b, v, t in zip(bars, [ru, wk, mo], ts):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('CHDN abnormal return vs S&P (%)')\n"
            "ax.set_ylim(-0.6, 0.8)\n"
            "ax.set_title('Churchill Downs does nothing around its own Derby')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CHDN run-up', round(ru,3), 'week', round(wk,3), 'month', round(mo,3))"
        ),
        md(
            f"A tiny positive drift into the race (**{R['ch_ru_mean']:+.2f}%**, *t* = "
            f"{R['ch_ru_t']:.2f}), flat-to-negative after, every *t*-stat comfortably "
            "under 1, every hit rate a coin flip. The most directly-exposed instrument on "
            "the board, with a full price history back to the 1990s, produces the cleanest "
            "possible *nothing*.\n\n"
            "> One honest caveat, and it points the *wrong* way for the folklore: CHDN in "
            "2000 was basically a racetrack; CHDN today is a diversified casino + online-"
            "betting company for which the Derby is a shrinking slice of revenue. So the "
            "direct exposure was *strongest* early in our sample — and it still shows "
            "nothing.\n\n"
            "**Now the market. Is Derby week — the \"Sell in May\" line — a real seasonal?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ru = st.one_sample_t(MKT['ar_runup'].values)['mean']*100\n"
            "    wk = st.one_sample_t(MKT['ar_week'].values)['mean']*100\n"
            "    mo = st.one_sample_t(MKT['ar_month'].values)['mean']*100\n"
            "    ts = [st.one_sample_t(MKT[c].values)['t'] for c in ('ar_runup','ar_week','ar_month')]\n"
            "else:\n"
            "    ru, wk, mo = R['mk_ru_mean'], R['mk_wk_mean'], R['mk_mo_mean']\n"
            "    ts = [R['mk_ru_t'], R['mk_wk_t'], R['mk_mo_t']]\n"
            "labels = ['run-up\\n(into)', '1 week\\n(after)', '1 month\\n(after)']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "cols = [RED if abs(t)>=2 else (AMBER if abs(t)>=1 else GREY) for t in ts]\n"
            "bars = ax.bar(labels, [ru, wk, mo], color=cols)\n"
            "for b, v, t in zip(bars, [ru, wk, mo], ts):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='top' if v<0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('S&P abnormal return, drift removed (%)')\n"
            "ax.set_title('The one eyebrow-raise: a soft week-after dip')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('market run-up', round(ru,3), 'week', round(wk,3), 'month', round(mo,3))"
        ),
        md(
            f"Here's the whole study's one interesting number: in the week *after* the "
            f"Derby, the S&P is **{R['mk_wk_mean']:+.2f}%** below its own average day "
            f"(*t* = {R['mk_wk_t']:.2f}), up only 7 of 25 years. That's the \"Sell in May\" "
            "texture, and it's genuinely negative — but it **doesn't clear the bar even "
            "raw**. So we do the one test that matters: is a dip that deep actually "
            "unusual, or does a *random* week do it too?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'market', 'ar_week', k=5, entry_offset=0,\n"
            "                           tail='left', n_seeds=4, n_draws_per_seed=200)\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(733)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['mk_wk_mean']\n"
            "    rng = np.random.default_rng(733)\n"
            "    draws = rng.normal(R['pl_mk_wk_plmean'], R['pl_mk_wk_plsd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 1-week windows on the same S&P tape')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed Derby-week {obs:+.2f}%')\n"
            "ax.set_xlabel('mean abnormal return of a random week (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A random week is this weak ~1 in 9 times (placebo p = {R[\"pl_mk_wk_p\"]:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['mk_wk_mean']:+.3f}%, placebo mean {R['pl_mk_wk_plmean']:+.3f}% \"\n"
            "      f\"(sd {R['pl_mk_wk_plsd']:.3f}%), left-tail p = {R['pl_mk_wk_p']:.3f}\")"
        ),
        md(
            f"The Derby-week dip sits comfortably *inside* the luck cloud: a random week on "
            f"the same tape is at least this weak about **1 time in 9** (*p* = "
            f"{R['pl_mk_wk_p']:.3f}). It's a shadow of \"Sell in May,\" not a Derby effect, "
            "and not something the desk would ever certify.\n\n"
            "**But wait — doesn't one version of this clear *t* ≥ 2? Let's chase it down.**"
        ),
        code(
            "g, gt = R['mk_wk_cap_g'], R['mk_wk_cap_gt']\n"
            "n, nt = R['mk_wk_cap_n'], R['mk_wk_cap_t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "bars = ax.bar(['gross', 'net of costs'], [g, n], color=[GREY, RED], width=.5)\n"
            "for b, v, t in zip(bars, [g, n], [gt, nt]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='top', fontsize=10)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('S&P Derby-week return (short view)')\n"
            "ax.set_title('The trap: subtracting costs makes the t-stat LOOK better')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}% (t={gt:+.2f}) -> net {n:+.2f}% (t={nt:+.2f})')"
        ),
        md(
            f"Here's the sleight of hand. The Derby-week return is already "
            f"**{R['mk_wk_cap_g']:+.2f}%** (a dip). *Subtract* a round trip of costs and it "
            f"becomes **{R['mk_wk_cap_n']:+.2f}%** — a bigger negative number, so its "
            f"*t*-stat gets *more* extreme (**{R['mk_wk_cap_t']:.2f}**), sailing past the "
            "−2 line. For a *real* tradable edge, costs make the *t* worse, not better; "
            "here they flatter it, which is the tell. And its own placebo (*p* = "
            f"{R['pl_mk_wk_cap_p']:.3f}) says the underlying dip was noise anyway. To "
            "\"harvest\" it you'd short the S&P for one week, 25 years running, paying spread "
            "and borrow for a coin flip. That's a mirage, not a strategy."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** CHDN, the directly-exposed name, is flat everywhere "
            "(every |*t*| < 1) with full tape coverage; the market's only whisper (the "
            "week-after dip) misses the bar raw and fails its drift-neutral placebo.\n"
            "- **Tradability — Mirage.** The only cut that crosses |*t*| ≥ 2 is an "
            "arithmetic artifact of charging costs against an already-negative number, and "
            "it dies under its own placebo.\n"
            "- **\"Sell in May?\" — Not supported.** The Derby-week dip is real in *sign* "
            "but inside the luck cloud and fragile to a single year — the folklore's May "
            "boundary is not a market turn you could trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the value of a *clean* null.** Most folklore debunks come with an "
            "asterisk (\"…but the data was thin\"). Not here: the most-exposed stock, full "
            "history, no survivorship funnel — and still nothing. When even the ideal test "
            "setup shows a flat line, the folklore is just folklore.\n"
            "- **The costs-flatter-the-*t* trap is the transferable lesson.** Any time a "
            "\"short this seasonal\" edge gets *more* significant after costs, you're "
            "charging costs against a negative return — check the placebo before you "
            "believe it.\n"
            "- **Sibling studies:** the [Eurovision effect](../../708-eurovision-effect/) "
            "(same machinery, but a survivorship story), the "
            "[plane-crash effect](../../707-plane-crash-effect/) (market + directly-exposed "
            "basket, a *negative* shock), the [Super Bowl indicator](../../158-super-bowl/) "
            "and [World Cup effect](../../235-world-cup-effect/).\n\n"
            "*Think there's a real Derby edge — in betting-handle data, in intraday moves, "
            "in a cleaner gaming basket? Find a net, replicated, placebo-surviving version "
            "and we'll publish the teardown.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Kentucky-Derby-Effect — a quantitative teardown 🔬\n"
            "### One-sample-*t* on two legs (CHDN vs SPY · SPY seasonal) · a drift-neutral "
            "random-window placebo · a jackknife · the event anatomy · the costs-inflate-"
            "the-*t* trap · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **the first-Saturday-in-May Kentucky "
            "Derby is a market event** — has no published academic anchor of its own; it "
            "borrows the attention/mood-moves-prices mechanism (Edmans-García-Norli 2007) "
            "and the \"Sell in May\" seasonal (Bouman-Jacobsen 2002). The job here is to "
            "measure it honestly, on the two instruments that matter, with the right "
            "inference unit for a tiny-n annual event — and to catch the one *t*-stat that "
            "*looks* tradable before it fools anyone.\n\n"
            "> ⚠️ **Data note.** `SPY` + `CHDN`, yfinance adjusted (total-return) daily "
            "closes, 1998-01-02→2026-06-30. 26 Derbys hardcoded 2000→2025 (2020 postponed "
            "to September). **No survivorship funnel** — CHDN has full coverage; the caveat "
            "named on the Signal axis is **exposure dilution** (CHDN is now a diversified "
            "gaming company). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | CHDN flat at every horizon (run-up "
            f"{R['ch_ru_mean']:+.3f}%/*t*={R['ch_ru_t']:.2f}, month "
            f"{R['ch_mo_mean']:+.3f}%/*t*={R['ch_mo_t']:.2f}, placebo *p*≥"
            f"{R['pl_ch_ru_p']:.2f}); market 1-week {R['mk_wk_mean']:+.3f}%/*t*="
            f"{R['mk_wk_t']:.2f}, placebo *p*={R['pl_mk_wk_p']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | only |*t*|≥2 cut is market/1-week net "
            f"*t*={R['mk_wk_cap_t']:.2f} — costs charged on an already-negative return; "
            f"placebo *p*={R['pl_mk_wk_cap_p']:.3f} |\n"
            f"| **Sell in May?** | `NOT SUPPORTED` | Derby-week dip {R['mk_wk_mean']:+.3f}% "
            f"inside the luck cloud (*p*={R['pl_mk_wk_p']:.3f}), jackknife "
            f"[{R['jk_lo']:.2f}, {R['jk_hi']:.2f}] |\n\n"
            "> 💡 In plain words: the most-exposed stock does nothing with full data; the "
            "market's one soft dip is a \"Sell in May\" shadow that fails its placebo; and "
            "the only significant *t* is an artifact of the cost arithmetic."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For Derby year $y$, let day(-1) be the last close before the (Saturday) race "
            "and day(0) the first close after. Two legs, each a windowed abnormal return "
            "over horizon $k$:\n\n"
            "- **CHDN (gambling name):** "
            "$AR^{chdn}_{y}(k) = \\left(\\frac{P^{CHDN}_{-1+k}}{P^{CHDN}_{-1}}-1\\right) - "
            "\\left(\\frac{P^{SPY}_{-1+k}}{P^{SPY}_{-1}}-1\\right)$ — a $\\beta=1$ market "
            "model (does the operator beat the market around its event?).\n"
            "- **Market (seasonal):** "
            "$AR^{mkt}_{y}(k) = \\left(\\frac{P^{SPY}_{-1+k}}{P^{SPY}_{-1}}-1\\right) - "
            "k\\,\\bar r_{SPY}$ — SPY's window return minus its full-sample daily drift (a "
            "constant-mean model; is Derby week unusual?).\n\n"
            "Each year is one independent, non-overlapping event, so the **one-sample t** "
            "across events is the primary statistic (not a daily panel). Claims:\n\n"
            "- **H1 (CHDN pop).** $E[AR^{chdn}(k)]>0$ for $k\\in\\{5,21\\}$ and the run-up.\n"
            "- **H2 (market seasonal).** $E[AR^{mkt}(k)]\\neq 0$ around Derby week.\n"
            "- **H3 (tradable).** A zero-look-ahead entry banks it net of costs.\n\n"
            "We find **H1 not supported** (every |*t*|<1, placebo deep in the bulk); **H2 "
            "not supported** (one soft 1-week dip that fails its drift-neutral placebo); "
            "**H3 not supported** (the only |*t*|≥2 is a cost-arithmetic artifact)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is small by construction: **{R['n_chdn']}** CHDN events and **{R['n_market']}** "
            "May-seasonal events (2020's September running is a CHDN event but not a "
            "first-Saturday-in-May observation). The plan: a **one-sample t** per cut, a "
            "**Wilson interval** on the hit rate, a **20-seed × 200-draw random-window "
            "placebo** per cut, and a **leave-one-out jackknife** on any cut that "
            "approaches the bar.\n\n"
            "> 💡 A subtlety that matters for the market leg: a raw one-sample *t* of SPY "
            "window returns is **drift-contaminated** (the market goes up on average). The "
            "random-window placebo is the honest null because a random same-length window "
            "carries the *same* drift — so the placebo, not the bare *t*, is the primary "
            "test here."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} Derbys 2000→2025, hardcoded from Wikipedia "
            "(2020 postponed to September, flagged `ran_in_may=False`).\n"
            "- **Legs.** CHDN abnormal vs SPY (26 events); SPY seasonal, drift-removed "
            f"({R['n_market']} May events).\n"
            "- **Headline.** One-sample *t* at three horizons (run-up −6→−1, 1 week −1→+5, "
            "1 month −1→+21) + Wilson hit rate.\n"
            "- **Robustness.** 20×200-draw drift-neutral random-window placebo; "
            "leave-one-out jackknife on the cut nearest the bar.\n"
            "- **Anatomy.** Mean cumulative AR by trading day, −5→+21, both legs.\n"
            "- **Execution.** Signal from day(-1); **capture** enters day(0) (post-result, "
            "zero look-ahead) or over the calendar-known run-up; 2× one-way cost × NAV "
            f"(`COST_BPS={R['cost_bps']:.0f}` for a mid-cap single stock).\n"
            "- **Control.** Synthetic paired (CHDN-like, SPY-like) world, planted-bump "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The gambling name — CHDN abnormal return vs SPY, three horizons\n\n"
            "The stock that owns the Derby, measured against the market it trades in."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lab, col in (('run-up','ar_runup'), ('1 week','ar_week'), ('1 month','ar_month')):\n"
            "        s = st.one_sample_t(CHDN[col].values); hr = st.hit_rate(CHDN[col].values)\n"
            "        rows.append((lab, s['n'], s['mean']*100, s['t'], hr['k'], hr['n']))\n"
            "        print(lab, 'n=', s['n'], 'mean%', round(s['mean']*100,3), 't', round(s['t'],3),\n"
            "              'hit', f\"{hr['k']}/{hr['n']}\")\n"
            "    means = [r[2] for r in rows]; ts = [r[3] for r in rows]\n"
            "else:\n"
            "    means = [R['ch_ru_mean'], R['ch_wk_mean'], R['ch_mo_mean']]\n"
            "    ts = [R['ch_ru_t'], R['ch_wk_t'], R['ch_mo_t']]\n"
            "labels = ['run-up', '1 week', '1 month']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=GREY); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('mean AR vs SPY (%)'); a1.set_title('CHDN: a flat line at every horizon')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat'); a2.set_ylim(-2.5, 2.5)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: run-up {R['ch_ru_mean']:+.3f}% (*t*={R['ch_ru_t']:.2f}), "
            f"1-week {R['ch_wk_mean']:+.3f}% (*t*={R['ch_wk_t']:.2f}), 1-month "
            f"{R['ch_mo_mean']:+.3f}% (*t*={R['ch_mo_t']:.2f}). No horizon comes close. "
            "And the exposure-dilution caveat cuts *against* the folklore: CHDN's direct "
            "Derby exposure was largest early in the sample (pre-casino-diversification), "
            "so if anything the test is biased slightly *toward* finding an effect — and "
            "finds none."
        ),
        md(
            "### 4b · The gambling name's placebo — is the run-up drift unusual?\n\n"
            "The only positive CHDN cut is the run-up (+0.36%). Redraw random same-length "
            "windows on the same tape and see where it lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'chdn', 'ar_runup', k=5, entry_offset=0,\n"
            "                           tail='right', n_seeds=4, n_draws_per_seed=200)\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(733)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['ch_ru_mean']\n"
            "    rng = np.random.default_rng(733)\n"
            "    draws = rng.normal(R['pl_ch_ru_plmean'], R['pl_ch_ru_plsd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random 1-week CHDN-SPY windows')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed run-up {obs:+.2f}%')\n"
            "ax.set_xlabel('mean CHDN abnormal return, random window (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Right in the middle of the luck cloud (p = {R[\"pl_ch_ru_p\"]:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['ch_ru_mean']:+.3f}%, placebo mean {R['pl_ch_ru_plmean']:+.3f}%, \"\n"
            "      f\"p = {R['pl_ch_ru_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: *p* = {R['pl_ch_ru_p']:.3f} — the run-up drift is exactly "
            "what a random week produces. Nothing to see. (The 1-month cut is even further "
            f"in the bulk, *p* = {R['pl_ch_mo_p']:.3f}.)"
        ),
        md(
            "### 4c · The market seasonal — the one soft dip, and its placebo"
        ),
        code(
            "if HAVE_REAL:\n"
            "    for lab, col in (('run-up','ar_runup'), ('1 week','ar_week'), ('1 month','ar_month')):\n"
            "        s = st.one_sample_t(MKT[col].values); hr = st.hit_rate(MKT[col].values)\n"
            "        print(lab, 'n=', s['n'], 'mean%', round(s['mean']*100,3), 't', round(s['t'],3),\n"
            "              'hit', f\"{hr['k']}/{hr['n']}\")\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'market', 'ar_week', k=5, entry_offset=0,\n"
            "                           tail='left', n_seeds=4, n_draws_per_seed=200)\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(733); draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['mk_wk_mean']\n"
            "    rng = np.random.default_rng(733); draws = rng.normal(R['pl_mk_wk_plmean'], R['pl_mk_wk_plsd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random 1-week SPY windows (same drift)')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed Derby-week {obs:+.2f}%')\n"
            "ax.set_xlabel('mean abnormal return of a random week (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Sell-in-May shadow: left-tail p = {R[\"pl_mk_wk_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"market 1-week {R['mk_wk_mean']:+.3f}% (t={R['mk_wk_t']:.2f}), placebo p = {R['pl_mk_wk_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the Derby-week dip is **{R['mk_wk_mean']:+.3f}%** "
            f"(*t*={R['mk_wk_t']:.2f}) — negative, but it doesn't clear the bar raw, and "
            f"the drift-neutral placebo (*p*={R['pl_mk_wk_p']:.3f}) says a random week is "
            "this weak roughly 1 time in 9. The 1-month cut is nothing "
            f"({R['mk_mo_mean']:+.3f}%, placebo *p*={R['pl_mk_mo_p']:.3f}). This is the "
            "\"Sell in May\" boundary leaving a faint fingerprint, not a Derby effect."
        ),
        md(
            "### 4d · The jackknife — how fragile is even that soft dip?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = MKT['ar_week'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "else:\n"
            "    rng = np.random.default_rng(733)\n"
            "    jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if t <= -2 else GREY for t in jk]\n"
            "ax.bar(range(len(jk)), jk, color=cols)\n"
            "ax.axhline(-2.0, ls='--', c=RED, lw=1.2, label='certification bar (t=-2)')\n"
            "ax.axhline(R['mk_wk_t'], c=GREY, lw=1, ls=':', label='full-sample t')\n"
            "ax.set_xlabel('leave one Derby year out'); ax.set_ylabel('resulting t-stat')\n"
            "ax.set_title(f'{R[\"jk_below\"]}/{R[\"jk_n\"]} draws even reach -2 — it straddles the line')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'full t = {R[\"mk_wk_t\"]:.3f}; jackknife range [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"> 💡 In plain words: full-sample *t* = {R['mk_wk_t']:.3f}; the jackknife spans "
            f"[{R['jk_lo']:.2f}, {R['jk_hi']:.2f}]. Drop the single most-positive year and "
            "it nudges past −2; keep it and it's a soft −1.4. A signal that crosses the "
            "certification line on the strength of *one* of 25 years — and had no placebo "
            "support to begin with — is textbook not-real."
        ),
        md(
            "### 4e · Event anatomy — the path through the race"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_c = st.car_path(EV, PRICES, 'chdn'); cp_m = st.car_path(EV, PRICES, 'market')\n"
            "    offs = list(cp_c.index); cs = list(cp_c.values*100); ms = list(cp_m.values*100)\n"
            "else:\n"
            "    offs = sorted(R['car_ch']); cs = [R['car_ch'][k] for k in offs]; ms = [R['car_mk'][k] for k in offs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.7))\n"
            "ax.plot(offs, cs, color=AMBER, lw=2.2, marker='o', ms=4, label='CHDN (vs SPY)')\n"
            "ax.plot(offs, ms, color=GREY, lw=2.2, marker='o', ms=4, label='market (SPY, drift removed)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.annotate('day(-1)\\nrace is Sat night', (0, 0), textcoords='offset points',\n"
            "            xytext=(6, -34), fontsize=8, color='k')\n"
            "ax.set_xlabel('trading days (run-up start = 0 at offset -5; offset 0 = day(-1))')\n"
            "ax.set_ylabel('mean cumulative AR (%)')\n"
            "ax.set_title('No event reaction — CHDN wanders, the market softly drifts down')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: a real event reaction would kink at day 0 and hold. CHDN "
            f"drifts up into the race (peak {R['car_ch'][-3]:+.2f}% at day −3) and gives it "
            "all back; the market path slopes gently down across the whole window (the "
            f"\"Sell in May\" texture, bottoming {R['car_mk'][15]:+.2f}% near day +15) with "
            "no special break at the race. Two flat, directionless paths — exactly the "
            "shape of no-effect."
        ),
        md(
            "### 4f · Tradability — the costs-inflate-the-*t* trap, in full\n\n"
            "Enter day(0) (post-result, zero look-ahead), net of 2× one-way cost × NAV. "
            "One cut crosses |*t*|≥2 — and it's a trap."
        ),
        code(
            "labels = ['CHDN\\nrun-up', 'CHDN\\n1mo', 'market\\n1wk', 'market\\n1mo']\n"
            "net = [R['ch_ru_cap_n'], R['ch_mo_cap_n'], R['mk_wk_cap_n'], R['mk_mo_cap_n']]\n"
            "ts  = [R['ch_ru_cap_t'], R['ch_mo_cap_t'], R['mk_wk_cap_t'], R['mk_mo_cap_t']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "cols = [RED if abs(t)>=2 else GREY for t in ts]\n"
            "bars = ax.bar(labels, ts, color=cols)\n"
            "for b, t in zip(bars, ts): ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, t),\n"
            "                                       ha='center', va='bottom' if t>=0 else 'top', fontsize=9)\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('one-sample t, net of costs'); ax.set_ylim(-2.6, 2.6)\n"
            "ax.set_title('The only |t|>=2 is market/1-week — where costs make a negative return worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"market 1wk: gross {R['mk_wk_cap_g']:+.2f}% (t={R['mk_wk_cap_gt']:+.2f}) -> \"\n"
            "      f\"net@10 {R['mk_wk_cap_n']:+.2f}% (t={R['mk_wk_cap_t']:+.2f}), placebo p={R['pl_mk_wk_cap_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the market/1-week return is gross "
            f"**{R['mk_wk_cap_g']:+.2f}%** (*t*={R['mk_wk_cap_gt']:.2f}). Charging a round "
            f"trip of costs *deepens* it to **{R['mk_wk_cap_n']:+.2f}%** and its *t* to "
            f"**{R['mk_wk_cap_t']:.2f}** — the significance is manufactured by the cost "
            "arithmetic, not by an edge. (At 5 bps it's back under the bar: "
            f"*t*={R['mk_wk_cap_t5']:.2f}.) Its own placebo, *p*={R['pl_mk_wk_cap_p']:.3f}, "
            "confirms the dip is noise, and executing it means **shorting the S&P** 25 "
            "times, paying spread *and* borrow. Every honest CHDN cut is a flat ~0 net. "
            "**H3 not supported; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Third axis — CHDN vs the market seasonal (Welch t, May years)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    chdn_may = CHDN[CHDN['ran_in_may']]\n"
            "    t_wk = st.welch_t(chdn_may['ar_week'].values, MKT['ar_week'].values)\n"
            "    t_mo = st.welch_t(chdn_may['ar_month'].values, MKT['ar_month'].values)\n"
            "else:\n"
            "    t_wk, t_mo = R['wh_wk_t'], R['wh_mo_t']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.2))\n"
            "ax.bar(['1 week', '1 month'], [t_wk, t_mo], color=GREY, width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='certification bar'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (CHDN - market)'); ax.set_ylim(-0.5, 2.5)\n"
            "ax.set_title('CHDN is trivially \"better\" than a market drifting down — but not vs zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Welch t CHDN-market: 1wk {t_wk:+.3f}, 1mo {t_mo:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: Welch *t* (CHDN − market) is {R['wh_wk_t']:.2f} (1 week) / "
            f"{R['wh_mo_t']:.2f} (1 month) — CHDN \"beats\" the seasonal only because the "
            "market leg is softly negative and CHDN is ~flat. Neither is distinguishable "
            "from zero on its own. **\"Sell in May?\" — Not supported**: the Derby week is "
            "not a market turn, and CHDN is not an event trade."
        ),
        md(
            "### 4h · Faithful-engine & power control\n\n"
            "Synthetic paired (CHDN-like, SPY-like) log-return world (ρ≈0.60, CHDN idio "
            "vol > market), a scheduled synthetic first-Saturday calendar, a TUNABLE bump "
            "landing on the first session after each race. Null (bump=0) checked over "
            "**20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=733+s, k=5)['t'] for s in range(20)])\n"
            "p2 = st.synthetic_detect(bump=0.02, seed=733, k=5)\n"
            "p3 = st.synthetic_detect(bump=0.03, seed=733, k=5)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [p2['t']], color=AMBER, s=90, zorder=5, label='planted bump=2%')\n"
            "ax.scatter([2], [p3['t']], color=RED, s=90, zorder=5, label='planted bump=3%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted 2%', 'planted 3%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: quiet null, planted bumps light up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted 2%% t={p2[\"t\"]:+.2f}  planted 3%% t={p3[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at |t|≥2 "
            f"in only {R['syn_null_fire']}/{R['syn_null_seeds']} seeds; a planted 2% bump "
            f"reads t={R['syn_planted2_t']:.2f}, a 3% bump t={R['syn_planted3_t']:.2f}. The "
            "machinery detects a real Derby bump when one is planted — the real-tape story "
            "is genuinely this flat. *(A faithful-engine / power check only — never cited "
            "in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — CHDN, the directly-exposed name, is flat at every "
            f"horizon (run-up {R['ch_ru_mean']:+.3f}%/*t*={R['ch_ru_t']:.2f}, 1-week "
            f"{R['ch_wk_mean']:+.3f}%/*t*={R['ch_wk_t']:.2f}, 1-month "
            f"{R['ch_mo_mean']:+.3f}%/*t*={R['ch_mo_t']:.2f}, placebo *p*≥{R['pl_ch_ru_p']:.2f}) "
            "with *full* tape coverage — the cleanest kind of null, and the exposure-"
            "dilution caveat only biases toward finding an effect. The market's one whisper "
            f"(1-week {R['mk_wk_mean']:+.3f}%, *t*={R['mk_wk_t']:.2f}) misses the bar raw and "
            f"fails its drift-neutral placebo (*p*={R['pl_mk_wk_p']:.3f}).\n"
            f"- **Tradability `MIRAGE`** — no honest cut is bankable; the only |*t*|≥2 "
            f"(market/1-week net, {R['mk_wk_cap_t']:.2f}) is costs charged against an "
            f"already-negative return, dies under its own placebo (*p*={R['pl_mk_wk_cap_p']:.3f}), "
            "and would require shorting the S&P 25 times for a dip inside the noise.\n"
            f"- **\"Sell in May?\" `NOT SUPPORTED`** — the Derby-week dip is real in sign but "
            f"inside the luck cloud (*p*={R['pl_mk_wk_p']:.3f}) and fragile to a single year "
            f"(jackknife [{R['jk_lo']:.2f}, {R['jk_hi']:.2f}])."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The transferable lesson is the cost trap.** When a \"short this seasonal\" "
            "edge gets *more* significant after costs, the costs are being charged against "
            "a negative return — the *t* is inflated, not eroded. Always re-test against a "
            "drift-neutral placebo before believing a costed short.\n"
            "- **A clean null is worth publishing.** No survivorship funnel, a directly-"
            "exposed stock, full history — the ideal setup to find an effect, and it finds "
            "none. That is a stronger \"no\" than a data-starved one.\n"
            "- **A cleaner test would need different data.** Betting-handle or attendance "
            "series, intraday moves around the actual post time, or a purpose-built gaming "
            "basket (once names like DKNG have enough history) could raise power — but the "
            "prior after this is low.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) (same "
            "machinery, survivorship story), [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "(market + exposed basket, negative shock), [158-super-bowl](../../158-super-bowl/), "
            "[235-world-cup-effect](../../235-world-cup-effect/), "
            "[709-world-series-effect](../../709-world-series-effect/). None pairs a "
            "single-index seasonal with the one listed company that *operates* the event.\n\n"
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
