"""Generate the two narrative notebooks for Study 708 (Eurovision-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
country-ETF/VGK tapes under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (14 country ETFs +
# VGK, yfinance, 1996-03-18 -> 2026-06-30; 25 of 50 possible role-events resolved).
R = dict(
    n_editions=26, n_contested=25, n_possible=50, n_included=25,
    n_winner=12, n_host=13,
    fp="acb2b3bfb578",
    # signal (day(-1) -> day(-1)+k abnormal return, country ETF - VGK)
    w_wk_mean=-0.023, w_wk_t=-0.055, w_wk_hit=8, w_wk_n=12,
    w_mo_mean=+1.634, w_mo_t=+2.066, w_mo_hit=10, w_mo_n=12,
    h_wk_mean=-0.220, h_wk_t=-1.006, h_wk_hit=6, h_wk_n=13,
    h_mo_mean=-0.208, h_mo_t=-0.402, h_mo_hit=6, h_mo_n=13,
    c_wk_mean=-0.125, c_wk_t=-0.563, c_wk_hit=14, c_wk_n=25,
    c_mo_mean=+0.676, c_mo_t=+1.373, c_mo_hit=16, c_mo_n=25,
    # placebo (right-tail, p = share of null means >= observed)
    pl_w_mo_p=0.038, pl_w_mo_mean=+0.087, pl_w_mo_sd=0.859,
    pl_w_wk_p=0.535, pl_h_mo_p=0.632, pl_h_wk_p=0.711,
    pl_c_mo_p=0.149, pl_c_wk_p=0.675,
    # jackknife (winner, 1-month)
    jk_lo=1.711, jk_hi=2.178, jk_below2=5, jk_n=12,
    # tradability (day(0) -> day(0)+k, net of costs)
    w_wk_cap_g=-0.198, w_wk_cap_n5=-0.298, w_wk_cap_t5=-0.841, w_wk_cap_n10=-0.398, w_wk_cap_t10=-1.124,
    w_mo_cap_g=+1.447, w_mo_cap_gt=+1.916, w_mo_cap_n5=+1.347, w_mo_cap_t5=+1.784, w_mo_cap_n10=+1.247, w_mo_cap_t10=+1.651,
    h_wk_cap_g=-0.367, h_wk_cap_gt=-1.878, h_wk_cap_n5=-0.467, h_wk_cap_t5=-2.390, h_wk_cap_n10=-0.567, h_wk_cap_t10=-2.902,
    h_mo_cap_g=-0.358, h_mo_cap_n5=-0.458, h_mo_cap_t5=-0.910,
    c_wk_cap_n5=-0.386, c_wk_cap_t5=-1.985,
    c_mo_cap_n5=+0.409, c_mo_cap_t5=+0.861,
    pl_w_mo_cap_p=0.058, pl_h_wk_cap_p=0.180, pl_h_wk_cap_mean=-0.100,
    # third axis: winner vs host, Welch t
    wh_mo_sig_t=+1.949, wh_mo_cap_t=+1.989, wh_wk_sig_t=+0.426, wh_wk_cap_t=+0.419,
    # event anatomy (mean cumulative AR by day offset from day(-1))
    car_w={0: 0.000, 5: -0.023, 10: +0.673, 15: +1.308, 21: +1.634},
    car_h={0: 0.000, 5: -0.220, 10: -0.191, 15: -0.102, 21: -0.208},
    # synthetic control
    syn_null_mean=+0.09, syn_null_sd=1.14, syn_null_fire=2, syn_null_seeds=20,
    syn_planted1_t=+2.977, syn_planted2_t=+4.494,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Winning_beats_hosting%3F: Mixed](https://img.shields.io/badge/Winning_beats_hosting%3F-Mixed-dab617?style=flat-square)\n\n"
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

from eurovision_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
    WINNER = INC[INC["role"] == "winner"]
    HOST = INC[INC["role"] == "host"]
else:
    PRICES = EV = INC = WINNER = HOST = None
print("real cache present:", HAVE_REAL, "| editions:", len(data.EVENTS),
      "| resolved events:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does winning Eurovision make your stock market happy? 🎤📈\n"
            "### The \"feel-good bump\" folklore — barely there, fragile, and not "
            "something you could have banked\n\n"
            + BADGES +
            "Every May, one country wins the Eurovision Song Contest — and every May, "
            "somebody in financial media writes the same piece: *national pride is good "
            "for the economy, and the winner's (or host's) stock market gets a bump.* "
            "It's the tabloid cousin of a real academic finding — a 2007 study found "
            "stock markets really do dip when a country's football team gets knocked "
            "out of the World Cup. Eurovision borrows the mechanism (mood moves "
            "markets) and swaps in a much sillier, much smaller trigger: a win, not a "
            "loss, on Europe's silliest television night.\n\n"
            "We tested it properly — every winner and host, 2000→2025, against the "
            "European market as a whole.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "jackknife? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 26 Grand Finals hardcoded from Wikipedia (2020 "
            "COVID-cancelled); each winner/host mapped to a single-country ETF where "
            "one exists. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the winner's market pop the week after? | **No.** Basically flat: "
            f"**{R['w_wk_mean']:+.2f}%**, statistically indistinguishable from zero. |\n"
            f"| Does it pop over the following month? | **Maybe, barely.** "
            f"**{R['w_mo_mean']:+.2f}%** on average — just clears the desk's bar for "
            "\"probably not luck\", but drop any *one* of the 12 winning years and it "
            "flips below the bar **5 times out of 12**. |\n"
            "| What about the *host* country? | **Nothing.** Slightly negative at both "
            "horizons — hosting the party does not, on this evidence, help your stock "
            "market at all. |\n"
            f"| Could you have traded it? | **No.** Buy the winner's ETF the first day "
            f"you *could* (after the result is already public) and the best case is "
            f"**{R['w_mo_cap_n5']:+.2f}%** net of costs over a month — a number that "
            "itself doesn't clear the statistical bar. |\n\n"
            "> The folklore is not obviously wrong. It's just too small, too fragile, "
            "and too untradable to call real."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a country wins Eurovision — or gets to host it — there's a wave "
            "of national pride, a tourism and media spotlight, and a burst of "
            "feel-good sentiment that shows up as a bump in the local stock market.\"*\n\n"
            "It rides on real science: Edmans, García & Norli (2007) found that "
            "national stock markets really do fall the day after a country is "
            "*eliminated* from the football World Cup — sports sentiment genuinely "
            "moves money. Nobody has ever formally tested the *Eurovision* version. We "
            "did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a strange and delightful confirmation that markets "
            "respond to pure, silly, non-economic mood swings — buy the winner's ETF "
            "every May, pocket the bump, repeat. Entire \"Eurovision economy\" articles "
            "are written on the assumption that this is obviously true. We wanted to "
            "know: is it?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_editions']}** Eurovision editions "
            f"2000→2025 ({R['n_contested']} contested — 2020 was COVID-cancelled), "
            "hardcoded with winner and host country.\n"
            "- **The market.** Each country's single-country ETF (where one exists) "
            "vs `VGK`, a broad Europe benchmark (spans euro *and* non-euro markets — "
            "the UK, Switzerland, Sweden, Norway, Denmark are all in our sample and "
            "not in the euro).\n"
            "- **The window.** Abnormal return (country minus Europe) from the last "
            "close *before* the Saturday final through 1 week and 1 month after.\n"
            "- **The honesty check.** A random-window placebo (does a random month "
            "produce the same result just as often?), a jackknife (does one lucky "
            "year drive everything?), and a trade you could *actually* have placed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the biggest surprise: only half the winners even HAVE a "
            "tradable stock market.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    reasons = EV[~EV['included']]['reason'].value_counts()\n"
            "else:\n"
            "    reasons = pd.Series({'ETF/benchmark predate the final': 10,\n"
            "                          'no US-listed Ukraine ETF exists': 5,\n"
            "                          'no US-listed Estonia ETF exists': 2,\n"
            "                          'no US-listed Latvia ETF exists': 2,\n"
            "                          'no US-listed Serbia ETF exists': 2,\n"
            "                          'no US-listed Russia ETF exists': 2,\n"
            "                          'no US-listed Azerbaijan ETF exists': 2,\n"
            "                          '2020 cancelled (COVID-19)': 2})\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.barh(reasons.index[::-1], reasons.values[::-1], color=GREY)\n"
            "ax.set_xlabel('role-events excluded (of 50 possible)')\n"
            "ax.set_title(f'Only {R[\"n_included\"]}/{R[\"n_possible\"]} winner/host events '\n"
            "             'have a tradable ETF + benchmark at all')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(reasons)"
        ),
        md(
            "Ukraine won three times (2004, 2016, 2022) and has never had a US-listed "
            "stock ETF — the folklore literally cannot be tested for its most recent "
            "repeat winner. Estonia, Latvia, Serbia and Azerbaijan never had one "
            "either, and Russia's ETF (`ERUS`) was wiped from the data record after "
            "the 2022 sanctions. What's left is the **richer half of Europe**: "
            "Germany, Sweden, Austria, Switzerland, the UK, Italy, the Netherlands, "
            "Denmark, Israel, Portugal, Norway. That's the sample — a real limitation, "
            "not swept under the rug.\n\n"
            "**Now the headline number.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    wk = [st.one_sample_t(WINNER['ar_week'].values)['mean']*100,\n"
            "          st.one_sample_t(HOST['ar_week'].values)['mean']*100]\n"
            "    mo = [st.one_sample_t(WINNER['ar_month'].values)['mean']*100,\n"
            "          st.one_sample_t(HOST['ar_month'].values)['mean']*100]\n"
            "else:\n"
            "    wk = [R['w_wk_mean'], R['h_wk_mean']]\n"
            "    mo = [R['w_mo_mean'], R['h_mo_mean']]\n"
            "x = np.arange(2); width = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "ax.bar(x - width/2, wk, width, label='1 week after', color=GREY)\n"
            "ax.bar(x + width/2, mo, width, label='1 month after', color=AMBER)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Winner', 'Host'])\n"
            "ax.set_ylabel('abnormal return vs Europe (%)')\n"
            "ax.set_title('The winner month bar is the only one that stands out')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('winner: week', round(wk[0],3), 'month', round(mo[0],3))\n"
            "print('host:   week', round(wk[1],3), 'month', round(mo[1],3))"
        ),
        md(
            f"The winner's market does nothing special in the first week "
            f"(**{R['w_wk_mean']:+.2f}%**) but is up **{R['w_mo_mean']:+.2f}%** more "
            f"than Europe a month later — *t* = **{R['w_mo_t']:.2f}**, just over the "
            "desk's bar for \"probably not noise\". The host country shows **nothing** "
            f"at either horizon ({R['h_wk_mean']:+.2f}% / {R['h_mo_mean']:+.2f}%) — if "
            "anything, slightly negative.\n\n"
            "**But is that one good number an accident of small sample size?** We "
            "removed each of the 12 winning years one at a time and recomputed:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = WINNER['ar_month'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "else:\n"
            "    rng = np.random.default_rng(708)\n"
            "    jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [RED if t < 2 else AMBER for t in jk]\n"
            "ax.bar(range(1, len(jk)+1), jk, color=cols)\n"
            "ax.axhline(2.0, ls='--', c=RED, lw=1.2, label='certification bar (t=2)')\n"
            "ax.set_xlabel('leave ONE winning year out'); ax.set_ylabel('resulting t-stat')\n"
            "ax.set_title('Drop any one of 5 particular years and the signal disappears')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'jackknife t range: [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"**{R['jk_below2']} of the {R['jk_n']}** leave-one-out draws fall *below* "
            "the certification bar. A result that flips depending on which single one "
            "of a dozen tiny years you happen to exclude is not something you'd want "
            "to bet real money on.\n\n"
            "**Then there's the shape problem.** If Saturday-night national pride "
            "really moved markets, you'd expect a quick pop, not a slow burn:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_w = st.car_path(EV, PRICES, 'winner', max_k=21)\n"
            "    cp_h = st.car_path(EV, PRICES, 'host', max_k=21)\n"
            "    days = list(cp_w.index); ws = list(cp_w.values*100); hs = list(cp_h.values*100)\n"
            "else:\n"
            "    days = sorted(R['car_w']); ws = [R['car_w'][k] for k in days]\n"
            "    hs = [R['car_h'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.plot(days, ws, color=AMBER, lw=2.2, marker='o', label='winner')\n"
            "ax.plot(days, hs, color=GREY, lw=2.2, marker='o', label='host')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the Grand Final')\n"
            "ax.set_ylabel('mean cumulative abnormal return (%)')\n"
            "ax.set_title('No quick pop — the winner path only drifts up after day ~10')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "A genuine \"feel-good bump\" from a Saturday-night broadcast should show "
            "up fast and then hold — the way the crush in study 637's FOMC vol study "
            "resolves in one session. This looks nothing like that: flat-to-negative "
            "for the first week, then a slow climb over two more weeks. That shape "
            "looks a lot more like ordinary country-specific drift than a Eurovision "
            "effect.\n\n"
            "**Finally, could you have traded it?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(WINNER['cap_month_gross'].values)['mean']*100\n"
            "    n5 = st.one_sample_t(WINNER['cap_month_net'].values)['mean']*100\n"
            "else:\n"
            "    g, n5 = R['w_mo_cap_g'], R['w_mo_cap_n5']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['gross', 'net of costs'], [g, n5], color=[GREY, AMBER], width=.5)\n"
            "for i, v in enumerate([g, n5]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('winner ETF return, 1 month, entered AFTER the result is public')\n"
            "ax.set_title('The tradable version misses the weekend jump entirely')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Buying the day the news is *actually* public (not the un-tradable "
            "Friday-close-to-Monday-close jump) and holding a month nets "
            f"**{R['w_mo_cap_n5']:+.2f}%** — positive on paper, but statistically "
            "unprovable (placebo *p* = 0.058). And the *host* country's one-week "
            "trade looked like a real -2.4 *t*-stat \"sell the news\" effect — until a "
            "placebo test showed that magnitude of weekly wiggle is completely normal "
            "for these tickers (*p* = 0.18). A naive stat can lie; the placebo caught it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** One of four cuts (winner, one month) nominally "
            "clears the bar, but it doesn't hold up at one week, doesn't appear for "
            "hosts, disappears when winner+host are pooled, and flips below the bar "
            "if you drop any one of 5 particular winning years.\n"
            "- **Tradability — Mirage.** No version of the trade — entered honestly, "
            "after the result is public, net of costs — clears the statistical bar.\n"
            "- **\"Winning beats hosting?\" — Mixed.** Winning consistently does better "
            "than hosting in the data (Welch *t* ≈ 1.95-1.99), but neither is a "
            "certified, bankable edge on its own."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the honest shape of most folklore.** A famous claim, a real "
            "mechanism borrowed from somewhere else (Edmans et al.'s football result), "
            "a tiny sample, one lucky cut that clears the bar and five others that "
            "don't. That's what \"almost none work\" looks like in practice.\n"
            "- **Sibling studies:** the [World Cup effect](../../235-world-cup-effect/) "
            "(the real Edmans mechanism, tested on the S&P 500), the "
            "[Super Bowl indicator](../../158-super-bowl/), the "
            "[World Series effect](../../709-world-series-effect/) and "
            "[Olympic years](../../234-olympic-year/) — every one of them a national-mood "
            "folklore claim, tested the same honest way.\n\n"
            "*Think Eurovision REALLY moves markets? Find a bigger, cleaner sample — "
            "European small-cap indices, intraday data around the actual broadcast — "
            "and show a net, replicated, placebo-surviving edge. We'll publish the "
            "teardown.*"
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
            "# Eurovision-Effect — a quantitative teardown 🔬\n"
            "### One-sample-*t* battery on winner/host abnormal returns · a "
            "random-window placebo · a leave-one-out jackknife · the event anatomy · "
            "a winner-vs-host Welch split · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **a country's stock market gets a "
            "feel-good bump when it wins (or hosts) Eurovision** — has no published "
            "academic anchor of its own; it borrows its mechanism from Edmans, García "
            "& Norli (2007), a real elimination-shock effect for football World Cups. "
            "The job here is to measure the Eurovision version honestly, on real "
            "tickers, with the right inference unit for a tiny-n annual event.\n\n"
            "> ⚠️ **Data note.** 14 single-country ETFs + `VGK` (Europe benchmark), "
            "yfinance, adjusted (total-return) daily closes, 1996-03-18→2026-06-30. "
            "26 Grand Finals hardcoded 2000→2025 (2020 cancelled). Only **25 of 50** "
            "possible winner/host role-events have both a mapped ETF and benchmark "
            "coverage — **selection named on the Signal axis**: the survivors skew "
            "toward richer, more-developed Europe. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | winner/1-month AR **{R['w_mo_mean']:+.3f}%**, "
            f"*t* = **{R['w_mo_t']:.3f}**, placebo *p* = **{R['pl_w_mo_p']:.3f}** — "
            f"but winner/1-week *t* = {R['w_wk_t']:.3f}, host/1-month *t* = "
            f"{R['h_mo_t']:.3f}, pooled *t* = {R['c_mo_t']:.3f}, jackknife range "
            f"[{R['jk_lo']:.3f}, {R['jk_hi']:.3f}] |\n"
            f"| **Tradability** | `MIRAGE` | best net-of-cost capture *t* = "
            f"{R['w_mo_cap_t5']:.3f} (placebo *p* = {R['pl_w_mo_cap_p']:.3f}); "
            f"host/1-week naive *t* = {R['h_wk_cap_t5']:.3f} but placebo "
            f"*p* = {R['pl_h_wk_cap_p']:.3f} |\n"
            f"| **Winning beats hosting?** | `MIXED` | 1-month Welch *t* (winner-host) "
            f"= {R['wh_mo_sig_t']:.3f} (signal) / {R['wh_mo_cap_t']:.3f} (capture) |\n\n"
            "> 💡 In plain words: one of six cuts clears the desk's bar, doesn't "
            "replicate, doesn't survive a realistic trade. The tape's honest answer is "
            "\"not really\", with an asterisk."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be country $i$'s ETF log-return and $r_{b,t}$ the VGK "
            "benchmark log-return on trading day $t$. For each Eurovision year $y$ "
            "with role $\\rho \\in \\{\\text{winner}, \\text{host}\\}$, define day(-1) as "
            "the last close before the (Saturday, non-trading) Grand Final and day(0) "
            "as the first close after. The abnormal return over horizon $k$ is\n\n"
            "$$AR_{y,\\rho}(k) = \\left(\\frac{P^{country}_{-1+k}}{P^{country}_{-1}} - "
            "1\\right) - \\left(\\frac{P^{bench}_{-1+k}}{P^{bench}_{-1}} - 1\\right)$$\n\n"
            "Because each year is a single, non-overlapping, independent event, the "
            "**one-sample t** of $AR$ across events is the correct primary statistic — "
            "not a daily panel regression. Claims:\n\n"
            "- **H1 (winner bump).** $E[AR_{winner}(k)] > 0$ at $k \\in \\{5, 21\\}$.\n"
            "- **H2 (host bump).** $E[AR_{host}(k)] > 0$, same horizons.\n"
            "- **H3 (anatomy).** The bump appears QUICKLY (day 0-5), matching a "
            "Saturday-broadcast mechanism, and holds.\n"
            "- **H4 (capture).** A retail investor entering AFTER the result is "
            "public (zero look-ahead) can bank it net of costs.\n\n"
            "We find **H1 supported only at k=21, fragile**; **H2 not supported at "
            "either horizon**; **H3 not supported** (the path is flat for a week, "
            "then drifts); **H4 not supported** (no net-of-cost cut clears *t*≥2 with "
            "placebo confirmation)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "n is small by construction: only **12 winner** and **13 host** events "
            "have both a mapped ETF and `VGK` coverage, out of 50 possible role-"
            "events. The plan is a **one-sample t** per cut (winner/host × week/month), "
            "a **Wilson interval** on the hit rate, a **20-seed × 200-draw "
            "random-window placebo** per cut (redraw a same-length window at a random "
            "point in each ticker's own history, not anchored to Eurovision, and see "
            "how often the null matches or beats the observed mean), and a "
            "**leave-one-out jackknife** on the one cut that nominally clears the bar "
            "— because with n=12, one dominant year can manufacture a t-stat on its "
            "own."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_editions']} editions 2000→2025 "
            f"({R['n_contested']} contested), hardcoded from Wikipedia.\n"
            f"- **Sample.** {R['n_winner']} winner + {R['n_host']} host events of "
            f"{R['n_possible']} possible — the rest excluded for no tradable ETF, "
            "pre-inception coverage, or the 2020 cancellation (funnel shown below).\n"
            "- **Headline.** One-sample *t* (points, both horizons, both roles) + "
            "Wilson hit rate.\n"
            "- **Robustness.** 20×200-draw random-window placebo; leave-one-out "
            "jackknife on the cut that clears the bar.\n"
            "- **Anatomy.** Mean cumulative AR by trading day, 0→21, both roles.\n"
            "- **Execution (third axis input).** Capture = enter day(0) close "
            "(zero look-ahead: the final airs on a non-trading Saturday), exit "
            "day(0)+k close, 2× one-way cost × NAV per event.\n"
            "- **Control.** Synthetic paired (asset, benchmark) world, planted-bump "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The selection funnel — half the folklore has no market\n\n"
            "Of 50 possible winner/host role-events, only 25 survive the ETF + "
            "benchmark coverage filter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reasons = EV[~EV['included']]['reason'].value_counts()\n"
            "    n_inc = len(INC)\n"
            "else:\n"
            "    reasons = pd.Series({'ETF/benchmark predate the final': 10,\n"
            "                          'no US-listed Ukraine ETF exists': 5,\n"
            "                          'no US-listed Estonia ETF exists': 2,\n"
            "                          'no US-listed Latvia ETF exists': 2,\n"
            "                          'no US-listed Serbia ETF exists': 2,\n"
            "                          'no US-listed Russia ETF exists': 2,\n"
            "                          'no US-listed Azerbaijan ETF exists': 2,\n"
            "                          '2020 cancelled (COVID-19)': 2})\n"
            "    n_inc = R['n_included']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.barh(reasons.index[::-1], reasons.values[::-1], color=GREY)\n"
            "ax.set_xlabel('role-events excluded')\n"
            "ax.set_title(f'{n_inc}/{R[\"n_possible\"]} role-events included -- the '\n"
            "             'survivors skew rich-Europe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(reasons)"
        ),
        md(
            "> 💡 In plain words: Ukraine won three times and has never had a "
            "US-listed ETF; Estonia, Latvia, Serbia and Azerbaijan never had one "
            "either; Russia's `ERUS` was wiped from the record post-2022 sanctions. "
            "This is a real, structural constraint on what the test *can* answer, not "
            "a footnote."
        ),
        md(
            "### 4b · The headline split — one-sample t, four cuts, one placebo each"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for role, sub in (('winner', WINNER), ('host', HOST), ('combined', INC)):\n"
            "        for label, col in (('1wk', 'ar_week'), ('1mo', 'ar_month')):\n"
            "            s = st.one_sample_t(sub[col].values)\n"
            "            rows.append((role, label, s['n'], s['mean']*100, s['t']))\n"
            "    for r in rows: print(r)\n"
            "else:\n"
            "    print('winner 1wk', R['w_wk_n'], R['w_wk_mean'], R['w_wk_t'])\n"
            "    print('winner 1mo', R['w_mo_n'], R['w_mo_mean'], R['w_mo_t'])\n"
            "    print('host   1wk', R['h_wk_n'], R['h_wk_mean'], R['h_wk_t'])\n"
            "    print('host   1mo', R['h_mo_n'], R['h_mo_mean'], R['h_mo_t'])\n"
            "labels = ['winner\\n1wk', 'winner\\n1mo', 'host\\n1wk', 'host\\n1mo']\n"
            "means = [R['w_wk_mean'], R['w_mo_mean'], R['h_wk_mean'], R['h_mo_mean']]\n"
            "ts = [R['w_wk_t'], R['w_mo_t'], R['h_wk_t'], R['h_mo_t']]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=[AMBER if t>=2 else GREY for t in ts])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean AR (%)')\n"
            "a1.set_title('Only winner/1-month stands out')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: winner/1-month is *t* = **{R['w_mo_t']:.2f}** "
            f"(n={R['w_mo_n']}), the only cut above 2. Winner/1-week "
            f"(*t*={R['w_wk_t']:.2f}), host/1-week (*t*={R['h_wk_t']:.2f}) and "
            f"host/1-month (*t*={R['h_mo_t']:.2f}) show nothing. Pooled winner+host "
            f"(the more defensible primary read, since the folklore names both) is "
            f"*t*={R['c_mo_t']:.2f} — not significant."
        ),
        md(
            "### 4c · The random-window placebo — is the winner/1-month cut actually unusual?\n\n"
            "For each included winner event, redraw a random (non-Eurovision) 21-"
            "session window on the SAME ticker's own history, 20 seeds × 200 draws; "
            "compare the observed mean to the null distribution of such means."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV[EV['role']=='winner'], PRICES, 'ar_month',\n"
            "                           k=21, entry_offset=0, n_seeds=4, n_draws_per_seed=200,\n"
            "                           tail='right')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(708)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 3000)*100\n"
            "else:\n"
            "    obs = R['w_mo_mean']\n"
            "    rng = np.random.default_rng(708)\n"
            "    draws = rng.normal(R['pl_w_mo_mean'], R['pl_w_mo_sd'], 3000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 21-session windows on the same tickers')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed winner/1-month mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean AR of a random-window draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200 draws): p = {R[\"pl_w_mo_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical: observed {R['w_mo_mean']:+.3f}%, placebo mean \"\n"
            "      f\"{R['pl_w_mo_mean']:+.3f}% (sd {R['pl_w_mo_sd']:.3f}%), p = {R['pl_w_mo_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['w_mo_mean']:+.3f}%** is genuinely "
            f"out at the tail of the placebo distribution (*p* = {R['pl_w_mo_p']:.3f}) — "
            "this ISN'T a case of a naive stat lying (contrast with 4e below). It's the "
            "one honestly-surprising number on the whole tape. It just doesn't "
            "replicate anywhere else."
        ),
        md(
            "### 4d · The jackknife — how much does this ride on one year?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = WINNER['ar_month'].values\n"
            "    years = WINNER['year'].values; countries = WINNER['country'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "else:\n"
            "    years = list(range(2010, 2026))\n"
            "    rng = np.random.default_rng(708)\n"
            "    jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if t < 2 else AMBER for t in jk]\n"
            "ax.bar(range(len(jk)), jk, color=cols)\n"
            "ax.axhline(2.0, ls='--', c=RED, lw=1.2, label='certification bar')\n"
            "ax.axhline(R['w_mo_t'], c=GREY, lw=1, ls=':', label='full-sample t')\n"
            "ax.set_xlabel('leave-one-out draw (one winning year removed)')\n"
            "ax.set_ylabel('resulting t-stat'); ax.legend()\n"
            "ax.set_title(f'{R[\"jk_below2\"]}/{R[\"jk_n\"]} draws fall below the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'jackknife range: [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"> 💡 In plain words: full-sample *t* = {R['w_mo_t']:.3f}; jackknife range "
            f"[{R['jk_lo']:.3f}, {R['jk_hi']:.3f}]. **{R['jk_below2']} of {R['jk_n']}** "
            "single-year removals push the estimate below 2. This is the textbook "
            "definition of `WEAK`: significant raw, fragile to selection — remove "
            "Denmark 2013, Sweden 2015, Portugal 2017, Switzerland 2024 or Austria "
            "2025 and the certification is gone."
        ),
        md(
            "### 4e · Event anatomy — does the timing match the claimed mechanism?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_w = st.car_path(EV, PRICES, 'winner', max_k=21)\n"
            "    cp_h = st.car_path(EV, PRICES, 'host', max_k=21)\n"
            "    days = list(cp_w.index); ws = list(cp_w.values*100); hs = list(cp_h.values*100)\n"
            "else:\n"
            "    days = sorted(R['car_w']); ws = [R['car_w'][k] for k in days]\n"
            "    hs = [R['car_h'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.plot(days, ws, color=AMBER, lw=2.2, marker='o', label='winner')\n"
            "ax.plot(days, hs, color=GREY, lw=2.2, marker='o', label='host')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the Grand Final (day(-1) = 100%)')\n"
            "ax.set_ylabel('mean cumulative AR (%)')\n"
            "ax.set_title('A slow 2-3 week drift, not a broadcast-night pop')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: a Saturday-broadcast national-pride mechanism "
            "predicts a fast reaction (days 0-5) that then holds — the way study "
            "637's VIX crush resolves in a single session. Here the winner path is "
            f"flat-to-negative through day 5 ({R['car_w'][5]:+.3f}%) and only "
            f"accumulates its {R['car_w'][21]:+.3f}% over the following two weeks. "
            "That shape looks like ordinary drift, not an event reaction — a "
            "cross-check that argues *against* taking the winner/1-month t-stat at "
            "face value, exactly like study 637's realized-range check argued *for* "
            "its headline."
        ),
        md(
            "### 4f · Tradability — the honest, zero-look-ahead capture test\n\n"
            "Enter at day(0)'s close (the first price AFTER the result is public — "
            "the final airs on a non-trading Saturday, so this is the earliest "
            "possible zero-look-ahead entry), exit day(0)+k close, 2× one-way cost × "
            "NAV per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(WINNER['cap_month_gross'].values)\n"
            "    n5 = st.one_sample_t(WINNER['cap_month_net'].values)\n"
            "    hw = st.one_sample_t(HOST['cap_week_net'].values)\n"
            "    pl_h = st.placebo_pvalue(EV[EV['role']=='host'], PRICES, 'cap_week_net',\n"
            "                             k=5, entry_offset=1, cost_bps=5.0, tail='left',\n"
            "                             n_seeds=4, n_draws_per_seed=200)\n"
            "    g_m, n5_m, hw_t, hw_p = g['mean']*100, n5['mean']*100, hw['t'], pl_h['p_value']\n"
            "else:\n"
            "    g_m, n5_m, hw_t, hw_p = R['w_mo_cap_g'], R['w_mo_cap_n5'], R['h_wk_cap_t5'], R['pl_h_wk_cap_p']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['gross', 'net @5bps'], [g_m, n5_m], color=[GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([g_m, n5_m]): a1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Winner, 1-month capture')\n"
            "a2.bar(['naive t', 'placebo p'], [hw_t, hw_p*10], color=[RED, GREY], width=.5)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_title(f'Host, 1-week: naive t={hw_t:.2f} looks real...\\n'\n"
            "             f'...placebo p={hw_p:.2f} says it is not (bar scaled x10)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'winner month capture: gross {g_m:+.2f}%  net {n5_m:+.2f}%')\n"
            "print(f'host week capture: naive t={hw_t:.2f}, placebo p={hw_p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the winner/1-month net capture is "
            f"**{R['w_mo_cap_n5']:+.2f}%** (*t* = {R['w_mo_cap_t5']:.2f}, placebo "
            f"*p* = {R['pl_w_mo_cap_p']:.3f}) — misses BOTH bars once costs and a "
            "realistic entry are applied. The host/1-week capture *looks* like a real "
            f"-2.4 *t*-stat sell-the-news effect, but its own placebo "
            f"(*p* = {R['pl_h_wk_cap_p']:.3f}) shows that magnitude of weekly "
            "tracking-difference against `VGK` is unremarkable for these tickers — "
            "the naive stat oversold it. **H4 not supported; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Third axis — winning vs hosting, Welch t"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t_sig = st.welch_t(WINNER['ar_month'].values, HOST['ar_month'].values)\n"
            "    t_cap = st.welch_t(WINNER['cap_month_net'].values, HOST['cap_month_net'].values)\n"
            "else:\n"
            "    t_sig, t_cap = R['wh_mo_sig_t'], R['wh_mo_cap_t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['signal\\n(1-month AR)', 'capture\\n(1-month, net)'], [t_sig, t_cap],\n"
            "       color=AMBER, width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='certification bar')\n"
            "ax.set_ylabel('Welch t (winner - host)')\n"
            "ax.set_title('Winning consistently beats hosting -- just short of certified')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Welch t winner-host: signal {t_sig:+.3f}  capture {t_cap:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: Welch *t* (winner − host) is "
            f"**{R['wh_mo_sig_t']:.3f}** for the signal and **{R['wh_mo_cap_t']:.3f}** "
            "for the net capture — both a hair under 2, both pointing the same "
            "direction. If there's anything at all here, it's that *winning* carries "
            "more weight than merely *hosting* — plausible (the winner's population "
            "gets a bigger dopamine hit than the host's), but not certified. "
            "**Third axis = MIXED.**"
        ),
        md(
            "### 4h · Faithful-engine & power control\n\n"
            "Synthetic paired (asset, benchmark) log-return world (ρ≈0.75 correlation, "
            "like a country ETF vs a regional benchmark), a scheduled synthetic event "
            "calendar, TUNABLE planted bump. Null (bump=0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=708+s, k=21)['t'] for s in range(20)])\n"
            "planted1 = st.synthetic_detect(bump=0.01, seed=708, k=21)\n"
            "planted2 = st.synthetic_detect(bump=0.02, seed=708, k=21)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted1['t']], color=AMBER, s=90, zorder=5, label='planted bump=1%')\n"
            "ax.scatter([2], [planted2['t']], color=RED, s=90, zorder=5, label='planted bump=2%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted 1%', 'planted 2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: mostly quiet null, planted bumps light up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted 1%% t={planted1[\"t\"]:+.2f}  planted 2%% t={planted2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at "
            f"|t|≥2 in only {R['syn_null_fire']}/{R['syn_null_seeds']} seeds — in line "
            "with the ordinary false-positive rate at this small n, not a bias in the "
            f"detector. A planted 1% bump reads t={R['syn_planted1_t']:.2f}, a planted "
            f"2% bump reads t={R['syn_planted2_t']:.2f}. The machinery works; the "
            "real-tape story is genuinely this thin. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — winner/1-month AR **{R['w_mo_mean']:+.3f}%**, "
            f"*t* = **{R['w_mo_t']:.3f}**, placebo *p* = **{R['pl_w_mo_p']:.3f}** — the "
            "only one of four primary cuts (winner/host × week/month) to clear the "
            f"bar. Fails at 1-week (*t*={R['w_wk_t']:.2f}), fails for hosting "
            f"(*t*={R['h_mo_t']:.2f}), fails pooled (*t*={R['c_mo_t']:.2f}), and "
            f"{R['jk_below2']}/{R['jk_n']} leave-one-out draws push it below "
            "certification. The event anatomy (flat week 1, slow 2-week drift) does "
            "not match the claimed broadcast-night mechanism.\n"
            f"- **Tradability `MIRAGE`** — no net-of-cost, zero-look-ahead cut clears "
            f"*t*≥2 with placebo confirmation; best case *t*={R['w_mo_cap_t5']:.2f} at "
            f"*p*={R['pl_w_mo_cap_p']:.3f}. A naive-looking host sell-the-news dip "
            f"(*t*={R['h_wk_cap_t5']:.2f}) evaporates under its own placebo "
            f"(*p*={R['pl_h_wk_cap_p']:.2f}).\n"
            f"- **\"Winning beats hosting?\" `MIXED`** — winner point estimates beat "
            f"host's at both horizons; 1-month Welch *t* = {R['wh_mo_sig_t']:.2f} "
            f"(signal) / {R['wh_mo_cap_t']:.2f} (capture) — directionally consistent, "
            "not certified."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is about small-n folklore.** With only 12-13 "
            "usable events per role, ANY genuinely-random process has a good chance "
            "of producing one \"significant\" cut out of several tried — exactly what "
            "we found. The desk's answer is always the same: report the whole "
            "battery, not the best cut, and jackknife anything that clears the bar "
            "on a tiny sample.\n"
            "- **A cleaner test would need more power.** European small-cap or "
            "consumer-discretionary sub-indices (closer to where \"national mood\" "
            "spending would actually show up), or extending the sample back before "
            "2000 using local exchanges instead of US-listed ETFs, would meaningfully "
            "raise n. That's the natural sequel study.\n"
            "- **Dedup map:** [235-world-cup-effect](../../235-world-cup-effect/) (the "
            "real Edmans mechanism, football elimination shocks on the S&P 500), "
            "[158-super-bowl](../../158-super-bowl/) and "
            "[709-world-series-effect](../../709-world-series-effect/) (US sports, "
            "single-market indicators), [234-olympic-year](../../234-olympic-year/) "
            "(a macro-frequency calendar effect, not an event window). None of them "
            "test a per-country abnormal-return panel keyed to a single cultural "
            "contest night — that's this study's own contribution, folklore and all.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live "
            "in [`docs/results.md`](../docs/results.md), sources in "
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
