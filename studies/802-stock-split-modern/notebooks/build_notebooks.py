"""Generate the two narrative notebooks for Study 802 (Stock-Split-Modern).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached split
calendar + price panel under ../_cache/ and otherwise quote the frozen headline numbers
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance split calendar
# + total-return prices, 44 forward splits 2010-08-24 -> 2025-11-17; market-adjusted
# abnormal CARs; as-of 2026-06-30).
R = dict(
    start="2010-08-24", end="2025-11-17", asof="2026-06-30",
    n_events=44, n_pre=19, n_post=25, n_mega=7, price_rows=5406,
    # abnormal 3m CAR by cohort: (n, mean_pct, median_pct, hac_t, plain_t, win_pct)
    all_n=44, all_mean=-1.83, all_median=-2.65, all_thac=-0.82, all_tplain=-0.91, all_win=45,
    pre_n=19, pre_mean=3.37, pre_thac=1.48, pre_win=63,
    post_n=25, post_mean=-5.78, post_median=-5.45, post_thac=-2.21, post_tplain=-2.12,
    post_std=13.62, post_win=32,
    mega_mean=-4.95, mega_thac=-0.77, mega_win=43,
    era_diff_welch=-2.42,
    # raw vs market vs abnormal (post cohort)
    post_raw3m=-2.95, post_mkt3m=2.83, post_car3m=-5.78,
    post_raw12m=21.63, post_mkt12m=16.18, post_car12m=5.44,
    # horizon sweep post: label -> (mean_pct, hac_t)
    sweep_post={"1m (21d)": (-3.69, -2.27), "3m (63d)": (-5.78, -2.21),
                "6m (126d)": (-2.58, -0.76), "12m (252d)": (5.44, 0.97)},
    sweep_all={"1m (21d)": (-1.40, -0.94), "3m (63d)": (-1.83, -0.82),
               "6m (126d)": (0.85, 0.34), "12m (252d)": (5.07, 1.56)},
    ex_post_mean=0.46, ex_post_t=0.66,
    # placebo (post, 3m)
    placebo_obs=-5.78, placebo_mean=4.81, placebo_sd=3.95, placebo_p=0.999, placebo_draws=1500,
    # timer (post, 3m, long)
    timer5_gross=-5.78, timer5_net=-5.88, timer5_t=-2.16, timer5_sharpe=-0.55,
    timer5_worst=-34.5, evyr=1.64,
    timer20_net=-6.18, timer20_t=-2.27,
    timer_mega_net=-5.05, timer_mega_t=-0.78,
    # the 7 headline mega-cap events: (ticker, date, ratio, abn_3m, abn_12m)
    mega_events=[("TSLA", "2020-08-31", 5, 9.8, 16.4), ("AAPL", "2020-08-31", 4, -11.6, -12.8),
                 ("NVDA", "2021-07-20", 4, 15.3, 2.9), ("AMZN", "2022-06-06", 20, 5.8, -8.2),
                 ("GOOGL", "2022-07-18", 20, -5.5, -9.2), ("TSLA", "2022-08-25", 3, -34.4, -26.6),
                 ("NVDA", "2024-06-10", 10, -14.0, 5.0)],
    syn_null_mean=0.45, syn_null_sd=0.79, syn_null_fire=0,
    syn_planted_t=4.05, syn_planted_mean=8.57,
    fp="f41f9d523e91",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Real inverse drift%3F: Busted](https://img.shields.io/badge/Real_inverse_drift%3F-Busted-8b949e?style=flat-square)\n\n"
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

from stock_split_modern import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    SPLITS = data.load_splits()
    PANEL = st.build_event_panel(PRICES, SPLITS)
else:
    PRICES = SPLITS = PANEL = None
print("real cache present:", HAVE_REAL, "| forward-split events:",
      (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Did the Tesla/Nvidia \"split magic\" ever really exist? ✂️\n"
            "### Stock-Split-Modern — the famous post-split drift, re-tested on the "
            "post-2020 mega-caps, with the bull market taken out\n\n"
            + BADGES +
            "For a few years it looked like a cheat code: **Tesla, Apple, Nvidia, Amazon "
            "and Alphabet** all announced stock splits, and all of them soared. \"Buy the "
            "splitters\" became a whole genre of trading-content. The old academic version "
            "of this — stocks drift *up* for a year after a split — is one of the more "
            "respectable market anomalies (Ikenberry, Rankine & Stice, 1996).\n\n"
            "So does it still work? Here's the catch nobody in those videos mentions: "
            "**everything** went up 2020-2024. To know if *splits* did anything, you have to "
            "subtract what the market did anyway. Once you do, the magic disappears.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** yfinance split calendar + total-return prices, 44 forward "
            "splits 2010→2025, returns measured *market-adjusted* (minus SPY). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks still drift **up** after a split? | **No.** Across all 44 large-cap "
            f"splits since 2010, the market-adjusted 3-month return averages "
            f"**{R['all_mean']:+.1f}%** — basically zero, and if anything slightly negative. |\n"
            "| What about the famous mega-caps? | **Seven events, and they cancel out.** "
            f"TSLA & NVDA's *first* splits flew (+10% to +15% market-adjusted); their *second* "
            f"splits (and AAPL/GOOGL) sank (down to −34%). The average is **{R['mega_mean']:+.1f}%** "
            "— statistically nothing. |\n"
            "| Was the \"magic\" just the bull market? | **Largely, yes.** The mega-caps rose "
            "because *everything* rose. Subtract the market and the split itself adds no "
            "reliable lift. |\n"
            "| Is there *any* modern pattern? | **A weak, backwards one.** Post-2020 splitters "
            f"slightly **underperform** the market for a month or two (a mild \"sell the news\"), "
            "but it's gone within six months and flips positive by a year — too flimsy to "
            "trade. |\n\n"
            "> The drift that made splits famous is a pre-2020 relic. In the mega-cap era it's "
            "tiny-N noise wearing a great story."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A split signals management's confidence and makes shares more accessible — "
            "so the stock keeps climbing afterwards. Look at Tesla and Nvidia.\"*\n\n"
            "The academic anchor is real: Ikenberry, Rankine & Stice (1996) found ~+8% abnormal "
            "drift in the year after a split **announcement**. The modern retail version points "
            "at the 2020-2024 mega-cap splitters as living proof. We take that seriously — and "
            "test it honestly."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a gift: splits are announced in advance, so you'd have a "
            "clean, scheduled buy signal on the biggest, most liquid names in the market. That's "
            "exactly the kind of too-good-to-be-true claim worth checking — because \"these "
            "famous stocks went up after X\" is *survivorship bias in a trenchcoat*: we remember "
            "the splitters precisely because they're today's giants."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Market-adjusted returns.** For every split we measure the stock's total return "
            "**minus SPY's** over the same window. This is the whole game: it removes \"the "
            "market went up anyway\" so only the *abnormal* move is left.\n"
            f"- **All the events, not the memorable ones.** {R['n_events']} forward splits since "
            f"2010 ({R['n_pre']} before 2020, {R['n_post']} after), and the "
            f"{R['n_mega']} mega-cap splits the story actually names.\n"
            "- **A fair comparison.** Enter the same names on *random* non-split dates — did the "
            "split date beat a coin toss?\n"
            "- **The trade check.** Buy the ex-date close, hold three months, pay realistic "
            "costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the seven mega-cap splits everyone remembers.** Here's the market-adjusted "
            "3-month return of each — the actual signal, with the bull market removed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mp = PANEL[(PANEL['era']=='post') & PANEL['is_mega']].copy()\n"
            "    labels = [f\"{r.ticker}\\n{str(r.event_date.date())[:7]} {int(r.ratio)}:1\" for r in mp.itertuples()]\n"
            "    vals = list(mp['car_3m']*100)\n"
            "else:\n"
            "    labels = [f\"{t}\\n{d[:7]} {int(rt)}:1\" for (t,d,rt,a3,a12) in R['mega_events']]\n"
            "    vals = [a3 for (t,d,rt,a3,a12) in R['mega_events']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "cols = [GREEN if v>=0 else RED for v in vals]\n"
            "ax.bar(range(len(vals)), vals, color=cols, width=.65)\n"
            "ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8.5)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "mean_v = float(np.mean(vals))\n"
            "ax.axhline(mean_v, ls='--', c=GREY, lw=1.2, label=f'mean {mean_v:+.1f}%')\n"
            "ax.set_ylabel('market-adjusted 3-month return')\n"
            "ax.set_title('The seven mega-cap splits: enormous spread, mean near zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('mega-cap 3m abnormal returns (%):', [round(v,1) for v in vals], '| mean', round(mean_v,2))"
        ),
        md(
            f"Seven events, ranging from **+15%** (Nvidia's 2021 split) to **−34%** (Tesla's "
            f"*second* split, August 2022), averaging **{R['mega_mean']:+.1f}%**. The winners you "
            "remember and the losers you forgot roughly cancel. That's not a signal — that's "
            "**seven coin flips with a memorable name attached.**\n\n"
            "**Now widen the lens to all 44 splits, split by era.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ct = st.cohort_table(PANEL, 'car_3m')\n"
            "    pre, post, allc = ct.loc['pre-2020','mean_pct'], ct.loc['post-2020 (modern)','mean_pct'], ct.loc['all 2010+','mean_pct']\n"
            "else:\n"
            "    pre, post, allc = R['pre_mean'], R['post_mean'], R['all_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "bars = ax.bar(['pre-2020\\n(n=19)','post-2020\\n(n=25)','all splits\\n(n=44)'],\n"
            "              [pre, post, allc], color=[AMBER, RED, GREY], width=.6)\n"
            "for i,v in enumerate([pre, post, allc]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean market-adjusted 3-month return')\n"
            "ax.set_title('The drift had the right sign before 2020 — then it flipped')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2020 {pre:+.2f}% | post-2020 {post:+.2f}% | all {allc:+.2f}%')"
        ),
        md(
            f"Before 2020 the drift at least pointed the *right* way (**{R['pre_mean']:+.1f}%**) — "
            "though even then it wasn't strong enough to bank on. After 2020 the sign **flips "
            f"negative** (**{R['post_mean']:+.1f}%**). Pooled across all 44 splits it's "
            f"**{R['all_mean']:+.1f}%** — the famous upward drift simply isn't there.\n\n"
            "**But wait — that post-2020 number is negative. Is *that* a signal?** Only if it "
            "lasts. Let's watch it over time."
        ),
        code(
            "labels = list(R['sweep_post'].keys())\n"
            "if HAVE_REAL:\n"
            "    hs = st.horizon_sweep(PANEL, 'post')\n"
            "    means = list(hs['mean_pct'])\n"
            "else:\n"
            "    means = [R['sweep_post'][l][0] for l in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if v<0 else GREEN for v in means]\n"
            "ax.bar(labels, means, color=cols, width=.6)\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean market-adjusted return (post-2020)')\n"
            "ax.set_title('The \"dip\" evaporates: negative early, gone by 6m, positive by 12m')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('post-2020 abnormal CAR by horizon (%):', {l: round(m,2) for l,m in zip(labels, means)})"
        ),
        md(
            "The modern underperformance is a **short-term** thing — down a few percent in the "
            "first month or two — that **fades to nothing by six months and turns positive by "
            "twelve.** A signal that reverses its own sign depending on how long you wait isn't a "
            "signal; it's noise you've stared at too long.\n\n"
            "**Last check: was the split date even a *good entry* versus a random day?**"
        ),
        code(
            "obs = R['placebo_obs']; pm, ps = R['placebo_mean'], R['placebo_sd']\n"
            "rng = np.random.default_rng(802)\n"
            "draws = rng.normal(pm, ps, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random non-split entry dates (same names)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'actual split dates {obs:+.1f}%')\n"
            "ax.axvline(pm, c='k', lw=1, ls='--')\n"
            "ax.set_xlabel('mean market-adjusted 3-month return (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title('Split dates were a WORSE entry than random dates')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed split-date mean {obs:+.2f}% vs random-date mean {pm:+.2f}% -> split beat random almost never (p={R[\"placebo_p\"]:.3f})')"
        ),
        md(
            f"Buying these names on a **random** day beat buying them on their **split** day "
            f"(random averaged **{R['placebo_mean']:+.1f}%**, splits **{R['placebo_obs']:+.1f}%**). "
            "If the split carried a bullish signal, this would be the other way around. It's a "
            "mild \"sell the news\", not a buy signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The classic *upward* post-split drift is gone: all 44 splits "
            f"average **{R['all_mean']:+.1f}%** market-adjusted at 3 months, and the seven "
            "mega-caps that inspired the whole story average a statistically-nothing "
            f"**{R['mega_mean']:+.1f}%**.\n"
            "- **Tradability — Mirage.** ~1.6 events a year, no reliable edge in either "
            "direction, and one Tesla-2022 crash (−34%) swamps the average. Nothing to trade.\n"
            "- **Is the modern *dip* a real inverse signal? — Busted.** It's there for a month "
            "or two, then evaporates and flips positive — and it vanishes entirely on the actual "
            "mega-caps. A tiny-sample quirk, not a rule."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The honest caveat that keeps this fair:** yfinance gives the split's "
            "*effective* (ex) date, not the *announcement* (which comes ~3-6 weeks earlier). The "
            "1996 drift was measured from the announcement, so it's plausible any real edge is "
            "already priced by the ex-date. We test the weaker window and say so.\n"
            "- **Selection bias runs *for* the claim, not against it:** we're looking at today's "
            "winners. If even a hand-picked basket of survivors shows no drift, a random splitter "
            "certainly won't.\n"
            "- **Sibling studies:** [142-split-drift](../../142-split-drift/) tests the same "
            "anomaly on a 2000-2025 basket (`NONE`); [250-reverse-split](../../250-reverse-split/) "
            "tests the *opposite* action, the \"kiss of death\" reverse split. See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the announcement-window drift is alive where the ex-date one is dead? Pull "
            "the SEC 8-K announcement dates and show a certifiable, cost-surviving abnormal "
            "return — then we'll talk.*"
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
            "# Stock-Split-Modern — a quantitative teardown 🔬\n"
            "### Market-adjusted abnormal CARs · a HAC horizon sweep · the pre/post-2020 era "
            "difference · a random-date placebo · a costed long/short timer · survivorship "
            "accounting · a seeded synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **the classic positive post-split drift survives in the post-2020 mega-cap "
            "era** — is re-tested with an **SPY-hedged, total-return abnormal-return** lens that "
            "de-trends the 2020-24 bull market, on *all* 44 forward splits since 2010, not the "
            "memorable few. The decisive statistic is a Newey-West HAC one-sample *t* on the "
            "date-ordered CARs.\n\n"
            "> ⚠️ **Data note.** yfinance `.splits` (forward, ratio ≥ 1.5) + total-return "
            "(`auto_adjust`) closes for a 30-name large-cap basket **and SPY**, 2010→2025, "
            "cached. **Survivorship/selection is named on the Signal axis** — this is a "
            "*current-membership* basket of today's winners, a bias that points *for* the drift. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | all-splits abnormal 3m CAR **{R['all_mean']:+.2f}%**, "
            f"HAC **t = {R['all_thac']:.2f}**; pre-2020 {R['pre_mean']:+.2f}% (t = {R['pre_thac']:.2f}, "
            f"uncertifiable); post-2020 {R['post_mean']:+.2f}% (t = {R['post_thac']:.2f}); "
            f"era-difference Welch t = {R['era_diff_welch']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | long timer net {R['timer5_net']:+.2f}% "
            f"(t = {R['timer5_t']:.2f}), ~{R['evyr']:.1f} events/yr, worst {R['timer5_worst']:.1f}%; "
            f"mega-cap net {R['timer_mega_net']:+.2f}% (t = {R['timer_mega_t']:.2f}) |\n"
            f"| **Real inverse drift?** | `BUSTED` | post-2020 3m t = {R['post_thac']:.2f} but "
            f"6m t = {R['sweep_post']['6m (126d)'][1]:.2f}, 12m t = {R['sweep_post']['12m (252d)'][1]:.2f}; "
            f"mega-cap t = {R['mega_thac']:.2f} |\n\n"
            "> 💡 In plain words: the upward drift is absent everywhere; the post-2020 *negative* "
            "blip clears |t| = 2 only at short horizons, reverses by a year, and disappears on the "
            "seven names the story is actually about."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{s}_{[0,h]}$ be stock $s$'s total return from its ex-split close to $h$ "
            "trading days later, and $r^{m}_{[0,h]}$ SPY's over the identical window. The "
            "**abnormal CAR** is $\\text{CAR}^s_h = r^{s}_{[0,h]} - r^{m}_{[0,h]}$. The claims:\n\n"
            "- **H₁ (drift).** $E[\\text{CAR}_h] > 0$ for $h$ up to 12 months — stocks drift up "
            "*abnormally* after a split.\n"
            "- **H₂ (modern).** The effect holds in the **post-2020** cohort, and specifically on "
            "the mega-cap splitters (TSLA/NVDA/AMZN/GOOGL/AAPL).\n"
            "- **H₃ (capture).** A timer that buys the ex-date close banks it net of costs.\n\n"
            f"We find **H₁ rejected** (all-splits CAR₃ₘ {R['all_mean']:+.2f}%, HAC t = "
            f"{R['all_thac']:.2f}; not even the pre-2020 slice certifies, t = {R['pre_thac']:.2f}), "
            f"**H₂ rejected and reversed** (post-2020 {R['post_mean']:+.2f}%, t = {R['post_thac']:.2f}; "
            f"mega-cap {R['mega_mean']:+.2f}%, t = {R['mega_thac']:.2f}), **H₃ rejected** (the long "
            f"timer loses; the short leg is un-borrowable)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Split events **cluster in calendar time** (the 2022 wave especially), so a plain "
            "i.i.d. one-sample *t* overstates independence. The planned primary is a "
            "**Newey-West (Bartlett) HAC one-sample *t*** on the **date-ordered** abnormal CARs. "
            "We report it at four horizons (21/63/126/252d) for the *all*, *pre*, *post* and "
            "*mega* cohorts; the era split (2020-01-01) is tested as a **Welch difference**; the "
            "hit rate carries a **Wilson interval**; and a **random-non-split-date placebo** "
            "(same names, matched horizon) asks whether the split date beat a coin toss. "
            "Market-adjustment removes the bull-market confound; survivorship/selection bias "
            "**remains and points *for* H₁**, so a `NONE` here is conservative."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** {R['n_events']} forward splits (ratio ≥ 1.5) since 2010 with a usable "
            f"3m window; {R['n_pre']} pre-2020, {R['n_post']} post-2020, {R['n_mega']} mega-cap.\n"
            "- **Measure.** Abnormal CAR = stock − SPY total return, entered at the ex-date close "
            "(first earned return t+1 — the single documented lag), at 1/3/6/12m + a [−1, +1] "
            "around-ex window.\n"
            "- **Headline.** HAC t per cohort × horizon; Welch t of the era difference; Wilson "
            "win rate; random-date placebo.\n"
            "- **Execution (timer).** Buy the ex-date close, hold the horizon, 2 × one-way cost × "
            "NAV; long-only for forward splits (the short leg is graded as un-borrowable).\n"
            "- **Control.** Synthetic raw-stock + market paths, planted-abnormal-drift knob; the "
            "null must not fire across 12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cohort table — the upward drift is nowhere\n\n"
            "Market-adjusted 3-month abnormal CAR, by cohort, with the HAC *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ct = st.cohort_table(PANEL, 'car_3m')[['n','mean_pct','median_pct','t_hac','t_plain','win']]\n"
            "    print(ct.round(2).to_string())\n"
            "    rawpost = PANEL[PANEL['era']=='post']\n"
            "    raw3, mkt3, car3 = rawpost['raw_3m'].mean()*100, rawpost['mkt_3m'].mean()*100, rawpost['car_3m'].mean()*100\n"
            "else:\n"
            "    raw3, mkt3, car3 = R['post_raw3m'], R['post_mkt3m'], R['post_car3m']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['raw stock\\nreturn','SPY (market)\\nreturn','ABNORMAL\\n(stock - SPY)'],\n"
            "       [raw3, mkt3, car3], color=[GREY, '#5b8def', RED], width=.6)\n"
            "for i,v in enumerate([raw3, mkt3, car3]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean 3-month return, post-2020 cohort')\n"
            "ax.set_title('Market-adjustment is the whole story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post-2020 3m: raw {raw3:+.2f}%  market {mkt3:+.2f}%  abnormal {car3:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: pooled over all 44 splits the abnormal CAR₃ₘ is "
            f"**{R['all_mean']:+.2f}% (HAC t = {R['all_thac']:.2f})** — no drift. The pre-2020 "
            f"slice has the right sign (**{R['pre_mean']:+.2f}%**) but only **t = {R['pre_thac']:.2f}** "
            f"on 19 events — the anomaly can't even certify on its *home* era here. Post-2020 the "
            f"mean goes **{R['post_mean']:+.2f}%**, and the Welch t of the pre→post difference is "
            f"**{R['era_diff_welch']:.2f}**: a genuine regime change *away* from the drift."
        ),
        md(
            "### 4b · The HAC horizon sweep — the modern dip is fragile\n\n"
            "The post-2020 cohort's abnormal CAR and HAC *t* across all four horizons, next to "
            "the all-cohort sweep."
        ),
        code(
            "labels = list(R['sweep_post'].keys())\n"
            "if HAVE_REAL:\n"
            "    hp = st.horizon_sweep(PANEL, 'post'); ha = st.horizon_sweep(PANEL, 'all')\n"
            "    tp = list(hp['t_hac']); ta = list(ha['t_hac'])\n"
            "else:\n"
            "    tp = [R['sweep_post'][l][1] for l in labels]; ta = [R['sweep_all'][l][1] for l in labels]\n"
            "x = np.arange(len(labels)); w = .38\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x-w/2, tp, width=w, color=RED, label='post-2020 (modern)')\n"
            "ax.bar(x+w/2, ta, width=w, color=GREY, label='all splits 2010+')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('HAC one-sample t (abnormal CAR)')\n"
            "ax.set_title('Only the short-horizon post-2020 bars clear |t| = 2 — and they are negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('post-2020 HAC t:', {l: round(v,2) for l,v in zip(labels, tp)})\n"
            "print('all-splits HAC t:', {l: round(v,2) for l,v in zip(labels, ta)})"
        ),
        md(
            f"> 💡 In plain words: the post-2020 cohort clears |t| = 2 only at 1m "
            f"(t = {R['sweep_post']['1m (21d)'][1]:.2f}) and 3m (t = {R['sweep_post']['3m (63d)'][1]:.2f}) — "
            f"and *negative*. By 6m it's t = {R['sweep_post']['6m (126d)'][1]:.2f} and by 12m it has "
            f"flipped to **{R['sweep_post']['12m (252d)'][0]:+.2f}% (t = {R['sweep_post']['12m (252d)'][1]:.2f})**. "
            f"The *all-splits* sweep clears the bar at **no** horizon. Around the ex-date itself "
            f"the modern cohort is a flat {R['ex_post_mean']:+.2f}% (t = {R['ex_post_t']:.2f})."
        ),
        md(
            "### 4c · The placebo — was the split date even a good entry?\n\n"
            "Same names, entered on **random non-split dates**, market-adjusted; 1,500 baskets. "
            "In-notebook we draw from the frozen placebo moments; the canonical p is in "
            "`results.md`."
        ),
        code(
            "obs = R['placebo_obs']; pm, ps = R['placebo_mean'], R['placebo_sd']\n"
            "rng = np.random.default_rng(802)\n"
            "draws = rng.normal(pm, ps, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random non-split dates (same names)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed split-date mean {obs:+.1f}%')\n"
            "ax.set_xlabel('mean market-adjusted 3-month return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Split dates underperformed random dates (p = {R[\"placebo_p\"]:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}% vs placebo mean {pm:+.2f}% (sd {ps:.2f}); random beat split ~{R[\"placebo_p\"]*100:.1f}% of the time')"
        ),
        md(
            f"> 💡 In plain words: the split-date basket returned **{R['placebo_obs']:+.2f}%** vs "
            f"**{R['placebo_mean']:+.2f}%** for random dates in the same names — the split date was "
            f"a *worse* entry (**p = {R['placebo_p']:.3f}**). A bullish signal would show the "
            "reverse. Consistent with a mild 'sell the news', inconsistent with post-split drift."
        ),
        md(
            "### 4d · The timer — an honest cost sweep\n\n"
            "Buy the ex-date close, hold 3 months, market-hedged (long the stock, short SPY), "
            "2 × one-way cost per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t5 = st.timer_stats(PANEL, 'car_3m', cost_bps=5.0, cohort='post')\n"
            "    t20 = st.timer_stats(PANEL, 'car_3m', cost_bps=20.0, cohort='post')\n"
            "    tm = st.timer_stats(PANEL, 'car_3m', cost_bps=5.0, cohort='mega')\n"
            "    g, n5, n20 = t5['gross_pct'], t5['net_pct'], t20['net_pct']\n"
            "    tt5, ttm = t5['t_net'], tm['t_net']; nm = tm['net_pct']\n"
            "else:\n"
            "    g, n5, n20 = R['timer5_gross'], R['timer5_net'], R['timer20_net']\n"
            "    tt5, ttm, nm = R['timer5_t'], R['timer_mega_t'], R['timer_mega_net']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['gross','net @5bps','net @20bps','mega-cap\\nnet @5bps'], [g, n5, n20, nm],\n"
            "       color=[GREY, RED, RED, AMBER], width=.6)\n"
            "for i,v in enumerate([g, n5, n20, nm]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net return per event (%)')\n"
            "ax.set_title('The long timer loses; the mega-cap leg is statistically nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post long: gross {g:+.2f}% net@5 {n5:+.2f}% (t={tt5:+.2f}) net@20 {n20:+.2f}% | mega net {nm:+.2f}% (t={ttm:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the long timer nets **{R['timer5_net']:+.2f}%** per event "
            f"(t = {R['timer5_t']:.2f}) — it *loses*, because the modern abnormal return is "
            f"negative. Flipping to a **short** would mean borrowing high-beta mega-caps and "
            f"eating a **{R['timer5_worst']:.1f}%** single-event tail (Tesla, Aug-2022) on "
            f"~{R['evyr']:.1f} events a year — un-scalable, and often un-borrowable. On the seven "
            f"actual mega-caps the timer nets **{R['timer_mega_net']:+.2f}% (t = {R['timer_mega_t']:.2f})** "
            "— nothing. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic raw stock + market price paths with synthetic split dates and a TUNABLE "
            "planted **abnormal** post-split drift, run through the *same* market-adjusted panel. "
            "The null (planted = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    pr, mk, sp = data.synthetic_world(planted_bps=0.0, seed=802 + s_)\n"
            "    null_ts.append(st.synthetic_detect(pr, sp)['t_hac'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "pr, mk, sp = data.synthetic_world(planted_bps=8.0, seed=802)\n"
            "planted_t = st.synthetic_detect(pr, sp)['t_hac']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=42, label='null worlds (planted=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted +8 bps/day abnormal')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 12', 'planted'])\n"
            "ax.set_ylabel('HAC one-sample t (abnormal CAR)')\n"
            "ax.set_title('Control: no null fires; a planted abnormal drift lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires |t| ≥ 2 in "
            f"{R['syn_null_fire']}/12 seeds; a planted +8 bps/day abnormal drift reads "
            f"t = {R['syn_planted_t']:.2f} (mean CAR {R['syn_planted_mean']:+.2f}%). The machinery "
            "is unbiased and powered — so the real-tape *absence* of positive drift is genuine, "
            "not a dead pipeline. *(A faithful-engine / power check only — never cited in support "
            "of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — all-splits abnormal CAR₃ₘ **{R['all_mean']:+.2f}%** "
            f"(HAC t = {R['all_thac']:.2f}); pre-2020 {R['pre_mean']:+.2f}% (t = {R['pre_thac']:.2f}, "
            f"right sign but uncertifiable); post-2020 {R['post_mean']:+.2f}% (t = {R['post_thac']:.2f}); "
            f"era-difference Welch t = {R['era_diff_welch']:.2f}. No cohort clears **t ≥ 2 "
            "positive** on the real tape; if anything the anomaly reversed.\n"
            f"- **Tradability `MIRAGE`** — long timer net {R['timer5_net']:+.2f}% "
            f"(t = {R['timer5_t']:.2f}), ~{R['evyr']:.1f} events/yr, worst {R['timer5_worst']:.1f}%; "
            f"mega-cap net {R['timer_mega_net']:+.2f}% (t = {R['timer_mega_t']:.2f}); the short leg "
            "is un-borrowable/un-scalable.\n"
            f"- **Real inverse drift? `BUSTED`** — the post-2020 −5.78% clears |t| = 2 only at "
            f"1-3m, reverses to {R['sweep_post']['12m (252d)'][0]:+.2f}% by 12m, and vanishes on "
            f"the seven mega-caps (t = {R['mega_thac']:.2f}). A horizon-fragile, selection-biased "
            "'sell-the-news' blip, not a tradable inverse drift."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Effective vs announcement.** yfinance gives the *ex* date; the 1996 drift is "
            "measured from the *announcement* (~3-6 weeks earlier). The ex-date result being dead "
            "is consistent with the announcement signal being fully priced by the ex-date — the "
            "clean follow-up is to pull SEC 8-K announcement dates and re-run.\n"
            "- **Why the modern sign flips.** Post-2020 mega-cap splits arrived *after* enormous "
            "run-ups, into crowded, options-heavy names — a setup for near-term mean reversion "
            "('sell the news'), not continuation. Testable: does the dip scale with the pre-split "
            "run-up?\n"
            "- **Dedup map:** [142-split-drift](../../142-split-drift/) (same anomaly, 2000-2025 "
            "basket, `NONE`), [250-reverse-split](../../250-reverse-split/) (the opposite corporate "
            "action, distress-confounded `WEAK`/`Mirage`).\n\n"
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
