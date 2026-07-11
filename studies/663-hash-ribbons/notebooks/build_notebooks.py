"""Generate the two narrative notebooks for Study 663 (Hash-Ribbons).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached BTC-USD tape
under ../_cache/ and the hardcoded/interpolated hashrate path (no network either way), else
quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (BTC-USD yfinance 2014-09-17
# -> 2026-06-30; hardcoded month-end hashrate table interpolated to daily; 4 genuine signals).
R = dict(
    hr_lo="2014-01-31", hr_hi="2026-05-31", hr_n=4504,
    btc_lo="2014-09-17", btc_hi="2026-06-30", btc_n=4305,
    fp_btc="9529d5277775",
    n_signals=4,
    signals=[
        dict(date="2019-02-12", days=105, window="2018-10-30 -> 2019-02-11", decline=-19.7),
        dict(date="2020-04-30", days=30, window="2020-03-31 -> 2020-04-29", decline=-8.1),
        dict(date="2020-06-30", days=31, window="2020-05-30 -> 2020-06-29", decline=-8.1),
        dict(date="2021-08-15", days=89, window="2021-05-18 -> 2021-08-14", decline=-43.9),
    ],
    # horizon -> (signal mean %, signal median %, unconditional mean %, unconditional sd %,
    #             Welch t, placebo p)
    fwd={
        30: (10.3, 9.7, 5.90, 22.93, 0.97, 0.3076),
        90: (49.1, 32.6, 21.06, 53.69, 1.27, 0.1358),
        180: (112.7, 122.6, 51.46, 104.23, 1.15, 0.1278),
        365: (246.1, 233.0, 146.83, 268.65, 0.78, 0.1815),
    },
    placebo_draws=20000,
    # timer vs buy-and-hold
    n_entries=3, years=7.4,
    strat_total=769.8, strat_cagr=34.07, strat_sharpe=1.10, exposure=22.4,
    bh_total=1502.8, bh_cagr=45.65, bh_sharpe=0.92,
    # synthetic control
    syn_null_mean=-0.27, syn_null_sd=1.29, syn_null_fire=2, syn_planted_t=4.03,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Beats_random_buy_dates%3F: Busted](https://img.shields.io/badge/Beats_random_buy_dates%3F-Busted-8b949e?style=flat-square)\n\n"
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

from hash_ribbons import data, strategy as st

HR = data.hashrate_daily()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    BTC = data.load_btc()
    SIG = st.capitulation_signals(HR)
else:
    BTC = SIG = None
print("real cache present:", HAVE_REAL, "| hashrate path points:", len(HR),
      "| signals:", (0 if SIG is None else len(SIG)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Miners are capitulating. Is that when you buy? ⛏️🎗️\n"
            "### Hash Ribbons — a genuinely clever signal that has only fired **four** times "
            "in Bitcoin's history\n\n"
            + BADGES +
            "Bitcoin mining is a brutal, thin-margin business. When the price drops or the "
            "network gets more competitive, the *weakest* miners — the ones with the oldest "
            "machines and the most expensive electricity — get shut off first. Some of them "
            "sell their Bitcoin stash on the way out, just to cover the bills.\n\n"
            "**Hash Ribbons** watches for exactly that: a real, hard-to-fake drop in the "
            "network's total computing power (\"hashrate\"), followed by a recovery. The claim: "
            "once the weak hands are gone and hashrate turns back up, the selling pressure is "
            "over — and that moment has historically been a great time to buy.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats and the placebo test? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** This signal has fired **4 times** since 2014. Everything "
            "below should be read with that number tattooed on your forearm. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How many times has this actually fired? | **Just {R['n_signals']}**, in over "
            "twelve years of Bitcoin history: 2019, twice in 2020, and 2021. |\n"
            "| Did BTC rally afterward? | **Every single time**, at every horizon we checked "
            f"(1, 3, 6 and 12 months) — averaging **+{R['fwd'][180][0]:.0f}%** over the six "
            "months after a signal, versus **+" + f"{R['fwd'][180][2]:.0f}%" + "** in a "
            "randomly-chosen six months. |\n"
            "| So... it works? | **We can't say that.** With only 4 events, a coin-flip-picked "
            "calendar of 4 random dates does as well or better than the signal **about 1 time "
            "in 8** — nowhere near rare enough to call it proof. |\n"
            "| Could you have traded it? | You'd have beaten buy-and-hold on *risk-adjusted* "
            f"terms (Sharpe {R['strat_sharpe']:.2f} vs {R['bh_sharpe']:.2f}) — but you'd have "
            f"ended up with **less than half** the money (+{R['strat_total']:.0f}% vs "
            f"+{R['bh_total']:.0f}%) because you'd have been in cash "
            f"{100 - R['exposure']:.0f}% of the time, missing most of the ride. |\n\n"
            "> Four data points is not a strategy. It's an interesting coincidence with a "
            "believable story attached."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When miners capitulate — the weakest ones shutting off their machines and "
            "dumping their BTC to survive — the selling pressure that was weighing on price "
            "disappears right as the strongest miners consolidate control. Watch the 30-day "
            "hashrate average cross back above the 60-day average: that's the all-clear.\"*\n\n"
            "It's popular precisely because it isn't price-based circular reasoning — hashrate "
            "is a real commitment of capital (chips, power contracts) that can't be faked the "
            "way trading volume sometimes can."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be one of the purest \"buy the capitulation\" signals in "
            "crypto — grounded in the physical economics of mining rather than chart pattern "
            "recognition or sentiment surveys. Every crypto-analytics dashboard has a Hash "
            "Ribbons widget; it's cited constantly as a bottom-calling tool. The stakes: if it "
            "genuinely times entries better than just holding, that's real, bankable alpha in "
            "the world's most liquid crypto asset. If it doesn't, it's one more chart that "
            "*looks* rigorous because it uses an on-chain metric instead of price."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The signal.** Every time Bitcoin's 30-day hashrate average, after sitting "
            "below the 60-day average for at least three weeks *and* falling at least 8% "
            "peak-to-trough, crosses back above it.\n"
            "- **The test.** What did BTC actually do in the 30/90/180/365 days after each "
            "signal, versus a *random* day picked from the same twelve years?\n"
            "- **The luck check.** Draw 4 random dates instead of the real signal dates, "
            "20,000 times — how often does a random 4-date calendar do just as well?\n"
            "- **The trade check.** If you'd gone long BTC for 180 days after every signal and "
            "sat in cash otherwise, how would that compare to just buying and holding?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how rare is this, really?** Here's the hashrate path with every genuine "
            "capitulation-and-recovery highlighted."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig_dates = list(SIG['signal_date'])\n"
            "else:\n"
            "    sig_dates = pd.to_datetime([s['date'] for s in R['signals']])\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.6))\n"
            "ax.plot(HR.index, HR.values, color=GREY, lw=1.2, label='hashrate (EH/s, interpolated)')\n"
            "ax.set_yscale('log')\n"
            "for d in sig_dates:\n"
            "    ax.axvline(d, color=RED, lw=1.4, alpha=.85)\n"
            "ax.scatter(sig_dates, [HR.reindex(sig_dates, method='nearest').loc[d] for d in sig_dates],\n"
            "           color=RED, s=60, zorder=5, label=f'buy signal (n={len(sig_dates)})')\n"
            "ax.set_ylabel('hashrate, EH/s (log scale)')\n"
            "ax.set_title('Twelve years of Bitcoin mining — four genuine capitulations')\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print('signal dates:', [d.date() for d in sig_dates])"
        ),
        md(
            f"Four lines on twelve years of data. Each one marks a real, sizeable hashrate "
            "drop (8% to 44% peak-to-trough) followed by a recovery — 2018's mining bear "
            "market resolving in early 2019, the COVID crash and the May 2020 halving, and the "
            "2021 China mining-ban exodus. **Now: what happened to price next?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.forward_return_stats(BTC, SIG['signal_date'])\n"
            "    sig_means = [fr.loc[h, 'signal_mean_pct'] for h in [30, 90, 180, 365]]\n"
            "    unc_means = [fr.loc[h, 'uncond_mean_pct'] for h in [30, 90, 180, 365]]\n"
            "else:\n"
            "    sig_means = [R['fwd'][h][0] for h in [30, 90, 180, 365]]\n"
            "    unc_means = [R['fwd'][h][2] for h in [30, 90, 180, 365]]\n"
            "labels = ['+30d', '+90d', '+180d', '+365d']\n"
            "x = np.arange(len(labels)); w = 0.36\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(x - w/2, sig_means, width=w, color=RED, label='after a signal')\n"
            "ax.bar(x + w/2, unc_means, width=w, color=GREY, label='a random day')\n"
            "for i, v in enumerate(sig_means): ax.annotate(f'{v:+.0f}%', (x[i]-w/2, v), ha='center', va='bottom')\n"
            "for i, v in enumerate(unc_means): ax.annotate(f'{v:+.0f}%', (x[i]+w/2, v), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('mean forward BTC return')\n"
            "ax.set_title('Every horizon: ahead of average. But by how much can we trust it?')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"Every bar is bigger after a signal than on a random day — at 6 months, "
            f"**+{R['fwd'][180][0]:.0f}%** vs **+{R['fwd'][180][2]:.0f}%**. Looks great. Here's "
            "the catch — with only 4 events, how do we know that gap isn't just luck?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    p180 = fr.loc[180, 'placebo_p']\n"
            "else:\n"
            "    p180 = R['fwd'][180][5]\n"
            "rng = np.random.default_rng(663)\n"
            "draws = rng.normal(R['fwd'][180][2], R['fwd'][180][3], 3000)  # illustrative null shape\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='4 random dates, drawn many times')\n"
            "ax.axvline(R['fwd'][180][0], color=RED, lw=2.5, label=f\"observed signal mean +{R['fwd'][180][0]:.0f}%\")\n"
            "ax.set_xlabel('mean 180-day forward BTC return of a 4-date draw')\n"
            "ax.set_ylabel('frequency (illustrative)')\n"
            "ax.set_title(f'A random 4-date calendar beats the signal about 1 time in 8 (p = {p180:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'canonical placebo (results.md, 180d, {R[\"placebo_draws\"]:,} draws): p = {p180:.4f}')"
        ),
        md(
            "That's the honest picture: the signal's 6-month result sits inside the cloud a "
            "random calendar produces about **one time in eight** — nowhere close to the kind "
            "of \"this almost never happens by chance\" result the desk needs to call something "
            "**Real**. Now the trade check — would you have actually made money running this?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tb = st.timer_backtest(BTC, SIG['signal_date'], hold_days=180, cost_bps=10.0)\n"
            "    st_tot, bh_tot = tb['strat_total_pct'], tb['bh_total_pct']\n"
            "    st_shp, bh_shp = tb['strat_sharpe'], tb['bh_sharpe']\n"
            "else:\n"
            "    st_tot, bh_tot = R['strat_total'], R['bh_total']\n"
            "    st_shp, bh_shp = R['strat_sharpe'], R['bh_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['timer\\n(in 180d after signal)', 'buy & hold'], [st_tot, bh_tot], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([st_tot, bh_tot]): a1.annotate(f'{v:+.0f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('total return since the first signal'); a1.set_title('Total wealth: timer loses big')\n"
            "a2.bar(['timer', 'buy & hold'], [st_shp, bh_shp], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([st_shp, bh_shp]): a2.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('Sharpe ratio'); a2.set_title('Risk-adjusted: timer actually wins')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer {st_tot:+.0f}% (Sharpe {st_shp:.2f})  vs  buy&hold {bh_tot:+.0f}% (Sharpe {bh_shp:.2f})')"
        ),
        md(
            f"A split decision. Being long only **{R['exposure']:.0f}%** of the time, but in "
            "the *right* 22%, genuinely smooths the ride — the Sharpe ratio is actually "
            f"**better** than buy-and-hold ({R['strat_sharpe']:.2f} vs {R['bh_sharpe']:.2f}). "
            "But BTC spends most of its time going up, and sitting in cash the other 78% of "
            f"the time means the timer ends with **less than half** the wealth "
            f"(+{R['strat_total']:.0f}% vs +{R['bh_total']:.0f}%). The signal doesn't beat "
            "buy-and-hold; it trades some upside for a smoother ride, on a strategy built from "
            "**3 actual holding periods**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Every horizon after a Hash Ribbons signal beat the "
            "unconditional average, but with only 4 (partly overlapping) events, nothing "
            "clears the bar for \"statistically real\" — the best placebo result still lets a "
            "random calendar win 1 time in 8.\n"
            "- **Tradability — Fragile.** A timer built on the signal has a real Sharpe edge "
            "but gives up more than half the total wealth versus simply holding, built on just "
            "3 distinct trades.\n"
            "- **\"Beats random buy dates?\" — Busted.** Directionally ahead everywhere, never "
            "statistically proven anywhere."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The n=4 problem is the whole story.** No amount of clever statistics turns "
            "four historical episodes into a certified edge. The honest use of Hash Ribbons is "
            "as a *narrative* input (\"miners just capitulated, that's historically been a good "
            "sign\") — not as a mechanical, backtestable trading signal.\n"
            "- **What would change our mind:** true daily hashrate data (rather than an "
            "interpolated monthly series) and, mostly, *time* — another 5-10 genuine signals "
            "on the tape before the statistics have anything real to say.\n"
            "- **Sibling studies:** [292-bitcoin-hashrate](../../292-bitcoin-hashrate/) already "
            "shows continuous hashrate-growth has **zero** predictive power for next-month "
            "returns — this study's narrower, event-based question gets a softer but still "
            "unproven answer.\n\n"
            "*Think the 30/60-day windows or the 8% capitulation filter are cherry-picked? "
            "Fork the repo, try your own thresholds, and see if a different rule finds a "
            "cleaner signal in the same four historical windows — or a fifth one we missed.*"
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
            "# Hash-Ribbons — a quantitative teardown 🔬\n"
            "### The crossover-detection filter · Welch/placebo forward-return tests at four "
            "horizons · a 180-day timer vs buy-and-hold · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — a 30d/60d hashrate-SMA crossover marks the end of miner capitulation "
            "and precedes above-average BTC returns — is tested as a **discrete event study**, "
            "not a continuous exposure rule (that question already belongs to sibling study "
            "292-bitcoin-hashrate). The headline constraint, said loudly and often: **n = 4**.\n\n"
            "> ⚠️ **Data note.** Hardcoded month-end hashrate table (identical to "
            "292-bitcoin-hashrate), linearly interpolated to daily — smooths intra-month noise, "
            "a named limitation. BTC-USD daily close, yfinance, 2014-09-17 → 2026-06-30 "
            "(fingerprint `" + R["fp_btc"] + "`). No survivorship concept on the hashrate "
            "series itself (a single network's own metric); BTC-USD's single-survivor "
            "character is named on the Tradability axis. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | best Welch **t = {R['fwd'][90][4]:.2f}** (90d), best "
            f"placebo **p = {R['fwd'][180][5]:.3f}** (180d), n = {R['n_signals']} |\n"
            f"| **Tradability** | `FRAGILE` | timer Sharpe **{R['strat_sharpe']:.2f}** vs B&H "
            f"**{R['bh_sharpe']:.2f}**, but total return **+{R['strat_total']:.0f}%** vs "
            f"**+{R['bh_total']:.0f}%** on {R['n_entries']} holding episodes |\n"
            f"| **Beats random dates?** | `BUSTED` | placebo *p* never below "
            f"{min(v[5] for v in R['fwd'].values()):.3f} at any of the 4 horizons |\n\n"
            "> 💡 In plain words: the pattern is directionally consistent everywhere and "
            "certified nowhere — exactly what four data points should produce, whether or not "
            "there's a real effect underneath."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $S = \\{t_1, \\dots, t_n\\}$ be the set of Hash-Ribbons buy-signal dates and "
            "$r_{t,h}$ the $h$-day forward log-to-simple BTC return from day $t$. The claim:\n\n"
            "- **H₁ (predictive).** $E[r_{t,h} \\mid t \\in S] > E[r_{t,h}]$ (the unconditional "
            "mean) at economically meaningful horizons.\n"
            "- **H₂ (tradable).** A mechanical timer built on $S$ beats continuous "
            "buy-and-hold net of costs.\n\n"
            f"With $n = {R['n_signals']}$, this is fundamentally an **underpowered** test — a "
            "single additional or missing historical episode would materially move every "
            "statistic below. We report the numbers honestly and let the sample size speak for "
            "itself rather than dressing up a 4-point estimate as a certified finding."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Because $n$ is tiny and the events are not independent (two of the four signals "
            "sit two months apart in the same 2020 recovery), a Welch *t* against the "
            "unconditional distribution is reported but is explicitly **not** the primary "
            "arbiter — it is included because it's the desk's standard statistic, with a loud "
            "caveat that it is barely informative at this sample size. The **less "
            "parametric-dependent** check is a random-date placebo: draw $n$ random calendar "
            "dates (no replacement), repeat 20,000 times (20 seeds × 1,000 draws), and ask how "
            "often a random draw's mean forward return matches or beats the observed signal "
            "mean. That test only assumes we can resample dates fairly — it doesn't lean on "
            "small-sample normal-theory asymptotics the way a *t*-test does."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Hashrate path.** {R['hr_n']:,} daily points {R['hr_lo']} → {R['hr_hi']}, "
            "linear interpolation of the hardcoded month-end table.\n"
            "- **Signal filter.** SMA(30) crosses back above SMA(60), kept only if SMA(30) sat "
            "below SMA(60) for ≥21 days **and** hashrate itself fell ≥8% peak-to-trough during "
            "that stretch.\n"
            f"- **Tape.** BTC-USD daily close {R['btc_lo']} → {R['btc_hi']} "
            f"({R['btc_n']:,} rows). As-of 2026-06-30 (last complete month).\n"
            "- **Headline.** Forward returns at 30/90/180/365 days; Welch *t* + 20-seed × "
            "1,000-draw random-date placebo, per horizon.\n"
            "- **Execution.** Signal known at the crossover day's close; enter BTC at the "
            "**next** trading day's close (one-day lag, documented); 180-day fixed hold (a "
            "modeling choice, not folklore — Hash Ribbons has no native sell rule); 10 bps "
            "one-way × NAV × 2 legs per episode.\n"
            "- **Control.** Synthetic BTC-like world, 6 planted signal windows, tunable "
            "post-signal drift; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Signal detection — from 10 raw crossovers to 4 genuine capitulations\n\n"
            "The raw SMA(30)/SMA(60) crossover fires 10 times on this tape; most are single-"
            "digit-percent wobbles. The magnitude filter (≥21 days below, ≥8% peak-to-trough "
            "decline) keeps the four unmistakable capitulations."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = SIG\n"
            "else:\n"
            "    sig = pd.DataFrame(R['signals'])\n"
            "    sig = sig.rename(columns={'date': 'signal_date', 'days': 'capitulation_days',\n"
            "                              'decline': 'decline_pct'})\n"
            "    sig['signal_date'] = pd.to_datetime(sig['signal_date'])\n"
            "fig, ax = plt.subplots(figsize=(10.4, 4.6))\n"
            "ratio = (HR.rolling(30).mean() / HR.rolling(60).mean() - 1) * 100\n"
            "ax.plot(ratio.index, ratio.values, color=GREY, lw=1.0, label='SMA30/SMA60 - 1 (%)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.fill_between(ratio.index, 0, ratio.values, where=(ratio.values < 0),\n"
            "                color=RED, alpha=.15, label='capitulation (30d < 60d)')\n"
            "for _, row in sig.iterrows():\n"
            "    ax.axvline(row['signal_date'], color=RED, lw=1.6)\n"
            "ax.set_ylabel('ribbon spread (%)'); ax.set_title('The ribbon: 4 genuine crossings out of 10 raw ones')\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print(sig[['signal_date','capitulation_days','decline_pct']].to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the filter isn't arbitrary noise-removal — it separates two "
            "unmistakably large events (−19.7% in 2019, −43.9% in 2021) and two medium ones "
            "(both −8.1%, the COVID/halving 2020 pair) from six much smaller wobbles the "
            "original Hash Ribbons concept was never meant to flag."
        ),
        md(
            "### 4b · Forward returns — Welch *t* and the random-date placebo\n\n"
            "Per horizon: signal-conditional mean vs the unconditional distribution, Welch "
            "*t* (n≈4 — a formality, not a certificate), and the 20,000-draw placebo *p*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.forward_return_stats(BTC, SIG['signal_date'])\n"
            "    hz = list(fr.index)\n"
            "    sig_m = list(fr['signal_mean_pct']); unc_m = list(fr['uncond_mean_pct'])\n"
            "    ts = list(fr['welch_t']); ps = list(fr['placebo_p'])\n"
            "else:\n"
            "    hz = [30, 90, 180, 365]\n"
            "    sig_m = [R['fwd'][h][0] for h in hz]; unc_m = [R['fwd'][h][2] for h in hz]\n"
            "    ts = [R['fwd'][h][4] for h in hz]; ps = [R['fwd'][h][5] for h in hz]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "x = np.arange(len(hz))\n"
            "a1.bar(x, ts, color=[RED if abs(t) >= 2 else AMBER for t in ts], width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'+{h}d' for h in hz]); a1.set_ylabel('Welch t')\n"
            "a1.set_title('No horizon clears |t| = 2')\n"
            "a2.bar(x, ps, color=AMBER, width=.55)\n"
            "a2.axhline(0.05, ls='--', c=RED, lw=1, label='p = 0.05')\n"
            "for i, v in enumerate(ps): a2.annotate(f'{v:.3f}', (x[i], v), ha='center', va='bottom')\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'+{h}d' for h in hz]); a2.set_ylabel('placebo p-value')\n"
            "a2.set_title('No horizon clears p = 0.05 either'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, t, p in zip(hz, ts, ps): print(f'+{h}d: Welch t = {t:+.2f}  placebo p = {p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: the best Welch *t* is **{R['fwd'][90][4]:.2f}** (90 days) — "
            f"positive, but a third of the way to the bar. The best placebo is **p = "
            f"{R['fwd'][180][5]:.3f}** (180 days) — meaning a random 4-date draw does as well or "
            "better roughly **1 time in 8**. Both tests agree: directionally consistent, "
            "nowhere near certified."
        ),
        md(
            "### 4c · The timer backtest — a real Sharpe trade-off, not a free lunch\n\n"
            "180-day fixed hold after each signal (one-day entry lag), cash otherwise, 10 bps "
            "one-way × 2 legs per episode, compared to continuous buy-and-hold over the same "
            "[first signal → tape end] window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tb = st.timer_backtest(BTC, SIG['signal_date'], hold_days=180, cost_bps=10.0)\n"
            "else:\n"
            "    tb = dict(strat_total_pct=R['strat_total'], bh_total_pct=R['bh_total'],\n"
            "              strat_sharpe=R['strat_sharpe'], bh_sharpe=R['bh_sharpe'],\n"
            "              exposure_pct=R['exposure'], n_entries=R['n_entries'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(['timer', 'buy & hold'], [tb['strat_total_pct'], tb['bh_total_pct']],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([tb['strat_total_pct'], tb['bh_total_pct']]):\n"
            "    a1.annotate(f'{v:+.0f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('total return'); a1.set_title(f\"Exposure only {tb['exposure_pct']:.0f}% -> half the wealth\")\n"
            "a2.bar(['timer', 'buy & hold'], [tb['strat_sharpe'], tb['bh_sharpe']], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([tb['strat_sharpe'], tb['bh_sharpe']]): a2.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('Sharpe'); a2.set_title('...but a genuine Sharpe edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"{tb['n_entries']} holding episodes, exposure {tb['exposure_pct']:.1f}%: \"\n"
            "      f\"total {tb['strat_total_pct']:+.1f}% vs {tb['bh_total_pct']:+.1f}%; \"\n"
            "      f\"Sharpe {tb['strat_sharpe']:.2f} vs {tb['bh_sharpe']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: sitting out {100 - R['exposure']:.0f}% of the tape and "
            "still landing a better Sharpe ratio ({:.2f} vs {:.2f}) means the timer really is "
            "picking better-than-average windows — that's consistent with H₁'s direction. But "
            "Bitcoin's dominant feature over this sample is a near-monotone uptrend, and any "
            "rule that spends most of its time in cash gives up the compounding that comes with "
            "simply staying invested — the realized total-return gap "
            "(+{:.0f}% vs +{:.0f}%) is the price of that caution, built on just "
            "**{} distinct trades**.".format(
                R["strat_sharpe"], R["bh_sharpe"], R["strat_total"], R["bh_total"],
                R["n_entries"])
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic BTC-like daily-return world, 6 evenly-spaced planted signal windows, "
            "TUNABLE post-signal drift. The null (effect = 0) is checked over **20 seeds** — "
            "never a single stream, and deliberately at the same tiny event count as the real "
            "tape, so its false-positive rate is directly comparable to what we should expect "
            "from n≈4-6."
        ),
        code(
            "null_ts = []\n"
            "for s in range(20):\n"
            "    r, sidx = data.synthetic_world(effect=0.0, seed=663 + s)\n"
            "    null_ts.append(st.synthetic_detect(r, sidx))\n"
            "null_ts = np.asarray(null_ts)\n"
            "r, sidx = data.synthetic_world(effect=15.0, seed=663)\n"
            "planted_t = st.synthetic_detect(r, sidx)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (effect=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted +15%/yr drift')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (signal windows vs rest)')\n"
            "ax.set_title('Control: the detector is unbiased, and tiny-n noise is visible even in the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the detector recovers a real planted effect cleanly "
            f"(t = {R['syn_planted_t']:.2f}). But look at the null column: even with **no** true "
            f"effect, {R['syn_null_fire']}/20 seeds cross |t| = 2 purely by chance, because "
            "with only ~6 events per world, sampling noise alone can occasionally look "
            "\"significant.\" That's the exact caution this study applies to the real tape's "
            f"own Welch *t*'s of {min(v[4] for v in R['fwd'].values()):.2f}-"
            f"{max(v[4] for v in R['fwd'].values()):.2f} — they are unremarkable, and even if "
            "one of them had crossed 2, a tiny-n null can do that on its own roughly 10% of "
            "the time. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — every horizon (30/90/180/365 days) beats the unconditional "
            f"mean forward BTC return, but the best Welch *t* is **{R['fwd'][90][4]:.2f}** "
            f"(90d) and the best placebo is **p = {R['fwd'][180][5]:.3f}** (180d) — neither "
            f"clears the desk bar, and n = {R['n_signals']} (two overlapping) makes this "
            "fundamentally underpowered.\n"
            f"- **Tradability `FRAGILE`** — a 180-day timer built on the signal beats "
            f"buy-and-hold's Sharpe ({R['strat_sharpe']:.2f} vs {R['bh_sharpe']:.2f}) but gives "
            f"up more than half the total wealth (+{R['strat_total']:.0f}% vs "
            f"+{R['bh_total']:.0f}%), resting on only **{R['n_entries']}** distinct holding "
            "episodes and multiple hand-set thresholds (the 8% decline filter, the 21-day "
            "minimum, the 180-day hold).\n"
            "- **\"Beats random buy dates?\" `BUSTED`** — directionally ahead of a random-date "
            f"placebo at every horizon, never below *p* = "
            f"{min(v[5] for v in R['fwd'].values()):.3f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The fundamental constraint is time, not method.** No inference technique "
            "turns 4 historical episodes into a certified signal — the honest next step is "
            "simply waiting for more genuine capitulations to accumulate (or sourcing true "
            "daily hashrate data back further, if a free source ever exists).\n"
            "- **A cleaner test than ours:** run the same event-study machinery on a *basket* "
            "of proof-of-work coins with independent mining economics (Litecoin, Monero-era "
            "chains, etc.) — that would multiply the effective sample size without waiting "
            "decades, at the cost of testing a related-but-different claim.\n"
            "- **Dedup map:** [292-bitcoin-hashrate](../../292-bitcoin-hashrate/) (continuous "
            "hashrate-growth regression, NONE; a 3/6-month-MA exposure-rule version of Hash "
            "Ribbons that converges to buy-and-hold with no incremental skill — a different "
            "question from this study's discrete-event test), "
            "[221-mayer-multiple](../../221-mayer-multiple/) (price-based valuation), "
            "[323-btc-halving](../../323-btc-halving/) (the halving calendar), "
            "[210-crypto-trend](../../210-crypto-trend/) (200-day price SMA trend-following), "
            "[633-btc-vol-targeting](../../633-btc-vol-targeting/) (a vol-sizing overlay).\n\n"
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
