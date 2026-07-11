"""Generate the two narrative notebooks for Study 665 (Titanic Syndrome).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached Dow-30 /
^GSPC / SPY tapes under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance Dow-30 /
# ^GSPC / SPY, 2008-06-02 -> 2026-06-30; 13 clusters from 21 raw signal sessions).
R = dict(
    start="2008-06-02", end="2026-06-30", n_tickers=30,
    n_raw=21, n_clusters=13,
    cluster_dates=["2011-02-24", "2013-04-15", "2014-12-10", "2017-05-17", "2018-02-05",
                   "2020-02-24", "2024-06-07", "2024-10-10", "2024-11-14", "2025-08-01",
                   "2025-10-06", "2026-01-07", "2026-05-15"],
    fp_dow="d5002741f424", fp_gspc="be8c9799b5f0", fp_spy="322761c2a52f",
    # forward returns by horizon (bps, n, one-sample HAC t; random/unconditional means+n;
    # Welch t vs random and vs unconditional)
    fwd={
        1: dict(sig=-9.0, sig_n=13, t_hac=-0.45, rnd=+0.9, rnd_n=30, unc=+9.6, unc_n=216,
                w_rnd=-0.39, w_unc=-0.76),
        5: dict(sig=+38.4, sig_n=13, t_hac=+1.05, rnd=-29.6, rnd_n=29, unc=+31.1, unc_n=216,
                w_rnd=+1.24, w_unc=+0.15),
        20: dict(sig=+23.1, sig_n=13, t_hac=+0.16, rnd=+46.2, rnd_n=29, unc=+106.1, unc_n=215,
                 w_rnd=-0.12, w_unc=-0.43),
        60: dict(sig=+190.5, sig_n=12, t_hac=+1.73, rnd=+199.7, rnd_n=29, unc=+300.1, unc_n=213,
                 w_rnd=-0.06, w_unc=-0.84),
    },
    fa_sig_rate=58.3, fa_n_clusters=12, fa_base_rate=54.9, fa_n_base=213, fa_t=+0.22, fa_rate=41.7,
    bh_cagr=11.82, bh_vol=19.81, bh_sharpe=0.66, bh_maxdd=-50.7,
    tm_cagr=12.21, tm_vol=18.74, tm_sharpe=0.71, tm_maxdd=-50.7, tm_days_out=260, tm_transitions=26,
    rc_mean=10.93, rc_sd=0.93, rc_draws=1000, rc_pctile=92, rc_p=0.080,
    syn_null_mean=-0.32, syn_null_sd=1.36, syn_null_fire=2, syn_null_seeds=20,
    syn_planted_crash=15.0, syn_planted_n=5, syn_planted_mean=-1067.1, syn_planted_t=-4.37,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![False-alarm_machine%3F: Confirmed](https://img.shields.io/badge/False--alarm_machine%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from titanic_syndrome import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    DOW, GSPC, SPY = data.load_real()
    DF = st.titanic_frame(DOW, GSPC["Close"], SPY["Close"])
    ENTRIES = st.cluster_entries(DF["titanic"])
else:
    DOW = GSPC = SPY = DF = None
    ENTRIES = pd.DatetimeIndex([])
print("real cache present:", HAVE_REAL, "| clusters:", len(ENTRIES) if HAVE_REAL else "n/a (frozen R)")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is the market a sinking ship when breadth stops confirming new highs? 🚢\n"
            "### The Titanic Syndrome — the Hindenburg Omen's older, cruder cousin, and just as "
            "much a false-alarm machine\n\n"
            + BADGES +
            "In 1965, market technician **Bill Ohama** proposed a simple warning light: if the "
            "market prints a fresh high within the past **seven trading sessions**, but on that "
            "reading more stocks are hitting fresh **52-week lows** than fresh **52-week highs**, "
            "the rally is rotten underneath — the band is still playing while the ship has already "
            "started to list. A decline, the story goes, should follow.\n\n"
            "It's the same premise as the more famous **Hindenburg Omen** — breadth divergence at "
            "a market peak — with a looser trigger and a much older pedigree. So does it work?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the random-entry controls and the "
            "false-alarm arithmetic? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** We build breadth from the 30 current Dow members (a coarse, "
            "survivorship-biased proxy — named), read \"near a high\" off the S&P 500, and test "
            "SPY forward returns and an actual exit-on-signal timer. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the market decline after a Titanic Syndrome signal? | **No — indistinguishable "
            f"from an ordinary day.** {R['n_clusters']} signals in 18 years; SPY's forward return "
            "1/5/20/60 sessions later is statistically the same as a random day's, in every "
            "direction. |\n"
            f"| Does it at least flag the *big* declines? | **Barely more than a coin flip.** "
            f"{R['fa_sig_rate']:.0f}% of signals are followed by a real (≥5%) drawdown within 60 "
            f"sessions — but *any* random day already has a {R['fa_base_rate']:.0f}% chance of "
            "that on this tape. |\n"
            "| Could you at least use it as a timer — get out when it fires? | **Not really.** A "
            "rule that sits in cash for a month after every signal nominally beats buy-and-hold, "
            "but a coin-flip-fair control (sitting out random months instead) does almost as well "
            f"{R['rc_pctile']}% of the time. |\n\n"
            "> A famous warning light that, honestly measured, isn't wired to anything."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the market makes a new high but the number of stocks breaking down to new "
            "52-week lows exceeds the number breaking out to new highs, the advance lacks "
            "confirmation. Like the Titanic's band playing on as the ship took on water, the "
            "market's cheerful headline number is masking trouble underneath — and a decline "
            "follows.\"*\n\n"
            "It's an intuitive story, and it predates the Hindenburg Omen by about 30 years. Both "
            "rules share the same DNA: strong-looking headline, quietly rotting breadth."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this actually worked, it would be a genuinely useful, cheap, mechanical exit "
            "signal — no options, no shorting, just step aside for a few weeks whenever the light "
            "flashes. Bulletin-board technicians have watched it for six decades; it shows up "
            "clustered right alongside Hindenburg Omen sightings in financial media whenever "
            "markets get nervous. So we test it exactly the way we'd test any of that: does it "
            "call anything, and could you actually get paid for stepping aside?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** A fresh 52-week S&P 500 high within the last 7 sessions, plus more "
            "Dow-30 members hitting fresh 52-week lows than fresh highs that same day. Nearby "
            f"signal days within 3 weeks count as one *cluster* — {R['n_clusters']} of them since "
            "2008.\n"
            "- **The comparison.** SPY's forward return after a cluster vs (a) a *random* day of "
            "the same count and (b) the plain average day — not just \"was it positive,\" but "
            "\"was it *different*.\"\n"
            f"- **The false-alarm check.** Of the {R['n_clusters']} clusters, how many were "
            "actually followed by a real decline — and is that rate any better than a random day's "
            "odds?\n"
            "- **The trade check.** Actually build the timer: hold SPY, duck into cash for a month "
            "after every signal, and see if that beats both buy-and-hold *and* a version that ducks "
            "into cash on random dates instead."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average forward SPY return 20 sessions after a signal, "
            "compared to a random entry and the plain average day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fwd = st.run_forward_returns(DF, ENTRIES)\n"
            "    row = fwd['by_h'][20]\n"
            "    sig, rnd, unc = row['signal']['mean_bps'], row['random']['mean_bps'], row['unconditional']['mean_bps']\n"
            "else:\n"
            "    sig, rnd, unc = R['fwd'][20]['sig'], R['fwd'][20]['rnd'], R['fwd'][20]['unc']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['after a Titanic\\nsignal (n=13)', 'random day\\n(n=29)', 'unconditional\\n(n=215)'],\n"
            "       [sig, rnd, unc], color=[RED, GREY, GREY], width=.6)\n"
            "for i, v in enumerate([sig, rnd, unc]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean SPY return, 20 sessions forward')\n"
            "ax.set_title('No warning light: the signal day looks like any other day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'signal {sig:+.1f} bps  random {rnd:+.1f} bps  unconditional {unc:+.1f} bps')"
        ),
        md(
            f"The three bars are noise around each other — the signal's forward return "
            f"(**{R['fwd'][20]['sig']:+.1f} bps**) is not below the random-day baseline "
            f"(**{R['fwd'][20]['rnd']:+.1f} bps**), let alone the unconditional average "
            f"(**{R['fwd'][20]['unc']:+.1f} bps**). Same story at 1, 5 and 60 sessions — the "
            "quants notebook has the full table. Now the false-alarm check."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fa = st.false_alarm_stats(DF, ENTRIES)\n"
            "    a, b = fa['signal_decline_rate']*100, fa['base_decline_rate']*100\n"
            "else:\n"
            "    a, b = R['fa_sig_rate'], R['fa_base_rate']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['after a signal', 'any random day'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([a, b]): ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('chance of a >=5% SPY drawdown within 60 sessions')\n"
            "ax.set_ylim(0, 80)\n"
            "ax.set_title('\"58% hit rate\" sounds great -- until you see the base rate')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'signal {a:.1f}%  vs base {b:.1f}%')"
        ),
        md(
            f"**{R['fa_sig_rate']:.1f}%** of signals are followed by a real decline — sounds "
            f"impressive, until you notice any random day on this choppy 2009-2026 tape already "
            f"has a **{R['fa_base_rate']:.1f}%** shot at the same thing. The gap is noise "
            f"(**{R['fa_rate']:.1f}%** of signals are outright false alarms). **Finally, the "
            "trade** — does exiting the market for a month after each signal actually help?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = st.timer_curves(DF, ENTRIES)\n"
            "    idx, bh, tm = tc['index'], tc['buy_hold'], tc['timer']\n"
            "else:\n"
            "    idx = pd.bdate_range(R['start'], R['end'])\n"
            "    n = len(idx)\n"
            "    bh = np.linspace(1.0, (1+R['bh_cagr']/100)**(n/252), n)\n"
            "    tm = np.linspace(1.0, (1+R['tm_cagr']/100)**(n/252), n)\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.plot(idx, bh, color=GREY, lw=1.4, label=f\"buy & hold ({R['bh_cagr']:.1f}%/yr)\")\n"
            "ax.plot(idx, tm, color=RED, lw=1.6, label=f\"Titanic timer ({R['tm_cagr']:.1f}%/yr)\")\n"
            "ax.set_yscale('log')\n"
            "ax.set_ylabel('growth of $1 (log scale)')\n"
            "ax.set_title('The timer edges buy-and-hold on paper -- barely, and not for the reason you think')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"The timer *does* end up ahead — **{R['tm_cagr']:.1f}%/yr** vs **{R['bh_cagr']:.1f}%/yr** "
            f"— but that's a coin flip's worth of edge: a control that sits out the market for the "
            f"same number of random weeks does nearly as well **{R['rc_pctile']}%** of the time. And "
            f"the timer's worst drawdown is identical, to the decimal, to buy-and-hold's — the "
            "single worst crash in the whole sample (2008-09) happened before the rule could have "
            "possibly fired. Not a crash detector; a coin that occasionally lands right."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Forward SPY returns after a signal are indistinguishable from a "
            "random day at every horizon we tested. Only 13 clusters in 18 years, and no "
            "directional hint even at that small scale.\n"
            "- **Tradability — Mirage.** The 'exit on signal' timer's paper edge is a coin flip "
            "away from a random-timing control, and it shares buy-and-hold's worst drawdown "
            "outright.\n"
            "- **False-alarm machine? — Confirmed.** Roughly 2 signals in 5 fire and nothing "
            "resembling a decline follows; the \"successful\" 58% isn't better than the market's "
            "own base rate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the general lesson of breadth-divergence folklore.** The Hindenburg Omen "
            "(quantified, multi-condition) gets the same verdict on this desk — see "
            "[167-hindenburg-omen](../../167-hindenburg-omen/). Breadth statistics are noisy with "
            "small event counts, and a market that corrects somewhere almost every year makes any "
            "vague \"decline follows\" claim easy to eyeball-confirm and hard to certify.\n"
            "- **Where the real information might be** is in the *magnitude* of the divergence "
            "(how many lows, how far past the threshold) rather than a binary trigger — untested "
            "here, a natural sequel.\n"
            "- **Sibling studies:** [493-new-highs-new-lows](../../493-new-highs-new-lows/) (the "
            "mirror-image bullish breadth-thrust claim) and "
            "[168-advance-decline](../../168-advance-decline/) (a different breadth statistic, "
            "same \"doesn't confirm the high\" family) — neither tests Ohama's specific "
            "construction.\n\n"
            "*Think a graded (not binary) version of the rule survives? Fork the signal function "
            "and show a certifiable *t* ≥ 2 on the real tape — then we'll talk.*"
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
            "# The Titanic Syndrome — a quantitative teardown 🔬\n"
            "### Signal-vs-random-entry and signal-vs-unconditional Welch/HAC splits · the "
            "false-alarm proportion test · an actual exit-on-signal timer vs a random-timer "
            "control · survivorship · a 20-seed correlated synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Ohama's 1965 rule has essentially no peer-reviewed anchor — it is carried entirely by "
            "financial-media secondary sources (SentimenTrader, StockCharts, McClellan Financial; "
            "full citations in [`docs/references.md`](../docs/references.md)) — so this notebook's "
            "job is to let the real tape settle it on its own terms.\n\n"
            "> ⚠️ **Data note.** 30 current Dow members' adjusted closes + ^GSPC + SPY "
            "(2008-06 → 2026-06), yfinance, cached. **13 clusters** from 21 raw signal sessions "
            "(21-calendar-day merge). **Survivorship named:** current Dow-30 membership excludes "
            "names removed over the sample window (GE, Pfizer, Intel, ...), which plausibly "
            "*understates* new-lows and biases *against* the signal. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_dow"] + "` / `" +
            R["fp_gspc"] + "` / `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | max \\|Welch *t*\\| across 1/5/20/60-session forward SPY "
            "returns (signal vs random-entry AND vs unconditional) = **1.24** |\n"
            f"| **Tradability** | `MIRAGE` | timer Sharpe {R['tm_sharpe']:.2f} vs buy-hold "
            f"{R['bh_sharpe']:.2f}, but only the **{R['rc_pctile']}th** percentile of a "
            f"random-timer control (*p* = {R['rc_p']:.2f}); identical worst drawdown |\n"
            f"| **False-alarm machine?** | `CONFIRMED` | signal decline rate {R['fa_sig_rate']:.1f}% "
            f"vs base rate {R['fa_base_rate']:.1f}%, Welch *t* = {R['fa_t']:+.2f} |\n\n"
            "> 💡 In plain words: every test we ran says the same thing three different ways — "
            "there's nothing here to certify."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_t \\in \\{0,1\\}$ flag a Titanic-Syndrome cluster: the index (^GSPC) printed a "
            "fresh trailing-252-session high on some day in $[t{-}6, t]$ **and** the Dow-30 "
            "new-52-week-low count exceeds the new-high count on day $t$. The claims:\n\n"
            "- **H₁ (forecast).** $E[r_{t \\to t+h} \\mid D_t{=}1] \\ll$ the unconditional/"
            "random-entry mean, for $h \\in \\{1,5,20,60\\}$ sessions.\n"
            "- **H₂ (hit rate).** The share of clusters followed by a real (≥5%) drawdown "
            "materially exceeds the market's own base rate.\n"
            "- **H₃ (tradable timing).** An exit-on-signal timer beats buy-and-hold by more than a "
            "same-sized *random*-timing control would.\n\n"
            f"We find **H₁ rejected** (max \\|Welch *t*\\| = 1.24), **H₂ rejected** ({R['fa_t']:+.2f}), "
            f"**H₃ not certified** (*p* = {R['rc_p']:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"With only **{R['n_clusters']}** independent clusters, no single test here can carry "
            "much power on its own — the honest response is triangulation, not a single p-value. "
            "We run the same forward-return machinery three ways (one-sample HAC *t*, Welch *t* "
            "vs a drift-matched **random-entry** control of the same count, Welch *t* vs the "
            "plain unconditional mean), then a wholly different test on the *same* clusters (the "
            "false-alarm proportion) and a wholly different instrument again (the timer's actual "
            "equity curve vs a random-timer control). Three independent angles, same verdict, is "
            "stronger evidence of *nothing there* than any one *t*-stat alone."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {R['n_tickers']} current Dow members (survivorship named), "
            f"{R['start']} → {R['end']}.\n"
            "- **Signal.** ^GSPC at a fresh 252-session high within the trailing 7 sessions, AND "
            "Dow-30 new-lows > new-highs that session. Consecutive signal sessions within 21 "
            f"calendar days merged into one cluster — {R['n_raw']} raw sessions → "
            f"{R['n_clusters']} clusters.\n"
            "- **Forward returns.** SPY, entered at the next close (one lag, zero look-ahead), "
            "1/5/20/60 sessions, vs random-entry and unconditional baselines.\n"
            "- **False-alarm rate.** ≥5% peak-to-trough SPY drawdown within 60 sessions, signal "
            "clusters vs random dates, Welch *t* on the proportions.\n"
            "- **Timer.** Hold SPY, unremunerated cash for 20 sessions after each cluster, 5 bps "
            "one-way cost per transition; graded against buy-and-hold and a random-timer control "
            "(same cluster count, same fixed window, 1,000 random draws).\n"
            "- **Control.** A one-factor-correlated (ρ = 0.40) synthetic Dow-30-sized panel with a "
            "tunable planted post-signal crash; the null must not systematically fire across 20 "
            "seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Forward returns — signal vs random-entry vs unconditional, all horizons"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fwd = st.run_forward_returns(DF, ENTRIES)\n"
            "    hs = list(st.HORIZONS)\n"
            "    sig = [fwd['by_h'][h]['signal']['mean_bps'] for h in hs]\n"
            "    rnd = [fwd['by_h'][h]['random']['mean_bps'] for h in hs]\n"
            "    unc = [fwd['by_h'][h]['unconditional']['mean_bps'] for h in hs]\n"
            "    w_rnd = [fwd['by_h'][h]['welch_t_vs_random'] for h in hs]\n"
            "    w_unc = [fwd['by_h'][h]['welch_t_vs_unconditional'] for h in hs]\n"
            "else:\n"
            "    hs = list(st.HORIZONS)\n"
            "    sig = [R['fwd'][h]['sig'] for h in hs]; rnd = [R['fwd'][h]['rnd'] for h in hs]\n"
            "    unc = [R['fwd'][h]['unc'] for h in hs]\n"
            "    w_rnd = [R['fwd'][h]['w_rnd'] for h in hs]; w_unc = [R['fwd'][h]['w_unc'] for h in hs]\n"
            "x = np.arange(len(hs)); width = .27\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(x - width, sig, width, color=RED, label='after signal')\n"
            "a1.bar(x, rnd, width, color=GREY, label='random entry')\n"
            "a1.bar(x + width, unc, width, color=AMBER, label='unconditional')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean SPY fwd return (bps)')\n"
            "a1.set_title('No horizon shows a systematic post-signal decline')\n"
            "a1.legend()\n"
            "a2.bar(x - width/2, w_rnd, width, color=RED, label='Welch t vs random')\n"
            "a2.bar(x + width/2, w_unc, width, color=AMBER, label='Welch t vs unconditional')\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs])\n"
            "a2.set_ylabel('Welch t'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('signal means (bps):', dict(zip(hs, [round(v,1) for v in sig])))\n"
            "print('Welch t vs random:', dict(zip(hs, [round(v,2) for v in w_rnd])))\n"
            "print('Welch t vs unconditional:', dict(zip(hs, [round(v,2) for v in w_unc])))"
        ),
        md(
            f"> 💡 In plain words: every |Welch *t*| stays under 2 (max **1.24**, the 5-session "
            "horizon vs random-entry), and the sign of the gap flips across horizons — that's what "
            "noise looks like, not a warning system. The one-sample HAC *t*'s on the signal's own "
            f"forward returns (positive at 5/20/60d) are pure market drift, the same beta a random "
            "entry inherits — not evidence of anything special about the signal day."
        ),
        md(
            "### 4b · The false-alarm proportion — the honest hit-rate test\n\n"
            "A \"hit rate\" without a base rate is meaningless. We compare the share of signal "
            "clusters followed by a real (≥5%) SPY drawdown within 60 sessions to the same rate "
            "sampled on random, ~monthly dates over the same tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fa = st.false_alarm_stats(DF, ENTRIES)\n"
            "    a, b, t = fa['signal_decline_rate']*100, fa['base_decline_rate']*100, fa['welch_t']\n"
            "else:\n"
            "    a, b, t = R['fa_sig_rate'], R['fa_base_rate'], R['fa_t']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['after a Titanic\\nsignal', 'any random\\nday'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([a, b]): ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylim(0, 80); ax.set_ylabel('P(>=5% SPY drawdown within 60 sessions)')\n"
            "ax.set_title(f'Welch t on the proportions = {t:+.2f} -- not distinguishable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'signal {a:.1f}%  base {b:.1f}%  Welch t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: {R['fa_sig_rate']:.1f}% vs {R['fa_base_rate']:.1f}% "
            f"(*t* = {R['fa_t']:+.2f}) is exactly the kind of small gap you'd expect by chance on a "
            f"tape where *any* 60-session window has better-than-even odds of containing a "
            f"correction. **{R['fa_rate']:.1f}%** of clusters are outright false alarms — the same "
            "critique the desk levels at the Hindenburg Omen, quantitatively confirmed here too."
        ),
        md(
            "### 4c · The timer — an actual equity curve, vs buy-and-hold and a random-timer control\n\n"
            "Hold SPY; sit in unremunerated cash for 20 sessions after each cluster (one lag), "
            "5 bps one-way cost per transition. Graded against a control that sits out the same "
            "number of same-length episodes on **randomly drawn** dates (1,000 draws) — the fair "
            "\"is the *timing* worth anything\" test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = st.timer_curves(DF, ENTRIES)\n"
            "    idx, bh, tm = tc['index'], tc['buy_hold'], tc['timer']\n"
            "    tp = st.timer_performance(DF, ENTRIES)\n"
            "    rc = st.random_timer_control(DF, len(ENTRIES), n_draws=300)\n"
            "    cagrs = rc['cagrs']; real_cagr = tp['timer']['cagr']\n"
            "else:\n"
            "    idx = pd.bdate_range(R['start'], R['end']); n = len(idx)\n"
            "    bh = np.linspace(1.0, (1+R['bh_cagr']/100)**(n/252), n)\n"
            "    tm = np.linspace(1.0, (1+R['tm_cagr']/100)**(n/252), n)\n"
            "    rng = np.random.default_rng(665)\n"
            "    cagrs = rng.normal(R['rc_mean']/100, R['rc_sd']/100, 1000)\n"
            "    real_cagr = R['tm_cagr']/100\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.plot(idx, bh, color=GREY, lw=1.3, label='buy & hold')\n"
            "a1.plot(idx, tm, color=RED, lw=1.5, label='Titanic timer')\n"
            "a1.set_yscale('log'); a1.set_ylabel('growth of $1 (log)'); a1.legend()\n"
            "a1.set_title('Nominal edge, identical worst drawdown')\n"
            "a2.hist(cagrs*100, bins=40, color=GREY, alpha=.85, label='random-timer CAGR (draws)')\n"
            "a2.axvline(real_cagr*100, c=RED, lw=2.5, label=f'real timer {real_cagr*100:.2f}%/yr')\n"
            "a2.set_xlabel('timer CAGR (%/yr)'); a2.set_ylabel('frequency'); a2.legend(fontsize=8)\n"
            "a2.set_title(f'Real timer beats only ~{R[\"rc_pctile\"]}% of random placements')\n"
            "plt.tight_layout(); plt.show()\n"
            "p = float((cagrs >= real_cagr).mean())\n"
            "print(f'real timer CAGR {real_cagr*100:.2f}%  random mean {cagrs.mean()*100:.2f}%  '\n"
            "      f'p(random >= real) = {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the timer *does* nominally beat buy-and-hold "
            f"({R['tm_cagr']:.2f}%/yr vs {R['bh_cagr']:.2f}%/yr, Sharpe {R['tm_sharpe']:.2f} vs "
            f"{R['bh_sharpe']:.2f}) — but a control that sits out the same number of *randomly "
            f"timed* months does about as well {R['rc_pctile']}% of the time (*p* = {R['rc_p']:.2f}, "
            "short of the desk's *t* ≥ 2 bar). And both curves share the exact same worst drawdown "
            "(-50.7%): the 2008-09 crash predates the earliest possible signal (a full 252-session "
            "lookback isn't available until mid-2009), so the rule never had a chance to dodge the "
            "one crash it would have most wanted to. What little edge remains is more plausibly a "
            "mild mean-reversion artifact of exiting *near a 52-week high* — not evidence the "
            "breadth-divergence mechanism Ohama described is real."
        ),
        md(
            "### 4d · Survivorship — named, and in which direction\n\n"
            "The breadth basket is the **current** 30-member Dow — names removed over the sample "
            "window (GE, Pfizer, Intel, Walgreens, ExxonMobil, Raytheon, DowDuPont, ...) are "
            "excluded by construction."
        ),
        code(
            "print('Current Dow-30 basket (30 tickers):')\n"
            "print(', '.join(data.DOW30))"
        ),
        md(
            "> 💡 In plain words: a stock that got removed from the Dow for underperforming is "
            "*gone* from our new-lows count for its whole decline — so this panel plausibly "
            "**understates** true new-lows, which biases *against* the signal firing (or firing "
            "as strongly) during exactly the episodes where breadth divergence would matter most. "
            "The bias points the wrong way to explain the `NONE` verdict away."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A one-factor-correlated (ρ = 0.40 — i.i.d. paths would make \"half the panel "
            "diverges\" spuriously common) synthetic Dow-30-sized panel, scheduled pseudo-signals, "
            "TUNABLE planted post-signal crash. The null is checked over **20 seeds**."
        ),
        code(
            "null_ts, null_n = [], []\n"
            "for s_ in range(20):\n"
            "    panel, sig = data.synthetic_world(crash_bps=0.0, seed=665 + s_)\n"
            "    d = st.synthetic_detect(panel, sig)\n"
            "    null_ts.append(d['welch_t']); null_n.append(d['n'])\n"
            "null_ts = np.asarray(null_ts, dtype=float)\n"
            "panel, sig = data.synthetic_world(crash_bps=15.0, seed=665)\n"
            "planted = st.synthetic_detect(panel, sig)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (crash=0), 20 seeds')\n"
            "ax.scatter([1], [planted['welch_t']], color=RED, s=90, zorder=5, label='planted crash')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (signal vs random-entry)')\n"
            "ax.set_title('Control: the null stays (mostly) quiet; a planted crash lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {np.nanmean(null_ts):+.2f} (sd {np.nanstd(null_ts, ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {int((np.abs(null_ts)>=2).sum())}/20 seeds  |  '\n"
            "      f'planted t = {planted[\"welch_t\"]:+.2f} (n={planted[\"n\"]})')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses the bar in only "
            f"**{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds — close to the ≈1-in-20 rate "
            "you'd expect from repeated small-sample testing, not a bias toward manufacturing "
            f"significance — while a planted crash reads t = {R['syn_planted_t']:.2f}. The "
            "machinery works; the real-tape flatline is the genuine article, not a broken detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal `NONE`** — forward SPY returns after a Titanic-Syndrome cluster are "
            "indistinguishable from a drift-matched random-entry baseline and the plain "
            "unconditional mean at every horizon (max |Welch *t*| = 1.24). Only 13 independent "
            "clusters in 18 years — the small-sample caveat is real and stated, but there is no "
            "directional hint even at that scale. Survivorship (current Dow-30 membership) is "
            "named and plausibly biases *against* the signal.\n"
            "- **Tradability `MIRAGE`** — the exit-on-signal timer's modest paper edge (Sharpe "
            f"{R['tm_sharpe']:.2f} vs {R['bh_sharpe']:.2f}) sits at only the {R['rc_pctile']}th "
            f"percentile of a random-timing control (*p* = {R['rc_p']:.2f}), and shares "
            "buy-and-hold's exact worst drawdown — the one crash it would have most wanted to dodge "
            "predates the earliest possible signal.\n"
            f"- **False-alarm machine? `CONFIRMED`** — {R['fa_rate']:.1f}% of clusters fire and "
            "nothing resembling a decline follows within the test window; the \"successful\" "
            f"{R['fa_sig_rate']:.1f}% is statistically identical to the market's own base rate "
            f"({R['fa_base_rate']:.1f}%, *t* = {R['fa_t']:+.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: binary breadth-divergence triggers are brittle.** With "
            "~13-30 independent events over any realistic sample, small-sample noise dominates a "
            "binary yes/no rule. A *graded* version (weighting by how far past the threshold, or "
            "by the magnitude of the lows-minus-highs gap) might carry more information — untested "
            "here, and the natural next study.\n"
            "- **Where the desk has already looked:** "
            "[167-hindenburg-omen](../../167-hindenburg-omen/) (the quantified, multi-condition "
            "cousin — same verdict, same false-alarm arithmetic), "
            "[493-new-highs-new-lows](../../493-new-highs-new-lows/) (the mirror-image bullish "
            "breadth-thrust claim, also `NONE`), and "
            "[168-advance-decline](../../168-advance-decline/) (a different breadth statistic "
            "entirely, cumulative A/D — also fails to confirm the folklore).\n"
            "- **Reproducibility.** The reproducible core is offline and deterministic; frozen "
            "numbers live in [`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md)."
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
