"""Generate the two narrative notebooks for Study 709 (World-Series-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^GSPC tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^GSPC daily
# close 1949-11-01 -> 2026-06-30; 76 hardcoded World Series champions 1950-2025, 75
# actually played, 74 scoreable against a complete next calendar year).
R = dict(
    start=1950, end=2025, n_played=75, n_events=74,
    n_al=39, n_nl=35, n_ny=19, n_nonny=55,
    uncond_up_pct=73.0,
    # NL-omen (league variant, the ported NFC mnemonic)
    nl_mean_bull=11.21, nl_mean_bear=7.16, nl_contrast=4.05,
    nl_welch_t=1.07, nl_perm_p=0.2955,
    nl_hit=52.7, nl_hit_lo=41.5, nl_hit_hi=63.7, nl_binom_p=0.3675, nl_coin_p=0.7275,
    # NY-omen (city variant)
    ny_mean_bull=8.02, ny_mean_bear=9.43, ny_contrast=-1.41,
    ny_welch_t=-0.33, ny_perm_p=0.7512,
    ny_hit=39.2, ny_hit_lo=28.9, ny_hit_hi=50.6, ny_binom_p=0.5896, ny_coin_p=0.0805,
    # Could-you-trade-it timing strategy
    bah_ann=7.75,
    nl_strat_ann=4.68, nl_strat_adv=-3.06, nl_n_held=35,
    ny_strat_ann=1.73, ny_strat_adv=-6.01, ny_n_held=19,
    # Synthetic control
    syn_null_mean=0.09, syn_null_sd=0.88, syn_null_fire=0,
    syn_boost=10.0, syn_planted_t=3.01, syn_bull_mean=17.68, syn_bear_mean=5.78,
    fp_gspc="50410c20273a",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
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

from world_series_effect import data, strategy as st

WS = data.ws_table()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    GSPC = data.load_real()
    ANN = data.annual_returns(GSPC)
    EV = st.build_events(WS, ANN)
else:
    GSPC = ANN = EV = None
print("real cache present:", HAVE_REAL, "| played World Series:", len(WS),
      "| scoreable events:", (0 if EV is None else len(EV)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the World Series pick next year's stock market? ⚾📉\n"
            "### A baseball cousin of the Super Bowl Indicator — and it fails the same way\n\n"
            + BADGES +
            "Every October, someone in financial media half-jokingly asks whether the World "
            "Series winner tells you anything about next year's stock market — the same way the "
            "**Super Bowl Indicator** claims the NFC/AFC does. There are even two flavors: does "
            "the *league* (American vs National) matter, or does it just come down to whether "
            "**a New York team wins** — Wall Street's hometown club, riding high?\n\n"
            "We test both. Neither survives contact with the correct baseline.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the permutation test and the "
            "coin-flip math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 76 World Series champions hardcoded from MLB's own postseason "
            "record, 1950→2025 (1994's players'-strike gap named, not papered over). Every chart "
            "is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the champion's league predict next year? | **No.** NL-preceded years "
            f"averaged **+{R['nl_mean_bull']:.1f}%**, AL-preceded years **+{R['nl_mean_bear']:.1f}%** "
            f"— a gap that looks real until you check the statistics: *t* = **{R['nl_welch_t']:.2f}** "
            f"(the bar is 2), and a random relabeling produces a gap this big **{R['nl_perm_p']*100:.0f}% "
            "of the time**. |\n"
            "| Does \"a New York team wins\" predict a good year? | **No — and it doesn't even "
            f"point the right way.** NY-preceded years averaged **+{R['ny_mean_bull']:.1f}%** vs "
            f"**+{R['ny_mean_bear']:.1f}%** for everyone else — the hometown story runs backwards "
            "in this sample. |\n"
            f"| Does either omen beat the market's own habit of going up? | **No.** The S&P rose "
            f"in **{R['uncond_up_pct']:.0f}%** of all years in the sample regardless of who won "
            f"the World Series — and the league-omen's hit rate ({R['nl_hit']:.0f}%) and city-omen's "
            f"hit rate ({R['ny_hit']:.0f}%) are both *below* that baseline. |\n"
            "| Can you trade it? | **No.** Sitting in cash whenever the omen says \"bearish\" costs "
            f"you **{abs(R['nl_strat_adv']):.1f} to {abs(R['ny_strat_adv']):.1f} percentage points "
            "a year** versus just staying invested — before a single dollar of trading cost. |\n\n"
            "> There's a champion every year. The market goes up most years anyway. Everything "
            "else is a mascot."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a National League team wins the World Series, stocks do well the "
            "following year; when an American League team wins, they don't — the same "
            "'omen' logic as the Super Bowl Indicator (NFC good, AFC bad). And really, isn't "
            "it just about New York? When the Yankees (or Mets, or the old Brooklyn Dodgers) "
            "win it all, doesn't Wall Street's own backyard team winning mean good times are "
            "coming?\"*\n\n"
            "Unlike this desk's [FOMC vol-crush study](../../637-fomc-vol-crush/), which has a "
            "one-sentence causal mechanism (event-premium expiry), **nobody has ever proposed a "
            "reason** the World Series' league or host city should move the S&P. That's not "
            "disqualifying on its own — folklore doesn't need a mechanism to be tested — but it's "
            "worth naming up front: if this turns out to be a mirage, there's no theory to salvage."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a free, public, zero-look-ahead signal: the World Series "
            "ends by early November, weeks before you'd need to act on a January 1 entry. No "
            "insider information, no complex model — just watch the box score. That's exactly "
            "why it's worth testing seriously rather than dismissing on priors: cheap, testable "
            "claims deserve the same rigor as expensive ones. And if it's *not* real, it's a "
            "clean teaching example of how a coincidence gets dressed up as an omen."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_played']}** World Series actually played from 1950 "
            "to 2025 (1994 was cancelled by the players' strike — named, not imputed), hardcoded "
            "from MLB's own postseason record.\n"
            f"- **The comparison.** Each champion's league (AL/NL) and hometown (New York or not) "
            "against the S&P 500's return the *following* calendar year — "
            f"**{R['n_events']}** scoreable pairs in total.\n"
            "- **The correct baseline.** The S&P doesn't split its years 50/50 — it goes up about "
            f"**{R['uncond_up_pct']:.0f}%** of the time regardless. Any \"predicts bullish\" signal "
            "has to beat *that*, not a coin flip. Testing against 50% is the single most common "
            "mistake in these write-ups.\n"
            "- **The luck check.** Shuffle which seasons count as \"National League\" 20,000 times "
            "— how often does a random relabeling produce a gap this large?\n"
            "- **The trade check.** Hold the S&P only in years the omen calls bullish, sit in cash "
            "otherwise — does that beat just staying invested?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the league omen.** Average next-year S&P return after an NL vs AL champion."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull_nl = (EV['league'] == 'NL').to_numpy()\n"
            "    s = st.omen_stats(EV, bull_nl)\n"
            "    mb, mr = s['mean_bull_pct'], s['mean_bear_pct']\n"
            "else:\n"
            "    mb, mr = R['nl_mean_bull'], R['nl_mean_bear']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['NL champion\\n(next year)','AL champion\\n(next year)'], [mb, mr],\n"
            "       color=[AMBER, GREY], width=.6)\n"
            "for i,v in enumerate([mb, mr]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(R['uncond_up_pct']*0 , c='k', lw=0)  # keep axis tidy\n"
            "ax.set_ylabel('mean S&P return the following calendar year (%)')\n"
            "ax.set_title('A gap that looks real -- until you check the statistics')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'NL-preceded {mb:+.2f}%   AL-preceded {mr:+.2f}%')"
        ),
        md(
            f"A **{R['nl_contrast']:+.1f} percentage-point** gap, in the claimed direction. But the "
            f"Welch *t* is only **{R['nl_welch_t']:.2f}** — the desk's bar is 2 — and a "
            f"random-relabeling test says a gap this size shows up **{R['nl_perm_p']*100:.0f}% of "
            "the time** by pure chance. That's not a signal; that's 35 vs 39 draws from the same "
            "noisy bucket.\n\n"
            "**Now the base-rate trap** — the number that actually kills the omen."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hit, base = s['hit_rate_pct'], s['uncond_up_pct']\n"
            "else:\n"
            "    hit, base = R['nl_hit'], R['uncond_up_pct']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['omen hit rate\\n(NL-> bullish)','S&P unconditional\\nup-rate'], [hit, base],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([hit, base]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 100)\n"
            "ax.set_ylabel('%')\n"
            "ax.set_title('The omen calls the market WORSE than just assuming every year is an up year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'omen hit rate {hit:.1f}%   unconditional up-rate {base:.1f}%')"
        ),
        md(
            f"The S&P is up in about **{R['uncond_up_pct']:.0f}%** of all years regardless of "
            f"baseball. The league omen's hit rate is only **{R['nl_hit']:.1f}%** — meaning "
            "\"trust the omen\" calls the market's direction *less* accurately than simply "
            "assuming every year will be an up year, which it usually is anyway. This is the "
            "exact trap the Super Bowl Indicator falls into (see "
            "[study 158](../../158-super-bowl/)) — testing against a 50% coin instead of the "
            "market's real bias makes any \"predict up\" signal look prescient. It isn't.\n\n"
            "**Now the city variant** — does it at least point the right way?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull_ny = EV['is_ny'].to_numpy()\n"
            "    s2 = st.omen_stats(EV, bull_ny)\n"
            "    mb2, mr2 = s2['mean_bull_pct'], s2['mean_bear_pct']\n"
            "else:\n"
            "    mb2, mr2 = R['ny_mean_bull'], R['ny_mean_bear']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['New York champion\\n(next year)','everyone else\\n(next year)'], [mb2, mr2],\n"
            "       color=[GREY, GREY], width=.6)\n"
            "for i,v in enumerate([mb2, mr2]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean S&P return the following calendar year (%)')\n"
            "ax.set_title(\"'Wall Street's hometown team' -- runs BACKWARDS in this sample\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'NY-preceded {mb2:+.2f}%   everyone else {mr2:+.2f}%')"
        ),
        md(
            f"No — a New York title is followed by **+{R['ny_mean_bull']:.1f}%** on average, "
            f"*worse* than the **+{R['ny_mean_bear']:.1f}%** everyone else gets. The pinstriped "
            "\"hometown of Wall Street\" story is exactly backwards on this tape, even before "
            "checking significance (it isn't significant either — the quants notebook has the "
            "numbers). Sixteen Yankee/Mets/Giants/Dodgers titles clustered in some strong-market "
            "decades and some weak ones; the eye sees a pattern where the arithmetic finds noise.\n\n"
            "**Finally, could you actually trade either omen?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t1 = st.timing_strategy(EV, bull_nl); t2 = st.timing_strategy(EV, bull_ny)\n"
            "    bah, nl_a, ny_a = t1['bah_ann_pct'], t1['strat_ann_pct'], t2['strat_ann_pct']\n"
            "else:\n"
            "    bah, nl_a, ny_a = R['bah_ann'], R['nl_strat_ann'], R['ny_strat_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['buy & hold','NL-omen\\ntiming','NY-omen\\ntiming'], [bah, nl_a, ny_a],\n"
            "       color=[GREEN, RED, RED], width=.6)\n"
            "for i,v in enumerate([bah, nl_a, ny_a]): ax.annotate(f'{v:+.2f}%/yr',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualized return')\n"
            "ax.set_title('Timing on either omen loses to just staying invested')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy-and-hold {bah:+.2f}%/yr   NL-timing {nl_a:+.2f}%/yr   NY-timing {ny_a:+.2f}%/yr')"
        ),
        md(
            "Sitting in cash whenever an omen says \"bearish\" means missing chunks of a market "
            "that goes up most years — and neither omen has a real signal underneath to pay that "
            "back. Both timing strategies lose to simply staying invested, before a single basis "
            "point of trading cost."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Neither the league omen (*t* = "
            f"{R['nl_welch_t']:.2f}, permutation *p* = {R['nl_perm_p']:.2f}) nor the city omen "
            f"(*t* = {R['ny_welch_t']:.2f}, and pointing the wrong direction) clears any bar. "
            f"With {R['n_events']} seasons split roughly in half, there just isn't enough "
            "information in a baseball trophy to move a $40-trillion market.\n"
            "- **Tradability — Mirage.** Both omen-timing strategies lose to buy-and-hold; sitting "
            "out a mostly-up market is expensive and there's no edge to offset it.\n"
            "- **Beats a coin? — Busted.** Neither omen's hit rate reliably beats a flat 50% coin "
            f"flip, let alone the market's real {R['uncond_up_pct']:.0f}% up-rate — and the city "
            "omen's hit rate is numerically *below* half."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The whole sports-omen family fails the same way.** This desk has now tested "
            "football ([158-super-bowl](../../158-super-bowl/)), the Olympics "
            "([234-olympic-year](../../234-olympic-year/)), Eurovision "
            "([708-eurovision-effect](../../708-eurovision-effect/)) and now baseball — all "
            "collapse once you check the correct baseline and run a permutation test. The lesson "
            "generalizes: an annual calendar event with a binary or small-cardinality label will "
            "*always* look mildly correlated with a mostly-up market by chance, and the fix is "
            "always the same two checks — the honest base rate, and a shuffle test.\n"
            "- **A pre-1950 or minor-league extension** could add power, but MLB expansion, league "
            "realignments (interleague play since 1997, the Astros' 2013 league switch) make the "
            "AL/NL label itself progressively less meaningful the further you push it.\n\n"
            "*Think there's a real omen hiding in some other sport's calendar? Show a Welch "
            "*t* ≥ 2 against the correct baseline, survive a permutation test, and we'll test it "
            "next.*"
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
            "# The World-Series-Effect — a quantitative teardown 🔬\n"
            "### League-omen and city-omen Welch/permutation splits · the base-rate-corrected "
            "binomial hit-rate test · a coin-flip myth-check · an omen-timing strategy vs "
            "buy-and-hold · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "This is the Super Bowl Indicator's baseball cousin — same claim shape, same "
            "absence of a proposed mechanism, tested with the same rigor as every other study "
            "on this desk.\n\n"
            "> ⚠️ **Data note.** ^GSPC daily close (1949-11-01 → 2026-06-30), yfinance, cached, "
            "resampled to December-close-to-December-close calendar-year returns (price-only — "
            "^GSPC carries no dividends). **76 hardcoded World Series champions, 1950 → 2025** "
            "(75 actually played — 1994's players' strike is a named quirk, not imputed); "
            f"**{R['n_events']} scoreable events** pair a champion with a COMPLETE next calendar "
            "year (the still-open 2025 champion has no scoreable next year yet). No survivorship "
            "on the Signal axis (^GSPC is an index, not a survivor panel). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_gspc"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | league-omen Welch **t = {R['nl_welch_t']:.2f}** "
            f"(permutation *p* = {R['nl_perm_p']:.4f}, binomial *p* = {R['nl_binom_p']:.4f} vs "
            f"{R['uncond_up_pct']:.1f}% baseline); city-omen Welch **t = {R['ny_welch_t']:.2f}** "
            f"(permutation *p* = {R['ny_perm_p']:.4f}), wrong-signed |\n"
            f"| **Tradability** | `MIRAGE` | omen-timing underperforms buy-and-hold by "
            f"**{R['nl_strat_adv']:+.2f} pp/yr** (league) / **{R['ny_strat_adv']:+.2f} pp/yr** (city) |\n"
            f"| **Beats a coin?** | `BUSTED` | coin-test *p* = {R['nl_coin_p']:.4f} (league), "
            f"{R['ny_coin_p']:.4f} (city, hit rate {R['ny_hit']:.1f}% — below 50%) |\n\n"
            "> 💡 In plain words: there's a pennant winner every year and the market goes up "
            "most years anyway; multiply those two facts together and you get exactly the noise "
            "measured below — nothing more."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $R_{y+1}$ be the ^GSPC calendar-year simple return following World Series "
            "season $y$, and $L_y \\in \\{AL, NL\\}$ the champion's league (known by early "
            "November of year $y$ — public well before the Jan 1 entry, zero look-ahead). The "
            "primary claim, ported from the Super Bowl Indicator's NFC/AFC mnemonic:\n\n"
            "- **H₁ (league omen).** $E[R_{y+1} \\mid L_y = NL] > E[R_{y+1} \\mid L_y = AL]$, and "
            "the split is large enough to certify.\n\n"
            "The brief's city-mythology variant, with $NY_y \\in \\{0,1\\}$ flagging a New York "
            "franchise (Yankees/Giants/Mets/Brooklyn Dodgers) champion:\n\n"
            "- **H₂ (city omen).** $E[R_{y+1} \\mid NY_y = 1] > E[R_{y+1} \\mid NY_y = 0]$.\n\n"
            "Both claims lack any stated economic mechanism — unlike, say, this desk's "
            "[FOMC vol-crush](../../637-fomc-vol-crush/) study — so the honest prior going in is "
            "already low; the job is to measure the data anyway and report exactly how it fails."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "World Series seasons are **single, non-overlapping annual events** (the same design "
            "as [158-super-bowl](../../158-super-bowl/)), so the planned primary is a **Welch "
            "*t*** on the two-group mean-return split. Because the null hypothesis's correct "
            "reference point is the sample's own **unconditional up-rate** — not a 50% coin — the "
            "binomial hit-rate test uses $p_0 = $ that up-rate as its null, exactly the correction "
            "158 applies to the Super Bowl Indicator. A **20,000-draw permutation test** on the "
            "two-sided mean contrast sidesteps any normality assumption at this small n, and a "
            "separate, looser **coin-flip test** ($p_0 = 0.5$) is reported purely as the grey "
            "\"myth-check\" third axis — never used to certify the Signal stamp."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_played']} World Series actually played {R['start']} → "
            f"{R['end']} (1994 cancelled by the players' strike, named and dropped), hardcoded "
            "from MLB's postseason record.\n"
            f"- **Tape.** ^GSPC daily close, resampled to December-to-December calendar-year "
            "returns; only COMPLETE years are ever used. As-of 2026-06-30 (last complete month) "
            f"— {R['n_events']} events score a champion against a complete next year.\n"
            "- **Headline.** Welch *t* + a base-rate-corrected binomial hit-rate test (Wilson "
            "interval) + a 20,000-draw permutation test, run for both the league omen and the "
            "city omen.\n"
            "- **Myth-check.** A separate, looser two-sided binomial test against a flat 50% "
            "coin — reported because it's the question the folklore itself would ask.\n"
            "- **Execution (tradability).** One documented convention: enter at the World Series "
            "season's December 31 close, hold the following calendar year — the champion's "
            "league/city is public weeks before that date, zero look-ahead.\n"
            "- **Control.** Synthetic annual-return generator with a random bull/bear flag "
            "(base rate matched to the real NL share) and a planted omen boost; the null must "
            "not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The league omen — Welch split, permutation, and the base-rate trap"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull_nl = (EV['league'] == 'NL').to_numpy()\n"
            "    s = st.omen_stats(EV, bull_nl)\n"
            "    mb, mr, t, pp = s['mean_bull_pct'], s['mean_bear_pct'], s['welch_t'], s['perm_p']\n"
            "    hit, base, bp = s['hit_rate_pct'], s['uncond_up_pct'], s['binom_p']\n"
            "else:\n"
            "    mb, mr, t, pp = R['nl_mean_bull'], R['nl_mean_bear'], R['nl_welch_t'], R['nl_perm_p']\n"
            "    hit, base, bp = R['nl_hit'], R['uncond_up_pct'], R['nl_binom_p']\n"
            "print(f\"NL-preceded {mb:+.2f}%  vs  AL-preceded {mr:+.2f}%   \"\n"
            "      f\"Welch t = {t:+.2f}   permutation p = {pp:.4f}\")\n"
            "print(f\"hit rate {hit:.1f}%  vs  unconditional up-rate {base:.1f}%   \"\n"
            "      f\"binomial p = {bp:.4f}\")\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['NL','AL'], [mb, mr], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([mb, mr]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title(f'Welch t = {t:+.2f}')\n"
            "a1.set_ylabel('mean next-year return (%)')\n"
            "a2.bar(['omen hit rate','unconditional\\nup-rate'], [hit, base], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([hit, base]): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylim(0, 100); a2.set_title(f'binomial p = {bp:.3f} (vs correct baseline)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: {R['nl_contrast']:+.1f} pp looks like something until you see "
            f"the *t* ({R['nl_welch_t']:.2f}, bar is 2) and the permutation *p* "
            f"({R['nl_perm_p']:.2f} — chance alone reproduces this 30% of the time). Worse, the "
            f"omen's own hit rate ({R['nl_hit']:.1f}%) sits *below* the market's unconditional "
            f"up-rate ({R['uncond_up_pct']:.1f}%) — following it would make your directional "
            "calls *less* accurate than doing nothing."
        ),
        md(
            "### 4b · The city omen — same design, wrong sign"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull_ny = EV['is_ny'].to_numpy()\n"
            "    s2 = st.omen_stats(EV, bull_ny)\n"
            "    mb2, mr2 = s2['mean_bull_pct'], s2['mean_bear_pct']\n"
            "    t2, pp2 = s2['welch_t'], s2['perm_p']\n"
            "    hit2, bp2 = s2['hit_rate_pct'], s2['binom_p']\n"
            "else:\n"
            "    mb2, mr2 = R['ny_mean_bull'], R['ny_mean_bear']\n"
            "    t2, pp2 = R['ny_welch_t'], R['ny_perm_p']\n"
            "    hit2, bp2 = R['ny_hit'], R['ny_binom_p']\n"
            "print(f\"NY-preceded {mb2:+.2f}%  vs  everyone else {mr2:+.2f}%   \"\n"
            "      f\"Welch t = {t2:+.2f}   permutation p = {pp2:.4f}\")\n"
            "print(f\"hit rate {hit2:.1f}%   binomial p = {bp2:.4f}\")\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['New York\\nchampion','everyone\\nelse'], [mb2, mr2], color=[GREY, GREY], width=.55)\n"
            "for i,v in enumerate([mb2, mr2]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title(f'Welch t = {t2:+.2f} (wrong-signed vs the claim)')\n"
            "ax.set_ylabel('mean next-year return (%)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the point estimate runs {R['ny_contrast']:+.2f} pp — the "
            "*opposite* of the claimed direction — with *t* = "
            f"{R['ny_welch_t']:.2f} and permutation *p* = {R['ny_perm_p']:.2f}. Sixteen "
            "NY-area titles cluster across both strong- and weak-market decades; there is no "
            "pattern here, and the sign flip is a useful reminder that a plausible-sounding "
            "story (\"Wall Street's hometown team\") can point either way once you actually look."
        ),
        md(
            "### 4c · Myth-check — does either omen beat a flat coin?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp1, cp2 = s['coin_p'], s2['coin_p']\n"
            "else:\n"
            "    cp1, cp2 = R['nl_coin_p'], R['ny_coin_p']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar(['league omen','city omen'], [hit, hit2], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(50, ls='--', c='k', lw=1, label='coin flip (50%)')\n"
            "for i,(v,p) in enumerate([(hit, cp1), (hit2, cp2)]):\n"
            "    ax.annotate(f'{v:.1f}%\\n(coin p={p:.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 100); ax.legend()\n"
            "ax.set_ylabel('hit rate (%)')\n"
            "ax.set_title('Neither omen reliably beats a coin -- and the city omen is BELOW it')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: coin-test *p* = {R['nl_coin_p']:.2f} (league) and "
            f"{R['ny_coin_p']:.2f} (city). The city omen's hit rate ({R['ny_hit']:.1f}%) is "
            "numerically below 50% — if anything, a New York title is a very mild *contrary* "
            "indicator here, though not a certified one either."
        ),
        md(
            "### 4d · Could you trade either omen?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t1r = st.timing_strategy(EV, bull_nl); t2r = st.timing_strategy(EV, bull_ny)\n"
            "    bah = t1r['bah_ann_pct']\n"
            "    nl_a, nl_adv = t1r['strat_ann_pct'], t1r['ann_advantage_pct']\n"
            "    ny_a, ny_adv = t2r['strat_ann_pct'], t2r['ann_advantage_pct']\n"
            "else:\n"
            "    bah = R['bah_ann']\n"
            "    nl_a, nl_adv = R['nl_strat_ann'], R['nl_strat_adv']\n"
            "    ny_a, ny_adv = R['ny_strat_ann'], R['ny_strat_adv']\n"
            "print(f\"buy-and-hold {bah:+.2f}%/yr\")\n"
            "print(f\"NL-omen timing {nl_a:+.2f}%/yr  (advantage {nl_adv:+.2f} pp/yr)\")\n"
            "print(f\"NY-omen timing {ny_a:+.2f}%/yr  (advantage {ny_adv:+.2f} pp/yr)\")\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['buy & hold','NL-omen\\ntiming','NY-omen\\ntiming'], [bah, nl_a, ny_a],\n"
            "       color=[GREEN, RED, RED], width=.6)\n"
            "for i,v in enumerate([bah, nl_a, ny_a]): ax.annotate(f'{v:+.2f}%/yr',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualized return')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **{R['nl_strat_adv']:+.2f} pp/yr** (league) and "
            f"**{R['ny_strat_adv']:+.2f} pp/yr** (city) of underperformance versus simply "
            "staying invested, with a single rebalance a year (transaction costs don't move this "
            "conclusion). Sitting out roughly half — or three-quarters — of a market that's up "
            "most of the time is expensive, and there's no certified signal underneath to repay "
            "it. **H(tradability) fails on both variants; Tradability = MIRAGE.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic annual-return generator: a random bull/bear flag (base rate matched to "
            "the real NL share, 35/74 ≈ 47%) with a TUNABLE planted boost added to bull-flagged "
            "years' next-year return. The null (boost = 0) is checked over **20 seeds** — never a "
            "single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    syn, _ = data.synthetic_world(boost=0.0, seed=709 + s_)\n"
            "    null_ts.append(st.synthetic_detect(syn)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "syn, _ = data.synthetic_world(boost=10.0, seed=709)\n"
            "planted_t = st.synthetic_detect(syn)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (boost=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted boost = +10 pp')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (bull-flag vs bear-flag)')\n"
            "ax.set_title('Control: no null fires; a planted omen lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = +{R['syn_null_mean']:.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted +10 pp omen reads t = +{R['syn_planted_t']:.2f}. The machinery is "
            "unbiased — the real-tape null result above is a genuine finding, not a blind spot in "
            "the harness. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — league-omen Welch t = **{R['nl_welch_t']:.2f}** (permutation "
            f"p = {R['nl_perm_p']:.4f}, binomial p = {R['nl_binom_p']:.4f} vs the correct "
            f"{R['uncond_up_pct']:.1f}% baseline, hit rate {R['nl_hit']:.1f}%); city-omen Welch "
            f"t = **{R['ny_welch_t']:.2f}** (permutation p = {R['ny_perm_p']:.4f}), wrong-signed "
            f"and hit rate {R['ny_hit']:.1f}% (below a coin). n = {R['n_events']} seasons "
            f"({R['start']} → {R['end']-1}); neither variant has a proposed mechanism.\n"
            f"- **Tradability `MIRAGE`** — omen-timing underperforms buy-and-hold by "
            f"{R['nl_strat_adv']:+.2f} pp/yr (league) and {R['ny_strat_adv']:+.2f} pp/yr (city); "
            "sitting out a mostly-up market has a real cost and there's no edge to offset it.\n"
            "- **Beats a coin? `BUSTED`** — league-omen coin-test p = "
            f"{R['nl_coin_p']:.4f}; city-omen coin-test p = {R['ny_coin_p']:.4f} with a hit rate "
            "numerically below 50%. Neither omen reliably beats a flat coin flip, let alone the "
            "market's own upward bias."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The whole sports-omen family collapses the same way.** Football "
            "([158-super-bowl](../../158-super-bowl/)), the Olympics "
            "([234-olympic-year](../../234-olympic-year/)), Eurovision "
            "([708-eurovision-effect](../../708-eurovision-effect/)) and now baseball all fail "
            "once tested against the correct baseline with a permutation check — a reusable "
            "lesson for any calendar-labeled 'omen' claim, sporting or otherwise.\n"
            "- **A natural extension** is testing every possible binary split of the champion "
            "table (division, whether the champion was a wild card, series length) with an "
            "explicit multiple-comparisons correction — the 158-super-bowl playbook — though "
            "with n ≈ 74 the power floor makes most such splits uninformative before you even "
            "start.\n"
            "- **Dedup map:** [158-super-bowl](../../158-super-bowl/) (the football original, "
            "identical debunk shape), [235-world-cup-effect](../../235-world-cup-effect/) (a "
            "during-tournament drift claim, not a winner effect), "
            "[234-olympic-year](../../234-olympic-year/) (same ^GSPC/permutation machinery, a "
            "symmetric calendar marker) and [708-eurovision-effect](../../708-eurovision-effect/) "
            "(the omen family, a song contest instead of a ballgame).\n\n"
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
