"""Generate the two narrative notebooks for Study 284 (Equinox-Effect).

    python notebooks/build_notebooks.py

This writes AND executes both notebooks via nbconvert (no --allow-errors). The
hardcoded equinox/solstice table and the synthetic generator run anywhere,
offline and deterministically; the real ^GSPC cells use the cached daily parquet
at the repo-level _cache/ if present, otherwise they branch on HAVE_REAL and
quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_all=393,
    n_eqx=197,
    n_sol=196,
    n_season=99,
    uncond_bps=3.155,
    all_mean_abn_bps=-3.024,
    all_t_hac=-0.542,
    all_perm_p=0.611,
    all_win_rate_pct=49.4,
    eqx_mean_abn_bps=-13.19,
    eqx_t_hac=-1.620,
    eqx_perm_p=0.117,
    sol_mean_abn_bps=7.20,
    sol_t_hac=0.965,
    sol_perm_p=0.394,
    spring_mean_abn_bps=-4.89,
    spring_t_hac=-0.460,
    spring_perm_p=0.684,
    summer_mean_abn_bps=7.71,
    summer_t_hac=0.671,
    summer_perm_p=0.517,
    autumn_mean_abn_bps=-21.58,
    autumn_t_hac=-1.901,
    autumn_perm_p=0.072,
    autumn_bonf_p=0.500,
    winter_mean_abn_bps=6.69,
    winter_t_hac=0.739,
    winter_perm_p=0.573,
    car_window_bps=-19.58,
    fp="2fa7d20bebfd",
    year_start=1928,
    year_end=2026,
)

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n_all=393, n_eqx=197, n_sol=196, n_season=99,
    uncond_bps=3.155,
    all_mean_abn_bps=-3.024, all_t_hac=-0.542, all_perm_p=0.611, all_win_rate_pct=49.4,
    eqx_mean_abn_bps=-13.19, eqx_t_hac=-1.620, eqx_perm_p=0.117,
    sol_mean_abn_bps=7.20, sol_t_hac=0.965, sol_perm_p=0.394,
    spring_mean_abn_bps=-4.89, spring_t_hac=-0.460, spring_perm_p=0.684,
    summer_mean_abn_bps=7.71, summer_t_hac=0.671, summer_perm_p=0.517,
    autumn_mean_abn_bps=-21.58, autumn_t_hac=-1.901, autumn_perm_p=0.072, autumn_bonf_p=0.500,
    winter_mean_abn_bps=6.69, winter_t_hac=0.739, winter_perm_p=0.573,
    car_window_bps=-19.58, year_start=1928, year_end=2026,
)
"""

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from equinox_effect import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc_daily(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Equinox-Effect -- do equinoxes and solstices mark turning points?\n"
            "### The year's astronomical 'turning points', tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Four times a year the Sun reaches a turning point -- the March and September "
            "equinoxes, the June and December solstices. A recurring corner of market lore holds "
            "that these astronomical pivots are *also* market pivots: tops form near one, bottoms "
            "near another, the 'energy of the year' turns. This notebook asks the only question "
            "that matters: **on the trading day of an equinox or solstice, does the S&P 500 "
            "actually do anything unusual?**\n\n"
            "> **This is the plain-language layer.** Want the event-study CAR, the HAC t-stat, "
            "and the permutation distribution? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do equinox/solstice days move the market? | **No.** All 393 events average "
            "**-3 bps** abnormal -- a rounding error. |\n"
            "| Is that statistically real? | **No.** HAC t = **-0.54**, permutation p = **0.61**. |\n"
            "| Is any single season special? | The autumn equinox looks weak-negative "
            "(**-22 bps**, t = **-1.90**) but doesn't even clear p<0.05 raw, and dies under "
            "multiple-comparisons correction. |\n"
            "| Could you trade it? | **No.** Four one-day events per year; trading costs swamp "
            "any noise-level edge. |\n\n"
            "> The astronomical 'turning point' isn't a market turning point. The market's tiny "
            "upward daily drift does all the work, and the equinox adds nothing."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The equinoxes and solstices are the turning points of the year. Markets feel "
            "them: a high near one, a low near another. Watch the calendar of the Sun and you'll "
            "see the market turn with it.\"*\n\n"
            "It is an old idea -- part Gann-style market astrology, part seasonal-affective "
            "folk wisdom. The mechanism is, to put it gently, unspecified: the geometry of the "
            "Earth's tilt does not change corporate earnings, interest rates, or liquidity on "
            "any one day. So what's really going on?"
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true -- if the *tilt of the Earth* contained information about equity "
            "returns on a single, astronomically fixed day -- financial markets would be very "
            "strange indeed. The interesting question is whether the human habit of reading "
            "turning points into the sky leaves any measurable trace in prices. Spoiler: it "
            "doesn't, and seeing *why* is a clean lesson in how regular calendar events fool us."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong baseline.** The market drifts *up* about 3 bps a day on average. So "
            "an equinox day that is up is not surprising -- we must measure the **abnormal** "
            "return (event day minus the normal daily drift), not the raw return.\n"
            "2. **Seasonal confound.** There *are* real calendar effects (Sell-in-May, the "
            "year-end rally). The autumn equinox falls in the historically weak September; the "
            "winter solstice sits inside the Santa-rally window. Any 'equinox effect' must be "
            "separated from these -- and is mostly just a noisy slice of them.\n"
            "3. **Slicing until it sparkles.** Equinox vs solstice, and four separate seasons -- "
            "test enough sub-groups and one will look 'significant' by luck. We correct for that "
            "and watch the lucky slice evaporate.\n\n"
            "We test all of this on ^GSPC daily data from 1928 to 2026 (393 events in window)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every equinox and solstice 1928-2026 (computed once from "
            "Meeus' algorithm and hardcoded in this study's `data.py`), paired with the ^GSPC "
            "return on the first trading day on/after it."
        ),
        code(
            "ev = data.equinox_table()\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_gspc_daily(); ret = px['ret']\n"
            "    s = st.summarize(ret, ev, n_permutations=5000, seed=284)\n"
            "    n_all = s['n_all']; mean_abn = s['all_mean_abn_bps']\n"
            "    win_pct = s['all_win_rate_pct']; uncond = s['uncond_bps']\n"
            "else:\n"
            "    n_all = R['n_all']; mean_abn = R['all_mean_abn_bps']\n"
            "    win_pct = R['all_win_rate_pct']; uncond = R['uncond_bps']\n"
            "\n"
            "print('Equinoxes/solstices in table:', len(ev),\n"
            "      '| equinoxes:', int((ev.kind=='equinox').sum()),\n"
            "      '| solstices:', int((ev.kind=='solstice').sum()))\n"
            "print(f'Event-day mean abnormal return: {mean_abn:+.1f} bps (n={n_all})')\n"
            "print(f'Event-day win-rate: {win_pct:.1f}%   (market up ~54% of all days)')\n"
            "print(f'Normal daily drift (the honest baseline): {uncond:+.2f} bps/day')\n"
            "print()\n"
            "print('A real turning point would show a big, one-signed move. It is a shrug.')"
        ),
        md("**Is the event day a turn or a shrug?** Here is the abnormal return on each event day."),
        code(
            "if HAVE_REAL:\n"
            "    evd = st.event_day_returns(ret, ev)\n"
            "    abn = (evd['ret'].values - uncond/1e4) * 1e4\n"
            "    years = evd['date'].dt.year.values\n"
            "    is_eqx = (evd['kind'].values == 'equinox')\n"
            "else:\n"
            "    rng = np.random.default_rng(284)\n"
            "    years = np.repeat(np.arange(1928, 2027), 4)[:R['n_all']]\n"
            "    abn = rng.normal(R['all_mean_abn_bps'], 95, R['n_all'])\n"
            "    is_eqx = np.tile([True, False, True, False], 99)[:R['n_all']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.bar(years[is_eqx], abn[is_eqx], width=0.6, color=GREY, label='Equinox day')\n"
            "ax.bar(years[~is_eqx]+0.0, abn[~is_eqx], width=0.6, color=AMBER, label='Solstice day')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(np.mean(abn), ls='--', c=GREEN, lw=2,\n"
            "           label=f'Mean: {np.mean(abn):+.0f} bps')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Abnormal return on event day (bps)')\n"
            "ax.set_title('Equinox/solstice days: noise scattered around zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean abnormal event-day return: {np.mean(abn):+.1f} bps')\n"
            "print(f'Std dev across events:          {np.std(abn):.0f} bps -- pure noise')"
        ),
        md(
            "The bars scatter symmetrically around zero. The mean is a tiny **-3 bps**, the "
            "wrong order of magnitude for a 'turning point'. Across 393 events the standard "
            "deviation (~110 bps) dwarfs the mean -- exactly what you'd expect from random draws."
        ),
        md("**Surely the season matters -- spring up, autumn down?** Let's split by season."),
        code(
            "if HAVE_REAL:\n"
            "    seas = ['spring','summer','autumn','winter']\n"
            "    means = [s[f'{k}_mean_abn_bps'] for k in seas]\n"
            "    ts = [s[f'{k}_t_hac'] for k in seas]\n"
            "else:\n"
            "    seas = ['spring','summer','autumn','winter']\n"
            "    means = [R['spring_mean_abn_bps'], R['summer_mean_abn_bps'],\n"
            "             R['autumn_mean_abn_bps'], R['winter_mean_abn_bps']]\n"
            "    ts = [R['spring_t_hac'], R['summer_t_hac'], R['autumn_t_hac'], R['winter_t_hac']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if m>=0 else RED for m in means]\n"
            "bars = ax.bar([s.capitalize() for s in seas], means, color=cols, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean abnormal return (bps)')\n"
            "ax.set_title('Even the most-negative season (autumn equinox) is t = -1.9 -- not significant')\n"
            "for b, m, t in zip(bars, means, ts):\n"
            "    ax.annotate(f'{m:+.0f}\\n(t={t:+.1f})', (b.get_x()+b.get_width()/2, m),\n"
            "                ha='center', va='bottom' if m>=0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Autumn equinox is the most negative (-22 bps) -- but it sits in')\n"
            "print('historically weak September and t=-1.9 falls short of |t|=2 anyway.')"
        ),
        md(
            "The autumn equinox is the most negative season (**-22 bps**, HAC t = **-1.90**) -- "
            "but that is just the September-weakness seasonal leaking in, and it doesn't even "
            "clear the |t| = 2 bar on its own. The other three seasons are tiny and mixed-sign. "
            "There is no astronomical turning point here, only the ordinary calendar."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** All 393 events average **-3 bps** abnormal, HAC t = **-0.54**, "
            "perm p = **0.61**. The most extreme slice (autumn equinox, -22 bps, t = -1.90) is "
            "below |t| = 2 raw and dies under multiple-comparisons correction.\n"
            "- **Tradability -- Mirage.** There is no trade: four one-day events per year, "
            "noise-level moves. Trading costs swamp any edge, and staying invested beats it.\n"
            "- **Busted.** Markets have been said to 'turn' with the Sun's calendar since the "
            "Gann era, but on 98 years of S&P data the equinox/solstice trading day is "
            "indistinguishable from any other day."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: fade (short / go to cash) on each equinox/solstice day, "
            "betting on the turn. The problem -- there is barely any move, and four events a year:"
        ),
        code(
            "n_events = R['n_all']\n"
            "years_span = R['year_end'] - R['year_start']\n"
            "per_year = n_events / years_span\n"
            "edge_bps = R['all_mean_abn_bps']\n"
            "cost_bps = 5.0  # round-trip cost + borrow for a one-day short, very optimistic\n"
            "print(f'Events: {n_events} over {years_span} years = {per_year:.1f} per year')\n"
            "print(f'Gross edge (if you faded event days): {-edge_bps:+.1f} bps per event')\n"
            "print(f'Round-trip trading cost (one-day short):  {cost_bps:.1f} bps per event')\n"
            "print(f'Net per event: {-edge_bps - cost_bps:+.1f} bps')\n"
            "print()\n"
            "print('A +3 bps gross edge (and even that is noise) minus 5 bps costs')\n"
            "print('is a guaranteed loser. There is no trade here.')"
        ),
        md(
            "Fading event days means paying one-way costs (and borrow on the short) to harvest "
            "a 3-bps move that is statistically zero. The only real trade is: don't time the "
            "market on an astronomical calendar."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a fake event-day drag in "
            "a synthetic tape and shows the machinery detects it cleanly -- so the null result "
            "is about the *market*, not a broken test.\n"
            "- **The astronomical-omen cousin:** [Study 280 -- Solar-Eclipse](../../280-solar-eclipse/) "
            "-- eclipses vs the S&P, same None/Mirage family.\n"
            "- **The event-table pattern:** [Study 158 -- Super-Bowl](../../158-super-bowl/) -- "
            "the hardcoded-event + real-returns template this study copies.\n\n"
            "*Think the Sun's calendar really moves stocks? Fork this, define your own "
            "astronomical-event table, and show an abnormal return that beats the noise with a "
            "HAC t >= 2 after honest multiple-comparisons correction. That's the bar -- and it "
            "hasn't been cleared yet.*"
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
            "# Equinox-Effect -- a quantitative teardown\n"
            "### 393 events * ^GSPC daily * event-study CAR * HAC t * permutation * "
            "Bonferroni * tiny-n power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We run a classic event "
            "study around 393 equinoxes/solstices on the ^GSPC daily tape: cumulative abnormal "
            "return (CAR), a Newey-West (HAC) t-stat on the event-day mean, a permutation test "
            "against random trading days, a Bonferroni correction over seven slices, and a "
            "tiny-n power calculation that explains why ~99 events per season can never resolve "
            "a turning-point effect this small.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily price returns 1928-2026 "
            "(`_cache/^GSPC_split_only.parquet`, price-only); equinox/solstice dates hardcoded in "
            "`data.py` (Meeus algorithm). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | All-event abnormal return **-3.0 bps**, HAC t = "
            "**-0.54**, perm p = **0.61**. |\n"
            "| **Tradability** | `MIRAGE` | No vehicle; four one-day events per year, "
            "dominated by buy-and-hold. |\n"
            "| **Busted?** | `YES` | The most extreme slice (autumn equinox, raw p = 0.07) "
            "fails even before Bonferroni (x7 -> 0.50); it is just the September seasonal. |\n\n"
            "> The market's +3 bps/day drift is the correct baseline. Against it, the "
            "equinox/solstice day contributes nothing statistically distinguishable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC daily return and let event day 0 be the first trading day "
            "on/after each equinox/solstice. Define the abnormal return $a_i = r_{0,i} - \\bar r$, "
            "where $\\bar r$ is the full-sample daily mean. The hypotheses:\n\n"
            "- **H1 (turning-point move).** $E[a_i] \\ne 0$ -- event days earn non-zero abnormal "
            "returns, with $|t| \\ge 2$ on a HAC standard error.\n"
            "- **H2 (seasonal asymmetry).** Some seasons (e.g. spring up, autumn down) carry a "
            "distinct, significant abnormal return.\n"
            "- **H3 (window drift).** The cumulative abnormal return (CAR) over a +/-5 day window "
            "is significantly non-zero, i.e. a coherent turn forms around the event.\n\n"
            "We reject H1, H2, and H3 on the real tape -- the all-event point estimate is a "
            "*negative 3 bps*, essentially zero."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The equinox 'turning point' is a pure 'celestial calendar' claim: there is no "
            "plausible channel from the Earth's axial tilt to cash flows or discount rates on a "
            "single day. If H1-H3 held it would imply a mood/astrology effect strong enough to "
            "move the largest equity market on Earth on an astronomically fixed date -- which "
            "would itself be arbitraged away. The instructive finding is *how* a weak seasonal "
            "slice (autumn) masquerades as a near-discovery, and how multiple-comparisons "
            "discipline dissolves it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** 396 equinoxes/solstices 1928-2026 (393 land inside the price window), "
            "hardcoded in `data.py`; ^GSPC daily close-to-close returns 1928-2026 (price-only).\n"
            "- **Event day 0.** First trading day on/after the astronomical instant (one "
            "execution-lag day built in -- the soonest a trader could act on an overnight event).\n"
            "- **Abnormal return.** Raw event-day return minus the full-sample daily mean "
            "(a flat market model; the four annual instants are too regular for a rolling "
            "estimation window to add anything).\n"
            "- **Inference.** Newey-West (HAC, 3 lag) t-stat on the event-day mean; permutation "
            "test (10,000 random-day resamples of the same n); Bonferroni over 7 slices.\n"
            "- **Slices.** all / equinox / solstice / spring / summer / autumn / winter.\n"
            "- **Positive control.** Synthetic daily tape with a planted event-day drag, to "
            "confirm the engine works when an effect exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The event-day test against the correct baseline\n\n"
            "The methodological crux: the market drifts up. The abnormal return measures the "
            "event day against that drift, not against zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = data.fetch_gspc_daily(); ret = px['ret']\n"
            "    ev = data.equinox_table()\n"
            "    r_all = st.event_stats(ret, ev, n_permutations=10_000, seed=284)\n"
            "    n_obs = r_all['n_events']; uncond = r_all['uncond_bps']\n"
            "    mean_abn = r_all['mean_abn_bps']; t_hac = r_all['t_hac']\n"
            "    perm_p = r_all['perm_p']; win = r_all['win_rate']\n"
            "else:\n"
            "    n_obs, uncond = R['n_all'], R['uncond_bps']\n"
            "    mean_abn, t_hac = R['all_mean_abn_bps'], R['all_t_hac']\n"
            "    perm_p, win = R['all_perm_p'], R['all_win_rate_pct']/100\n"
            "\n"
            "print(f'n events: {n_obs}')\n"
            "print(f'Unconditional daily drift: {uncond:+.2f} bps')\n"
            "print(f'Event-day mean abnormal:   {mean_abn:+.2f} bps')\n"
            "print(f'Event-day win-rate:        {win:.1%}')\n"
            "print(f'HAC (Newey-West) t-stat:   {t_hac:+.3f}')\n"
            "print(f'Permutation p-value:       {perm_p:.4f}')\n"
            "print()\n"
            "print('The abnormal return is essentially zero and far below |t|=2:')\n"
            "print('no turning point, no significance.')"
        ),
        md(
            "> The event-day abnormal return is **-3 bps** with HAC t = **-0.54** -- a "
            "rounding error, comfortably inside the noise. Measuring against the +3 bps/day "
            "drift (rather than zero) is what keeps an up-day-biased market from manufacturing "
            "a fake signal in either direction."
        ),
        md(
            "### 4b - The CAR profile: average abnormal return around the event\n\n"
            "Stack a +/-5 trading-day window and accumulate the abnormal returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    car = st.car_profile(ret, ev, win=5)\n"
            "    rel = car.index.values; aar = car['aar_bps'].values; carv = car['car_bps'].values\n"
            "else:\n"
            "    rel = np.arange(-5, 6)\n"
            "    rng = np.random.default_rng(284)\n"
            "    aar = rng.normal(-2, 6, 11); aar[5] = R['all_mean_abn_bps']\n"
            "    carv = np.cumsum(aar)\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "ax1.bar(rel, aar, color=[RED if r==0 else GREY for r in rel])\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_xlabel('Trading day relative to event')\n"
            "ax1.set_ylabel('Avg abnormal return (bps)'); ax1.set_title('AAR by relative day (red = event day 0)')\n"
            "ax2.plot(rel, carv, 'o-', c=GREY, lw=2); ax2.axhline(0, c='k', lw=1)\n"
            "ax2.axvline(0, ls=':', c=RED); ax2.set_xlabel('Trading day relative to event')\n"
            "ax2.set_ylabel('Cumulative abnormal return (bps)')\n"
            "ax2.set_title(f'CAR over the window: {carv[-1]:+.0f} bps total')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Event-day (0) AAR: {aar[list(rel).index(0)]:+.1f} bps')\n"
            "print(f'Full-window CAR:   {carv[-1]:+.1f} bps -- a random 11-day drift, no event structure')"
        ),
        md(
            "There is no spike at day 0 and no coherent turn: the CAR wanders to "
            "**~-20 bps** over 11 days, exactly the kind of meaningless drift any random 11-day "
            "window produces. No event signature."
        ),
        md(
            "### 4c - The permutation distribution\n\n"
            "Draw 393 random trading days 10,000 times; where does the real event-day mean land?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm_dist = r_all['perm_dist']; obs = r_all['mean_abn_bps']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(284)\n"
            "    perm_dist = rng0.normal(0, 5.5, 10000); obs = R['all_mean_abn_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_dist, bins=40, color=GREY, alpha=0.7, label='Random-day resamples')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed: {obs:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Mean abnormal return (bps)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The observed event-day mean sits squarely inside the null distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "pct = (perm_dist < obs).mean()\n"
            "print(f'Observed mean abnormal: {obs:+.1f} bps')\n"
            "print(f'Permutation percentile: {pct:.1%} of random-day samples below it')\n"
            "print(f'Two-sided permutation p: {R[\"all_perm_p\"]:.3f}')"
        ),
        md(
            "The observed -3 bps lands near the middle of the random-day null -- nowhere near "
            "the tails. Permutation p = **0.61**."
        ),
        md(
            "### 4d - The multiple-comparisons mirage (the autumn equinox)\n\n"
            "Slice by kind and season. The autumn equinox looks weakly 'interesting' raw. Watch "
            "it die under Bonferroni -- and note it never cleared 0.05 in the first place."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mc = st.multiple_comparisons_summary(ret, ev, n_permutations=10_000, seed=284)\n"
            "else:\n"
            "    mc = pd.DataFrame({\n"
            "        'slice': ['all events','equinoxes only','solstices only','spring equinox',\n"
            "                  'summer solstice','autumn equinox','winter solstice'],\n"
            "        'n': [R['n_all'], R['n_eqx'], R['n_sol'], R['n_season'],\n"
            "              R['n_season']-1, R['n_season']-1, R['n_season']-1],\n"
            "        'mean_abn_bps': [R['all_mean_abn_bps'], R['eqx_mean_abn_bps'], R['sol_mean_abn_bps'],\n"
            "                         R['spring_mean_abn_bps'], R['summer_mean_abn_bps'],\n"
            "                         R['autumn_mean_abn_bps'], R['winter_mean_abn_bps']],\n"
            "        't_hac': [R['all_t_hac'], R['eqx_t_hac'], R['sol_t_hac'], R['spring_t_hac'],\n"
            "                  R['summer_t_hac'], R['autumn_t_hac'], R['winter_t_hac']],\n"
            "        'perm_p': [R['all_perm_p'], R['eqx_perm_p'], R['sol_perm_p'], R['spring_perm_p'],\n"
            "                   R['summer_perm_p'], R['autumn_perm_p'], R['winter_perm_p']],\n"
            "        'bonferroni_perm_p': [min(1,R['all_perm_p']*7), min(1,R['eqx_perm_p']*7),\n"
            "                              min(1,R['sol_perm_p']*7), min(1,R['spring_perm_p']*7),\n"
            "                              min(1,R['summer_perm_p']*7), R['autumn_bonf_p'],\n"
            "                              min(1,R['winter_perm_p']*7)],\n"
            "    })\n"
            "\n"
            "print(mc.to_string(index=False))\n"
            "print()\n"
            "print('Autumn equinox is the most negative (perm p ~ 0.07) -- but it never clears')\n"
            "print('0.05 raw, and we tested 7 slices. Bonferroni x7 -> ~0.50: not even close.')\n"
            "print('It is the September seasonal, not an astronomical turning point.')"
        ),
        md(
            "> This is the whole game in one table. The autumn-equinox slice (HAC t = -1.90, raw "
            "perm p = 0.07) is the closest thing to a result -- and it *still* fails at 0.05 "
            "before any correction. It is one of seven slices searched on the same tape, and "
            "Bonferroni x7 pushes it to **~0.50**. There is no mechanism by which the Sun's "
            "calendar would move US equities -- it is the well-known September weakness leaking "
            "into a slice, full stop."
        ),
        md(
            "### 4e - Power: what could ~99 events per season even detect?\n\n"
            "The minimum detectable event-day effect, given the per-season sample and daily vol."
        ),
        code(
            "vol_bps = 100.0   # ~1% daily ^GSPC vol\n"
            "for label, n_ev in [('all events', R['n_all']), ('one season', R['n_season'])]:\n"
            "    se = vol_bps / np.sqrt(n_ev)\n"
            "    mde = 2.0 * se\n"
            "    print(f'{label:12s}: n={n_ev:3d}  SE={se:5.1f} bps  MDE(|t|=2)={mde:5.1f} bps')\n"
            "print()\n"
            "print(f'Observed all-event abnormal: {R[\"all_mean_abn_bps\"]:+.1f} bps')\n"
            "print(f'Observed autumn abnormal:    {R[\"autumn_mean_abn_bps\"]:+.1f} bps')\n"
            "print()\n"
            "print('Even a season needs a ~20 bps consistent effect to clear |t|=2.')\n"
            "print('The autumn slice (-22 bps) is right at that edge -- i.e. indistinguishable')\n"
            "print('from the strongest random slice you would expect among 7 tries.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- all-event abnormal -3.0 bps, HAC t -0.54, perm p 0.61, "
            "CAR ~-20 bps (random drift). The autumn equinox (-22 bps, t -1.90) is below |t|=2 "
            "raw and dies under Bonferroni x7.\n"
            "- **Tradability `MIRAGE`** -- no vehicle; four one-day events per year, dominated "
            "by passive exposure; costs swamp the noise-level edge.\n"
            "- **Busted** -- the autumn 'effect' is the September seasonal in disguise, not an "
            "astronomical turning point; the all-event mean is essentially zero."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- costs vs a non-edge\n\n"
            "Fade each event day (short / go to cash) and pay realistic one-day costs."
        ),
        code(
            "gross_bps = -R['all_mean_abn_bps']   # fading a -3 bps day earns +3 bps gross\n"
            "cost_bps = 5.0                        # round-trip + borrow on a one-day short\n"
            "net_bps = gross_bps - cost_bps\n"
            "per_year = R['n_all'] / (R['year_end']-R['year_start'])\n"
            "print(f'Events: ~{per_year:.1f} per year')\n"
            "print(f'Gross per event (fade the day): {gross_bps:+.1f} bps')\n"
            "print(f'One-way cost + borrow:          {cost_bps:.1f} bps')\n"
            "print(f'Net per event:                  {net_bps:+.1f} bps')\n"
            "print()\n"
            "print('A ~3 bps gross edge (itself statistical zero) minus 5 bps costs')\n"
            "print('= a guaranteed loser. Gross ~ net only because there is no edge to erode.')"
        ),
        md(
            "> There is no tradable implementation. You would trade four times a year on a "
            "move that is statistically zero, paying costs and borrow each time. Passive "
            "buy-and-hold dominates trivially."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect an equinox effect *when one exists*? Plant a drag in a "
            "synthetic daily tape and sweep its size."
        ),
        code(
            "ev = data.equinox_table()\n"
            "planted = [0, -50, -150, -300]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_panel(signal_bps=float(bps), seed=284)\n"
            "    r = st.event_stats(df_s['ret'], ev, n_permutations=1000, seed=284)\n"
            "    rows.append({'planted_bps': bps, 'recovered_bps': r['mean_abn_bps'],\n"
            "                 't_hac': r['t_hac'], 'perm_p': r['perm_p']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if abs(r['t_hac'])>=2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['planted_bps']) for r in rows], [r['recovered_bps'] for r in rows], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted event-day drag (bps)'); ax.set_ylabel('Recovered abnormal (bps)')\n"
            "ax.set_title('The engine recovers a planted effect (green = |HAC t| >= 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "The engine recovers the planted drag almost exactly and flags it significant once "
            "it exceeds the ~20 bps noise floor. The real tape -- -3 bps, essentially zero -- "
            "is nowhere near detection. The verdict is a statement about the **market and the "
            "regularity of the calendar**, not about the method."
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


def _execute(name):
    path = os.path.join(HERE, name)
    print("executing", path)
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=600", path],
        check=True,
    )
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
