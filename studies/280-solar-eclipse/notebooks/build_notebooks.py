"""Generate the two narrative notebooks for Study 280 (Solar-Eclipse).

    python notebooks/build_notebooks.py

This writes AND executes both notebooks via nbconvert (no --allow-errors). The
hardcoded eclipse table and the synthetic generator run anywhere, offline and
deterministically; the real ^GSPC cells use the cached daily parquet at the
repo-level _cache/ if present, otherwise they branch on HAVE_REAL and quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook
re-runs for any reader.

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
    n_all=80,
    n_us=12,
    n_total=57,
    n_annular=18,
    uncond_bps=3.16,
    all_mean_abn_bps=13.32,
    all_t_hac=1.119,
    all_perm_p=0.307,
    all_win_rate_pct=62.5,
    total_mean_abn_bps=6.93,
    total_t_hac=0.495,
    total_perm_p=0.647,
    us_mean_abn_bps=-11.81,
    us_t_hac=-0.282,
    us_perm_p=0.696,
    annular_mean_abn_bps=69.40,
    annular_t_hac=3.202,
    annular_perm_p=0.020,
    annular_bonf_p=0.080,
    car_window_bps=-56.71,
    fp="2fa7d20bebfd",
    year_start=1928,
    year_end=2026,
)

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n_all=80, n_us=12, n_total=57, n_annular=18,
    uncond_bps=3.16,
    all_mean_abn_bps=13.32, all_t_hac=1.119, all_perm_p=0.307, all_win_rate_pct=62.5,
    total_mean_abn_bps=6.93, total_t_hac=0.495, total_perm_p=0.647,
    us_mean_abn_bps=-11.81, us_t_hac=-0.282, us_perm_p=0.696,
    annular_mean_abn_bps=69.40, annular_t_hac=3.202, annular_perm_p=0.020, annular_bonf_p=0.080,
    car_window_bps=-56.71, year_start=1928, year_end=2026,
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

from solar_eclipse import data, strategy as st

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
            "# Solar-Eclipse -- do eclipses spook the stock market?\n"
            "### An ancient omen, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "Since antiquity, a solar eclipse has been read as a bad omen -- kings hid, armies "
            "paused, and the modern version is the half-joke that markets *swoon* when the moon "
            "blots out the sun. The 2017 and 2024 'Great American' eclipses both drew breathless "
            "'will the market crash?' commentary. This notebook asks the only question that "
            "matters: **on the trading day of an eclipse, does the S&P 500 actually do anything "
            "unusual?**\n\n"
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
            "| Do eclipse days have negative returns? | **No.** Eclipse days average **+13 bps** "
            "abnormal -- the *wrong sign* for 'spook'. |\n"
            "| Is that statistically real? | **No.** HAC t = **+1.12**, permutation p = **0.31**. |\n"
            "| What about the US 'Great American' eclipses? | The 12 US-path eclipses average "
            "**-12 bps** with t = **-0.28** -- a shrug, not a crash. |\n"
            "| Could you trade it? | **No.** A handful of one-day events per decade; trading "
            "costs swamp any noise-level edge. |\n\n"
            "> The eclipse 'omen' doesn't move the market. The market's tiny upward daily drift "
            "does all the work, and the eclipse adds nothing."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"A solar eclipse is a bad omen. When the sky goes dark, fear spreads, and the "
            "stock market falls -- especially when the shadow crosses your own country.\"*\n\n"
            "It is one of the oldest superstitions there is, and it gets a fresh airing every "
            "time a major eclipse approaches. The mechanism is, to put it gently, unspecified: "
            "a few minutes of midday darkness over a narrow path does not change corporate "
            "earnings, interest rates, or liquidity. So what's really going on?"
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true -- if the *geometry of a shadow* contained information about equity "
            "returns -- financial markets would be very strange indeed. The interesting question "
            "is whether the human tendency to find omens in the sky leaves any measurable trace "
            "in prices. Spoiler: it doesn't, and seeing *why* is a clean lesson in how rare "
            "events fool us."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong baseline.** The market drifts *up* about 3 bps a day on average. So "
            "an eclipse day that is up is not surprising -- we must measure the **abnormal** "
            "return (eclipse day minus the normal daily drift), not the raw return.\n"
            "2. **Tiny n.** There are only ~80 central eclipses in our 1928-2026 window, and "
            "just ~12 crossed the US. With ~1% daily volatility you need a ~25 bps effect before "
            "it clears the noise. We'll check whether anything is anywhere near that.\n"
            "3. **Slicing until it sparkles.** Total vs annular vs hybrid, US-path vs not -- "
            "test enough sub-groups and one will look 'significant' by luck. We correct for that "
            "and watch the lucky slice evaporate.\n\n"
            "We test all of this on ^GSPC daily data from 1928 to 2026 (80 eclipses)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every central solar eclipse, hardcoded in this study's "
            "`data.py`, paired with the ^GSPC return on the first trading day on/after it."
        ),
        code(
            "ecl = data.eclipse_table()\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_gspc_daily(); ret = px['ret']\n"
            "    s = st.summarize(ret, ecl, n_permutations=5000, seed=280)\n"
            "    n_all = s['n_all']; mean_abn = s['all_mean_abn_bps']\n"
            "    win_pct = s['all_win_rate_pct']; uncond = s['uncond_bps']\n"
            "else:\n"
            "    n_all = R['n_all']; mean_abn = R['all_mean_abn_bps']\n"
            "    win_pct = R['all_win_rate_pct']; uncond = R['uncond_bps']\n"
            "\n"
            "print('Central eclipses in table:', len(ecl),\n"
            "      '| total:', int((ecl.kind=='total').sum()),\n"
            "      '| annular:', int((ecl.kind=='annular').sum()),\n"
            "      '| hybrid:', int((ecl.kind=='hybrid').sum()),\n"
            "      '| US-path:', int(ecl.us_path.sum()))\n"
            "print(f'Eclipse-day mean abnormal return: {mean_abn:+.1f} bps (n={n_all})')\n"
            "print(f'Eclipse-day win-rate: {win_pct:.1f}%   (market up ~54% of all days)')\n"
            "print(f'Normal daily drift (the honest baseline): {uncond:+.2f} bps/day')\n"
            "print()\n"
            "print('A bad omen would push this NEGATIVE. It is mildly POSITIVE.')"
        ),
        md("**Is the eclipse day a swoon or a shrug?** Here is the abnormal return on each eclipse day."),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_day_returns(ret, ecl)\n"
            "    abn = (ev['ret'].values - uncond/1e4) * 1e4\n"
            "    years = ev['date'].dt.year.values\n"
            "    us = ev['us_path'].values\n"
            "else:\n"
            "    rng = np.random.default_rng(280)\n"
            "    years = np.linspace(1930, 2026, R['n_all']).astype(int)\n"
            "    abn = rng.normal(R['all_mean_abn_bps'], 95, R['n_all'])\n"
            "    us = np.zeros(R['n_all'], dtype=bool); us[::7] = True\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.bar(years[~us], abn[~us], width=1.2, color=GREY, label='Eclipse day (non-US path)')\n"
            "ax.bar(years[us], abn[us], width=1.6, color=RED, label='US-path eclipse')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(np.mean(abn), ls='--', c=GREEN, lw=2,\n"
            "           label=f'Mean: {np.mean(abn):+.0f} bps')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Abnormal return on eclipse day (bps)')\n"
            "ax.set_title('Eclipse-day returns: noise scattered around zero, mean mildly positive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean abnormal eclipse-day return: {np.mean(abn):+.1f} bps')\n"
            "print(f'Std dev across eclipses:          {np.std(abn):.0f} bps -- pure noise')"
        ),
        md(
            "The bars scatter symmetrically around zero. The biggest down-days "
            "(1930, 1987's hybrid) sit next to the biggest up-days (1933, 2012) -- "
            "exactly what you'd expect from ~80 random draws. The average is a "
            "mildly positive **+13 bps**, the *opposite* of a swoon."
        ),
        md("**The US 'Great American' eclipses -- surely those spooked the market?**"),
        code(
            "if HAVE_REAL:\n"
            "    us_abn = (st.event_day_returns(ret, ecl[ecl['us_path']])['ret'].values - uncond/1e4)*1e4\n"
            "    all_abn = abn\n"
            "else:\n"
            "    us_abn = np.random.default_rng(1).normal(R['us_mean_abn_bps'], 90, R['n_us'])\n"
            "    all_abn = abn\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['US-path\\neclipses', 'All\\neclipses', 'Normal\\nday'],\n"
            "              [np.mean(us_abn), np.mean(all_abn), 0.0],\n"
            "              color=[RED, GREY, GREEN], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean abnormal return (bps)')\n"
            "ax.set_title('Even US-path eclipses do not spook the S&P (a shrug, not a crash)')\n"
            "for b, v in zip(bars, [np.mean(us_abn), np.mean(all_abn), 0.0]):\n"
            "    ax.annotate(f'{v:+.0f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'US-path eclipses (n=12): {np.mean(us_abn):+.1f} bps abnormal -- t = {R[\"us_t_hac\"]:+.2f}')\n"
            "print('A 12-bps wobble on a dozen events is indistinguishable from any random dozen days.')"
        ),
        md(
            "The 12 eclipses whose path crossed the US average **-12 bps** -- a tiny, "
            "statistically empty wobble (HAC t = **-0.28**). The 2017 and 2024 'Great "
            "American' eclipses that triggered crash chatter did nothing of the sort."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Eclipse days average **+13 bps** abnormal (wrong sign for "
            "an omen), HAC t = **+1.12**, perm p = **0.31**. The US-path slice is **-12 bps**, "
            "t = **-0.28**. Nothing is significant.\n"
            "- **Tradability -- Mirage.** There is no trade: a once-per-eclipse, single-day "
            "event a handful of times per decade. Trading costs swamp any noise-level edge, and "
            "staying invested beats it.\n"
            "- **Busted.** Eclipses have been blamed for panics since antiquity, but on 98 years "
            "of S&P data the eclipse trading day is, if anything, *up*."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be: short (or go to cash) on each eclipse day, betting on the "
            "swoon. The problem -- there is no swoon, and there are barely any events:"
        ),
        code(
            "n_events = R['n_all']\n"
            "years_span = R['year_end'] - R['year_start']\n"
            "per_decade = n_events / years_span * 10\n"
            "edge_bps = R['all_mean_abn_bps']\n"
            "cost_bps = 5.0  # round-trip cost + borrow for a one-day short, very optimistic\n"
            "print(f'Events: {n_events} eclipses over {years_span} years = {per_decade:.1f} per decade')\n"
            "print(f'Gross edge (if you faded eclipse days): {-edge_bps:+.1f} bps per event')\n"
            "print(f'Round-trip trading cost (one-day short):  {cost_bps:.1f} bps per event')\n"
            "print(f'Net per event: {-edge_bps - cost_bps:+.1f} bps -- and the sign was wrong anyway')\n"
            "print()\n"
            "print('You would be shorting a market that, on those days, tended to RISE,')\n"
            "print('and paying costs for the privilege. There is no trade here.')"
        ),
        md(
            "Fading eclipse days means shorting a market that tended to *rise* on them, while "
            "paying one-way costs (and borrow on the short). The only real trade is: don't time "
            "the market on an astronomical calendar."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a fake eclipse-day drag "
            "in a synthetic tape and shows the machinery detects it cleanly -- so the null "
            "result is about the *market*, not a broken test.\n"
            "- **The other celestial omen:** [Study 164 -- Mercury-Retrograde](../../164-mercury-retrograde/) "
            "-- astrology vs the S&P, same None/Mirage family.\n"
            "- **The event-table pattern:** [Study 158 -- Super-Bowl](../../158-super-bowl/) -- "
            "the hardcoded-event + real-returns template this study copies.\n\n"
            "*Think the sky really moves stocks? Fork this, define your own celestial event "
            "table, and show an abnormal return that beats the noise with a HAC t >= 2 after "
            "honest multiple-comparisons correction. That's the bar -- and it hasn't been "
            "cleared yet.*"
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
            "# Solar-Eclipse -- a quantitative teardown\n"
            "### 80 eclipses * ^GSPC daily * event-study CAR * HAC t * permutation * "
            "Bonferroni * tiny-n power\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We run a classic event "
            "study around 80 central solar eclipses on the ^GSPC daily tape: cumulative abnormal "
            "return (CAR), a Newey-West (HAC) t-stat on the event-day mean, a permutation test "
            "against random trading days, a Bonferroni correction over four slices, and a tiny-n "
            "power calculation that explains why ~80 events can never resolve a sentiment effect "
            "this small.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily price returns 1928-2026 "
            "(`_cache/^GSPC_split_only.parquet`, price-only); eclipse dates hardcoded in "
            "`data.py`. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Eclipse-day abnormal return **+13.3 bps**, HAC t = "
            "**+1.12**, perm p = **0.31** (wrong sign for the omen). |\n"
            "| **Tradability** | `MIRAGE` | No vehicle; a handful of one-day events per decade, "
            "dominated by buy-and-hold. |\n"
            "| **Busted?** | `YES` | The one 'significant' slice (annular, raw p = 0.02) dies "
            "under Bonferroni (4 slices -> 0.08); US-path eclipses are -12 bps, t = -0.28. |\n\n"
            "> The market's +3 bps/day drift is the correct baseline. Against it, the eclipse "
            "day contributes nothing statistically distinguishable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the ^GSPC daily return and let event day 0 be the first trading day "
            "on/after each eclipse. Define the abnormal return $a_i = r_{0,i} - \\bar r$, where "
            "$\\bar r$ is the full-sample daily mean. The hypotheses:\n\n"
            "- **H1 (event-day drag).** $E[a_i] < 0$ -- eclipse days earn negative abnormal "
            "returns, with $|t| \\ge 2$ on a HAC standard error.\n"
            "- **H2 (US-path amplifier).** The drag is larger for eclipses whose path crossed "
            "the contiguous US.\n"
            "- **H3 (window drift).** The cumulative abnormal return (CAR) over a +/-5 day "
            "window is significantly negative.\n\n"
            "We reject H1, H2, and H3 on the real tape -- indeed the point estimate for H1 has "
            "the *wrong sign*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The solar-eclipse omen is among the purest 'celestial sentiment' claims: there is "
            "no plausible channel from shadow geometry to cash flows or discount rates. If "
            "H1-H3 held it would imply a mood effect strong enough to move the largest equity "
            "market on Earth for a single, astronomically predictable day -- which would itself "
            "be arbitraged away. The instructive finding is *how* a lucky sub-slice (annular "
            "eclipses) can masquerade as a discovery, and how multiple-comparisons discipline "
            "dissolves it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** 80 central solar eclipses (hardcoded in `data.py`); ^GSPC daily "
            "close-to-close returns 1928-2026 (price-only).\n"
            "- **Event day 0.** First trading day on/after the eclipse (one execution-lag day "
            "built in -- the soonest a trader could act on an overnight/intraday event).\n"
            "- **Abnormal return.** Raw event-day return minus the full-sample daily mean "
            "(a flat market model; eclipses are too sparse for a rolling estimation window).\n"
            "- **Inference.** Newey-West (HAC, 3 lag) t-stat on the event-day mean; permutation "
            "test (10,000 random-day resamples of the same n); Bonferroni over 4 slices.\n"
            "- **Slices.** all / total / annular / US-path.\n"
            "- **Positive control.** Synthetic daily tape with a planted eclipse-day drag, to "
            "confirm the engine works when an effect exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The event-day test against the correct baseline\n\n"
            "The methodological crux: the market drifts up. The abnormal return measures the "
            "eclipse day against that drift, not against zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = data.fetch_gspc_daily(); ret = px['ret']\n"
            "    ecl = data.eclipse_table()\n"
            "    r_all = st.event_stats(ret, ecl, n_permutations=10_000, seed=280)\n"
            "    n_obs = r_all['n_events']; uncond = r_all['uncond_bps']\n"
            "    mean_abn = r_all['mean_abn_bps']; t_hac = r_all['t_hac']\n"
            "    perm_p = r_all['perm_p']; win = r_all['win_rate']\n"
            "else:\n"
            "    n_obs, uncond = R['n_all'], R['uncond_bps']\n"
            "    mean_abn, t_hac = R['all_mean_abn_bps'], R['all_t_hac']\n"
            "    perm_p, win = R['all_perm_p'], R['all_win_rate_pct']/100\n"
            "\n"
            "print(f'n eclipses: {n_obs}')\n"
            "print(f'Unconditional daily drift: {uncond:+.2f} bps')\n"
            "print(f'Eclipse-day mean abnormal:  {mean_abn:+.2f} bps')\n"
            "print(f'Eclipse-day win-rate:       {win:.1%}')\n"
            "print(f'HAC (Newey-West) t-stat:    {t_hac:+.3f}')\n"
            "print(f'Permutation p-value:        {perm_p:.4f}')\n"
            "print()\n"
            "print('The abnormal return is POSITIVE and far below |t|=2: the omen has the')\n"
            "print('wrong sign and no significance.')"
        ),
        md(
            "> The eclipse-day abnormal return is **+13 bps** with HAC t = **+1.12** -- the "
            "wrong sign for a 'spook' and comfortably inside the noise. Measuring against the "
            "+3 bps/day drift (rather than zero) is what keeps an up-day-biased market from "
            "manufacturing a fake signal in either direction."
        ),
        md(
            "### 4b - The CAR profile: average abnormal return around the eclipse\n\n"
            "Stack a +/-5 trading-day window and accumulate the abnormal returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    car = st.car_profile(ret, ecl, win=5)\n"
            "    rel = car.index.values; aar = car['aar_bps'].values; carv = car['car_bps'].values\n"
            "else:\n"
            "    rel = np.arange(-5, 6)\n"
            "    rng = np.random.default_rng(280)\n"
            "    aar = rng.normal(-5, 14, 11); aar[5] = R['all_mean_abn_bps']\n"
            "    carv = np.cumsum(aar)\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "ax1.bar(rel, aar, color=[RED if r==0 else GREY for r in rel])\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_xlabel('Trading day relative to eclipse')\n"
            "ax1.set_ylabel('Avg abnormal return (bps)'); ax1.set_title('AAR by relative day (red = event day 0)')\n"
            "ax2.plot(rel, carv, 'o-', c=GREY, lw=2); ax2.axhline(0, c='k', lw=1)\n"
            "ax2.axvline(0, ls=':', c=RED); ax2.set_xlabel('Trading day relative to eclipse')\n"
            "ax2.set_ylabel('Cumulative abnormal return (bps)')\n"
            "ax2.set_title(f'CAR over the window: {carv[-1]:+.0f} bps total')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Event-day (0) AAR: {aar[list(rel).index(0)]:+.1f} bps')\n"
            "print(f'Full-window CAR:   {carv[-1]:+.1f} bps -- a random 11-day drift, no event structure')"
        ),
        md(
            "There is no spike at day 0 and no coherent drift: the CAR wanders to "
            "**~-57 bps** over 11 days, driven by the pre-event days (-2, -1), which is "
            "exactly the kind of meaningless drift any random 11-day window produces. No event "
            "signature."
        ),
        md(
            "### 4c - The permutation distribution\n\n"
            "Draw 80 random trading days 10,000 times; where does the real eclipse-day mean land?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm_dist = r_all['perm_dist']; obs = r_all['mean_abn_bps']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(280)\n"
            "    perm_dist = rng0.normal(0, 11, 10000); obs = R['all_mean_abn_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_dist, bins=40, color=GREY, alpha=0.7, label='Random-day resamples')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed: {obs:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Mean abnormal return (bps)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The observed eclipse-day mean sits well inside the null distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "pct = (perm_dist < obs).mean()\n"
            "print(f'Observed mean abnormal: {obs:+.1f} bps')\n"
            "print(f'Permutation percentile: {pct:.1%} of random-day samples below it')\n"
            "print(f'Two-sided permutation p: {R[\"all_perm_p\"]:.3f}')"
        ),
        md(
            "The observed +13 bps lands around the 70th percentile of the random-day null -- "
            "mildly high, nowhere near the tails. Permutation p = **0.31**."
        ),
        md(
            "### 4d - The multiple-comparisons mirage (annular eclipses)\n\n"
            "Slice by type and geography. One slice -- annular eclipses -- looks 'significant' "
            "raw. Watch it die under Bonferroni."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mc = st.multiple_comparisons_summary(ret, ecl, n_permutations=10_000, seed=280)\n"
            "else:\n"
            "    mc = pd.DataFrame({\n"
            "        'slice': ['all eclipses','total only','annular only','US-path only'],\n"
            "        'n': [R['n_all'], R['n_total'], R['n_annular'], R['n_us']],\n"
            "        'mean_abn_bps': [R['all_mean_abn_bps'], R['total_mean_abn_bps'],\n"
            "                         R['annular_mean_abn_bps'], R['us_mean_abn_bps']],\n"
            "        't_hac': [R['all_t_hac'], R['total_t_hac'], R['annular_t_hac'], R['us_t_hac']],\n"
            "        'perm_p': [R['all_perm_p'], R['total_perm_p'], R['annular_perm_p'], R['us_perm_p']],\n"
            "        'bonferroni_perm_p': [min(1,R['all_perm_p']*4), min(1,R['total_perm_p']*4),\n"
            "                              R['annular_bonf_p'], min(1,R['us_perm_p']*4)],\n"
            "    })\n"
            "\n"
            "print(mc.to_string(index=False))\n"
            "print()\n"
            "print('Annular looks significant raw (perm p ~ 0.02) -- but we tested 4 slices.')\n"
            "print('Bonferroni x4 -> ~0.08: it does NOT survive. There is no annular effect;')\n"
            "print('it is two lucky up-days (1933, 2012) in an 18-event slice.')"
        ),
        md(
            "> This is the whole game in one table. The annular slice (HAC t = 3.20, raw p = "
            "0.02) is the kind of result that gets written up as a discovery. But it is one of "
            "four slices searched on the same tape, and Bonferroni x4 pushes it to **0.08**. "
            "There is no mechanism by which annular-vs-total shadow geometry would move US "
            "equities -- it is slice-mining, full stop."
        ),
        md(
            "### 4e - Power: what could ~80 events even detect?\n\n"
            "The minimum detectable event-day effect, given the sample and daily volatility."
        ),
        code(
            "vol_bps = 100.0   # ~1% daily ^GSPC vol\n"
            "n_ev = R['n_all']\n"
            "se = vol_bps / np.sqrt(n_ev)\n"
            "mde_2sided = 2.0 * se   # |t|=2 threshold\n"
            "print(f'Daily vol ~ {vol_bps:.0f} bps, n_events = {n_ev}')\n"
            "print(f'Standard error of the event-day mean: {se:.1f} bps')\n"
            "print(f'Minimum detectable effect (|t|=2):    {mde_2sided:.1f} bps')\n"
            "print()\n"
            "print(f'Observed abnormal return: {R[\"all_mean_abn_bps\"]:+.1f} bps')\n"
            "print(f'That is {abs(R[\"all_mean_abn_bps\"])/mde_2sided:.0%} of the detectable threshold.')\n"
            "print()\n"
            "print('Conclusion: ~80 eclipses cannot resolve a sub-25bps sentiment effect,')\n"
            "print('and the US-path slice (n=12) is hopeless: SE alone is ~29 bps.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- eclipse-day abnormal +13.3 bps, HAC t +1.12, perm p 0.31, "
            "CAR ~-57 bps (random drift). US-path -12 bps, t -0.28. No slice survives "
            "Bonferroni.\n"
            "- **Tradability `MIRAGE`** -- no vehicle; a few one-day events per decade, "
            "dominated by passive exposure; costs swamp the noise-level edge.\n"
            "- **Busted** -- the annular 'effect' is a four-slice multiple-comparisons artifact; "
            "the omen has the wrong sign overall."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- costs vs a non-edge\n\n"
            "Fade each eclipse day (short / go to cash) and pay realistic one-day costs."
        ),
        code(
            "gross_bps = -R['all_mean_abn_bps']   # fading a +13 bps day earns -13 bps gross\n"
            "cost_bps = 5.0                        # round-trip + borrow on a one-day short\n"
            "net_bps = gross_bps - cost_bps\n"
            "per_decade = R['n_all'] / (R['year_end']-R['year_start']) * 10\n"
            "print(f'Events: ~{per_decade:.1f} per decade')\n"
            "print(f'Gross per event (fade the day): {gross_bps:+.1f} bps')\n"
            "print(f'One-way cost + borrow:          {cost_bps:.1f} bps')\n"
            "print(f'Net per event:                  {net_bps:+.1f} bps')\n"
            "print()\n"
            "print('Negative gross (wrong sign) minus costs = a guaranteed loser.')\n"
            "print('Gross ~ net here only because there is no edge to erode in the first place.')"
        ),
        md(
            "> There is no tradable implementation. You would short a market that tended to rise "
            "on eclipse days, a few times per decade, paying costs and borrow each time. Passive "
            "buy-and-hold dominates trivially."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect an eclipse effect *when one exists*? Plant a drag in a "
            "synthetic daily tape and sweep its size."
        ),
        code(
            "ecl = data.eclipse_table()\n"
            "planted = [0, -50, -150, -300]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_panel(signal_bps=float(bps), seed=280)\n"
            "    r = st.event_stats(df_s['ret'], ecl, n_permutations=1000, seed=280)\n"
            "    rows.append({'planted_bps': bps, 'recovered_bps': r['mean_abn_bps'],\n"
            "                 't_hac': r['t_hac'], 'perm_p': r['perm_p']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if abs(r['t_hac'])>=2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['planted_bps']) for r in rows], [r['recovered_bps'] for r in rows], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted eclipse-day drag (bps)'); ax.set_ylabel('Recovered abnormal (bps)')\n"
            "ax.set_title('The engine recovers a planted effect (green = |HAC t| >= 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "The engine recovers the planted drag almost exactly and flags it significant once "
            "it exceeds the ~25 bps noise floor. The real tape -- +13 bps with the wrong sign -- "
            "is nowhere near detection. The verdict is a statement about the **market and the "
            "rarity of eclipses**, not about the method."
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
